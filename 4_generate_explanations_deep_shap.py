import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pickle import load
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from tensorflow.keras.models import load_model
import shap
from pickle import dump
import time


dataset = 'diginetica'
rec_model = 'sasrec'


''' Background data. '''
# Import.
background_data = pd.read_csv(dataset+'_train.csv')


# Ordinal encode.
encoder = OrdinalEncoder(dtype='int32', handle_unknown='use_encoded_value', unknown_value=-1)

if rec_model == 'gru4rec':
    background_data['item_id'] = encoder.fit_transform(background_data[['item_id']])

    # Padding.
    background_data = background_data.sort_values(['timestamp', 'session_id']).reset_index(drop=True)
    background_data = background_data.drop(['timestamp'], axis=1)
    
    background_data = list(background_data.groupby('session_id', sort=False)['item_id'].apply(list))
    
    background_data = tf.keras.utils.pad_sequences(background_data, value=-1)
    background_data = np.expand_dims(background_data, axis=-1)
    
    background_data = background_data[-100:,:-1,:]
else:
    background_data['item_id'] = encoder.fit_transform(background_data[['item_id']])+1

    # Aggregating features.
    session_col = 'session_id'
    behavior_key_col = 'item_id'

    background_data = background_data.sort_values(['timestamp', 'session_id']).reset_index(drop=True)
    background_data = background_data.drop(['timestamp'], axis=1)

    background_data = list(background_data.groupby('session_id', sort=False)['item_id'].apply(list))
    for i in range(len(background_data)):
            background_data[i] = background_data[i][:-1]
    background_data = tf.keras.utils.pad_sequences(background_data, padding='pre')
    background_data = background_data[-100:]


''' Test data. '''
# Import.
test_data = pd.read_csv(dataset+'_test.csv')


# Ordinal encode.
encoder = load(open('ordinal_encoder_'+dataset+'.pkl', 'rb'))

if rec_model == 'gru4rec':
    test_data['item_id'] = encoder.transform(test_data[['item_id']])

    # Padding.
    test_data = test_data.sort_values(['timestamp', 'session_id']).reset_index(drop=True)
    test_data = test_data.drop(['timestamp'], axis=1)
    
    test_data = list(test_data.groupby('session_id', sort=False)['item_id'].apply(list))
    
    n_steps = []
    for i in range(len(test_data)):
            n_steps.append(len(test_data[i])-1)
    
    test_data = tf.keras.utils.pad_sequences(test_data, value=-1)
    test_data = np.expand_dims(test_data, axis=-1)
    
    test_data = test_data[:,:-1,:]
else:
    test_data['item_id'] = encoder.transform(test_data[['item_id']])+1

    # Aggregating features.
    seq_max_len = np.max(test_data.groupby('session_id').size())

    session_col = 'session_id'
    behavior_key_col = 'item_id'

    test_data = test_data.sort_values(['timestamp', 'session_id']).reset_index(drop=True)
    test_data = test_data.drop(['timestamp'], axis=1)

    test_data = list(test_data.groupby('session_id', sort=False)['item_id'].apply(list))
    n_pads = []
    for i in range(len(test_data)):
            n_pads.append(seq_max_len-len(test_data[i]))
            test_data[i] = test_data[i][:-1]
    test_data = tf.keras.utils.pad_sequences(test_data, padding='pre')


''' Model. '''
if rec_model == 'gru4rec':
    model_0 = load_model('gru4rec_model_'+dataset+'.h5')
    model_0.add(tf.keras.layers.Cropping1D(cropping=(background_data.shape[1]-1,0)))
    model_0.add(tf.keras.layers.Reshape((model_0.output_shape[-1],)))
    
    one_hot_layer = model_0.get_layer('time_distributed')
    
    model = tf.keras.Sequential()
    for layer in model_0.layers[1:]:
        model.add(layer)
    
    test_data = one_hot_layer(test_data).numpy()
    background_data = one_hot_layer(background_data).numpy()
    
    model.predict(background_data)
    
    n_pads = test_data.shape[1]-np.array(n_steps)
else:
    model = load_model('sasrec_model_'+dataset+'.h5')

    embedding_model = model.get_layer('model')
    model = model.get_layer('model_1')

    background_data = embedding_model.predict(background_data)
    test_data = embedding_model.predict(test_data)


''' Explanations. '''
shap_method = shap.DeepExplainer(model, background_data)

start = time.time()
shap_explanation = shap_method.shap_values(test_data, ranked_outputs=1, check_additivity=False)
end = time.time()
print(end - start)

shap_class = shap_explanation[1][:,0]

shap_explanation = shap_explanation[0][0]
shap_explanation = np.sum(shap_explanation, axis=2)

for i in range(len(shap_explanation)):
    shap_explanation[i,:n_pads[i]] = 0


dump(shap_explanation, open('deep_shap_explanation_'+rec_model+dataset+'.npy', 'wb'))
dump(shap_class, open('deep_shap_class_'+rec_model+dataset+'.npy', 'wb'))

