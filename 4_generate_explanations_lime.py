import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pickle import load
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from lime import lime_tabular
from pickle import dump
import time


dataset = '30music'
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
    
    background_data = background_data[:,:-1,:]
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
    background_data = np.expand_dims(background_data, axis=-1)


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
    test_data = np.expand_dims(test_data, axis=-1)


''' Model. '''
model = load_model(rec_model+'_model_'+dataset+'.h5')

if rec_model == 'gru4rec':
    model.add(tf.keras.layers.Cropping1D(cropping=(background_data.shape[1]-1,0)))
    model.add(tf.keras.layers.Reshape((model.output_shape[-1],)))



''' Explanations. '''
lime_explainer = lime_tabular.RecurrentTabularExplainer(background_data, feature_names = ['item'], categorical_features=range(background_data.shape[1]), random_state=42)


def explanation_transform(x_list):
    x_df = pd.DataFrame(x_list, columns=['feature', 'value'])
    x_df[['feature', 'timestep']] = x_df['feature'].str.split('_', n=1, expand=True)
    x_df[['timestep', 'action']] = x_df['timestep'].str.split('=', n=1, expand=True)
    x_df['timestep'] = x_df['timestep'].str.extract('(\d+)').astype(int)
    x_df = x_df.sort_values('timestep', ascending=False).reset_index(drop=True)
    return x_df


def model_predict(x):
    if rec_model == 'gru4rec':
        preds = model.predict(x)
    else:
        preds = model.predict(x[:,:,0])
    preds = np.exp(preds)/(np.sum(np.exp(preds),axis=1)[:,None])
    return preds

lime_class = np.argmax(model_predict(test_data), axis=1)


start = time.time()
lime_explanations = []
for i in range(len(test_data)):
    if rec_model == 'gru4rec':
        lime_explanation = lime_explainer.explain_instance(test_data[i:(i+1)], model_predict, top_labels=1, num_features=(n_steps[i])) 
        lime_explanation = lime_explanation.as_list(lime_class[i])
        lime_explanations.append(list(explanation_transform(lime_explanation)['value']))
    else:
        lime_explanation = lime_explainer.explain_instance(test_data[i:(i+1)], model_predict, top_labels=1, num_features=(seq_max_len-1)) 
        lime_explanation = lime_explanation.as_list(lime_class[i])
        lime_explanations.append(list(explanation_transform(lime_explanation)['value'])[n_pads[i]:])
end = time.time()
print(end - start)



dump(lime_explanations, open('lime_explanation_'+rec_model+dataset+'.npy', 'wb'))
dump(lime_class, open('lime_class_'+rec_model+dataset+'.npy', 'wb'))


