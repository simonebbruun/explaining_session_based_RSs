import pandas as pd
from pickle import load
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from integrated_gradients import integrated_gradients
from pickle import dump
import time


dataset = '30music'
rec_model = 'gru4rec'


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
    for l in range(len(test_data)):
        n_steps.append(len(test_data[l])-1)
        test_data[l] = np.expand_dims(np.expand_dims(test_data[l][:-1], axis=0), axis=-1)
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

    one_hot_layer = model_0.get_layer('time_distributed')
    
    model = tf.keras.Sequential()
    for layer in model_0.layers[1:]:
        model.add(layer)
    
    for i in range(len(test_data)):
        test_data[i] = one_hot_layer(test_data[i]).numpy()
    
    model.build((None,None,test_data[0].shape[-1]))
else:
    model = load_model('sasrec_model_'+dataset+'.h5')

    embedding_model = model.get_layer('model')
    model = model.get_layer('model_1')

    test_data = embedding_model.predict(test_data)

    top_idxs = np.argmax(model.predict(test_data), axis=1)


''' Explanations. '''
start = time.time()
if rec_model == 'gru4rec':
    ig_explanations = []
    top_idxs = []
    for i in range(len(test_data)):
        instance = tf.convert_to_tensor(test_data[i])
        baseline = tf.zeros(shape=tf.shape(instance))
        model_1 = tf.keras.models.clone_model(model)
        model_1.set_weights(model.get_weights())
        model_1.add(tf.keras.layers.Cropping1D(cropping=(n_steps[i]-1,0)))
        model_1.add(tf.keras.layers.Reshape((model_1.output_shape[-1],)))
        top_idx = tf.math.top_k(input=model_1(instance), k=1)[1][0][0]
    
        ig_explanation = integrated_gradients(baseline=baseline, instance=instance, model=model_1, target_class_idx=top_idx, batch_size=16)
        
        ig_explanations.append(tf.reduce_sum(ig_explanation, axis=-1).numpy()[0])
        top_idxs.append(top_idx.numpy())
else:
    ig_explanations = []
    for i in range(len(test_data)):
        instance = tf.convert_to_tensor(test_data[i])
        baseline = tf.zeros(shape=tf.shape(instance))

        ig_explanation = integrated_gradients(baseline=baseline, instance=instance, model=model, target_class_idx=top_idxs[i], batch_size=16)
        
        ig_explanations.append(tf.reduce_sum(ig_explanation, axis=-1).numpy())
    ig_explanations = np.stack(ig_explanations)
end = time.time()
print(end - start)


dump(ig_explanations, open('ig_explanation_'+rec_model+dataset+'.npy', 'wb'))
dump(top_idxs, open('ig_class_'+rec_model+dataset+'.npy', 'wb'))