import pandas as pd
from pickle import load
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import evaluation_functions as ef


dataset = '30music'
rec_model = 'sasrec'
expl_method = 'sb_occlusion'


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
    
    n_pads = test_data.shape[1]-np.array(n_steps)
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


''' Analysis. '''
model = load_model(rec_model+'_model_'+dataset+'.h5')

if rec_model == 'gru4rec':
    model.add(tf.keras.layers.Cropping1D(cropping=(test_data.shape[1]-1,0)))
    model.add(tf.keras.layers.Reshape((model.output_shape[-1],)))

explanation = load(open(expl_method+'_explanation_'+rec_model+dataset+'.npy', 'rb'))
classes = load(open(expl_method+'_class_'+rec_model+dataset+'.npy', 'rb'))
indices = load(open('indices_'+rec_model+dataset+'.npy', 'rb'))


if expl_method == 'sb_occlusion':
    explanation_1 = explanation
elif (rec_model == 'gru4rec' and expl_method == 'deep_shap') or (rec_model == 'sasrec' and expl_method in ['deep_shap', 'ig']):
    explanation_1 = []
    for i in range(len(explanation)):
        instance = explanation[i]
        explanation_dict = {}
        for index in indices[i]:
            explanation_dict[index] = np.sum(instance[[j for j in index]])
        explanation_1.append(explanation_dict)
else:
    explanation_1 = []
    for i in range(len(explanation)):
        instance = np.array(explanation[i])
        explanation_dict = {}
        for index in indices[i]:
            explanation_dict[index] = np.sum(instance[[(j-n_pads[i]) for j in index]])
        explanation_1.append(explanation_dict)


''' Evaluation. '''
if rec_model == 'gru4rec':
    padding = -1
else:
    padding = 0
    
n_pads_1 = np.copy(n_pads)
n_pads_1[:] = 0
attribution, diffs = ef.pred_diff(explanation_1, test_data, np.array(n_pads_1), model, pad_value=padding, class_index=classes)

rank_correlation = ef.rank_correlation(attribution, diffs)
print(np.mean(np.nan_to_num(rank_correlation)))
attribute_correlation = ef.attribute_correlation(attribution, diffs)
print(np.mean(np.nan_to_num(attribute_correlation)))
