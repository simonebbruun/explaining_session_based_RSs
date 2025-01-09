import pandas as pd
from pickle import load
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import evaluation_functions as ef
import analysis_functions as af


dataset = '30music'
rec_model = 'sasrec'
expl_method = 'lime'


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


''' Analysis. '''
model = load_model(rec_model+'_model_'+dataset+'.h5')

if rec_model == 'gru4rec':
    model.add(tf.keras.layers.Cropping1D(cropping=(test_data.shape[1]-1,0)))
    model.add(tf.keras.layers.Reshape((model.output_shape[-1],)))

explanation = load(open(expl_method+'_explanation_'+rec_model+dataset+'.npy', 'rb'))
classes = load(open(expl_method+'_class_'+rec_model+dataset+'.npy', 'rb'))


if rec_model == 'gru4rec':
    explanation_1 = []
    for i in range(len(explanation)):
        instance = explanation[i]
        explanation_dict = {}
        for j in range(len(instance)):
            explanation_dict[(j,)] = instance[j]
        explanation_1.append(explanation_dict)
    n_pads = test_data.shape[1]-np.array(n_steps)
elif rec_model == 'sasrec' and expl_method=='lime':
    explanation_1 = []
    for i in range(len(explanation)):
        instance = explanation[i]
        explanation_dict = {}
        for j in range(len(instance)):
            explanation_dict[(j,)] = instance[j]
        explanation_1.append(explanation_dict)
else:
    explanation_1 = []
    for i in range(len(explanation)):
        instance = explanation[i,n_pads[i]:]
        explanation_dict = {}
        for j in range(len(instance)):
            explanation_dict[(j,)] = instance[j]
        explanation_1.append(explanation_dict)

    
''' Analysis of repeated actions. '''
if rec_model == 'gru4rec':
    padding = -1
else: 
    padding = 0
    
    
repeated_split = af.repeated_split(test_data, n_pads, pad_value=padding)


# Evaluation over repeated actions only.
attribution_repeated, n_repeated = af.select_repeated(np.array(explanation_1)[repeated_split], test_data[repeated_split], np.array(n_pads)[repeated_split], pad_value=padding)
attribution_repeated, diffs_repeated = ef.pred_diff(attribution_repeated, test_data[repeated_split], np.array(n_pads)[repeated_split], model, pad_value=padding, class_index=np.array(classes)[repeated_split])
rank_correlation_repeated = ef.rank_correlation(attribution_repeated, diffs_repeated)
attribute_correlation_repeated = ef.attribute_correlation(attribution_repeated, diffs_repeated)

print(np.mean(np.nan_to_num(rank_correlation_repeated)))
print(np.mean(np.nan_to_num(attribute_correlation_repeated)))


# Evaluation over random actions.
mean_rank_correlation_random = []
mean_attribute_correlation_random = []
for i in range(5):
    attribution_random = af.select_random(np.array(explanation_1)[repeated_split], test_data[repeated_split], n_repeated, seed=i)
    attribution_random, diffs_random = ef.pred_diff(attribution_random, test_data[repeated_split], np.array(n_pads)[repeated_split], model, pad_value=padding, class_index=np.array(classes)[repeated_split])
    rank_correlation_random = ef.rank_correlation(attribution_random, diffs_random)
    attribute_correlation_random = ef.attribute_correlation(attribution_random, diffs_random)

    mean_rank_correlation_random.append(np.mean(np.nan_to_num(rank_correlation_random)))
    mean_attribute_correlation_random.append(np.mean(np.nan_to_num(attribute_correlation_random)))

print(np.mean(mean_rank_correlation_random))
print(np.mean(mean_attribute_correlation_random))


''' Analysis of sequential dependencies. '''
attribution, diffs = ef.pred_diff(explanation_1, test_data, np.array(n_pads), model, pad_value=padding, class_index=classes)

rank_correlation = ef.rank_correlation(attribution, diffs)
print(np.mean(np.nan_to_num(rank_correlation)))
attribute_correlation = ef.attribute_correlation(attribution, diffs)
print(np.mean(np.nan_to_num(attribute_correlation)))


sequentials_max, sequentials_min = af.sequential_dependencies(test_data, np.array(n_pads), model, classes)


# Swap the two actions with strongest sequential dependency.
attribution_swapped, test_data_swapped = af.swap_actions(explanation_1, test_data, np.array(n_pads), sequentials_max)

attribution_swapped, diffs = ef.pred_diff(attribution_swapped, test_data_swapped, np.array(n_pads), model, pad_value=padding, class_index=classes)

rank_correlation = ef.rank_correlation(attribution_swapped, diffs)
print(np.mean(np.nan_to_num(rank_correlation)))
attribute_correlation = ef.attribute_correlation(attribution_swapped, diffs)
print(np.mean(np.nan_to_num(attribute_correlation)))


# Swap the two actions with weakest sequential dependency.
attribution_swapped, test_data_swapped = af.swap_actions(explanation_1, test_data, np.array(n_pads), sequentials_min)

attribution_swapped, diffs = ef.pred_diff(attribution_swapped, test_data_swapped, np.array(n_pads), model, pad_value=padding, class_index=classes)

rank_correlation = ef.rank_correlation(attribution_swapped, diffs)
print(np.mean(np.nan_to_num(rank_correlation)))
attribute_correlation = ef.attribute_correlation(attribution_swapped, diffs)
print(np.mean(np.nan_to_num(attribute_correlation)))
