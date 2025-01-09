import pandas as pd
from pickle import load
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
import itertools
import random
import time
from pickle import dump
import explain_functions as ef


dataset = 'diginetica'
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

    n_steps = test_data.shape[1]


''' Model. '''
model = load_model(rec_model+'_model_'+dataset+'.h5')

if rec_model == 'gru4rec':
    model.add(tf.keras.layers.Cropping1D(cropping=(test_data.shape[1]-1,0)))
    model.add(tf.keras.layers.Reshape((model.output_shape[-1],)))


predictions = model.predict(test_data)
occlusion_class = np.argmax(predictions, axis=1)
predictions = np.max(predictions, axis=1)


if rec_model == 'gru4rec':
    padding = -1
else:
    padding = 0


def sample_combinations(choices, size, count):
    collected = {tuple(random.sample(choices, size)) for _ in range(count)}
    while len(collected) < count:
        collected.add(tuple(random.sample(choices, size)))
    return list(collected)

def generate_combinations_itertools(nums, n_samples):
    result = []
    n = len(nums)
    
    n_combinations = []
    for r in range(1,(n+1)):
        n_combinations.append(math.comb(n, r))
    total_combinations = np.sum(n_combinations, dtype='int64')
    n_combinations = np.cumsum(n_combinations, dtype='int64')
    
    if total_combinations <= n_samples:
        for r in range(1, (n+1)):
            combinations = itertools.combinations(nums, r)
            result.extend(combinations)
    else:
        samples = random.sample(range(1,total_combinations+1), n_samples)
        n_sample_combinations = np.zeros(n, dtype=int)
        for s in samples:
            n_sample_combinations[np.min(np.where(s<=n_combinations))]+=1
        for r in range(1, (n+1)):
            combinations = sample_combinations(nums, r, n_sample_combinations[r-1])
            result.extend(combinations)

    return result


start = time.time()
sequentials = ef.sequential_dependencies(test_data=test_data, model=model, class_index=occlusion_class, n_pads=n_pads, threshold=2, pad_value=padding)

sequential_indices = []
for i in range(len(sequentials)):
    sequential_indices.append(np.array([item for t in sequentials[i] for item in t]))

repeated = ef.repeated_items(test_data, n_pads, sequential_indices)

repeated_indices = []
for i in range(len(repeated)):
    repeated_indices.append(np.array([item for t in repeated[i] for item in t]))

indices = []
for i in range(len(n_pads)):
    index = []
    for j in range(n_pads[i],test_data.shape[1]):
        if j not in sequential_indices[i] and j not in repeated_indices[i]:
            index.append((j,))
    indices.append(index+sequentials[i]+repeated[i])

n_sequentials = [len(x) for x in sequential_indices]
n_repeated = []
for i in range(len(repeated_indices)):
    if len(repeated_indices[i]) == 1:
        n_repeated.append(0)
    else:
        n_repeated.append(len(repeated_indices[i]))


n_samples = 2000
random.seed(42)

occlusion_explanations = []
for i in range(len(test_data)):
    index = indices[i]
    mask_indices = generate_combinations_itertools(index, n_samples)
    
    instances_masked = []
    masks = []
    for j in range(len(mask_indices)):
        instance_masked = np.copy(test_data[i:(i+1)])
        instance_masked[:,[item for t in mask_indices[j] for item in t],...] = padding
        mask = np.zeros(instance_masked.shape[0:2])
        mask[:,[item for t in mask_indices[j] for item in t]] = 1
        instances_masked.append(instance_masked)
        masks.append(mask[0,:])
    instances_masked = np.vstack(instances_masked)
    masks = np.stack(masks)
    pred_diffs = predictions[i]-model.predict(instances_masked, verbose=0)[:,occlusion_class[i]]
    
    occlusion_explanation = pred_diffs[:,np.newaxis]*masks
    occlusion_explanation = np.mean(occlusion_explanation, axis=0)
    occlusion_explanation_1 = {}
    for k in index:
        occlusion_explanation_1[k] = occlusion_explanation[k[0]]
    occlusion_explanations.append(occlusion_explanation_1)
    print(i)
end = time.time()
print(end - start)


dump(occlusion_explanations, open('sb_occlusion_explanation_'+rec_model+dataset+'.npy', 'wb'))
dump(occlusion_class, open('sb_occlusion_class_'+rec_model+dataset+'.npy', 'wb'))
dump(indices, open('indices_'+rec_model+dataset+'.npy', 'wb'))