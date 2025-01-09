import pandas as pd
from pickle import load
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


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
            test_data[i] = test_data[i]
    test_data = tf.keras.utils.pad_sequences(test_data, padding='pre')


''' Model. '''
model = load_model(rec_model+'_model_'+dataset+'.h5')

if rec_model == 'gru4rec':
    one_hot_layer = model.get_layer('time_distributed')
    preds = model.predict(test_data[:,:-1,:])[:,-1,:]
    test_true = one_hot_layer(test_data[:,-1:,:])[:,0,:].numpy()
else:
    preds = model.predict(test_data[:,:-1])

preds = np.exp(preds)/(np.sum(np.exp(preds),axis=1)[:,None])


def reciprocal_rank_1(predictions, test_set, k):
    rank = (-predictions).argsort()
    ranked_items = rank.argsort()
    relevant_items = np.where(test_set == 1, ranked_items, np.nan)
    relevant_items1 = np.where(relevant_items >= k, np.nan, relevant_items)
    min_rank = np.nanmin(relevant_items1, axis=1)
    rr = 1/(min_rank+1)
    rr = np.nan_to_num(rr)
    return rr

def reciprocal_rank_2(predictions, test_set, k):
    rank = (-predictions).argsort()
    relevant_items = []
    for i in range(len(rank)):
        relevant_items.append(np.where(rank[i]==test_set[i]))
    relevant_items = np.vstack(relevant_items)
    relevant_items1 = np.where(relevant_items >= k, np.nan, relevant_items)
    rr = 1/(relevant_items1+1)
    rr = np.nan_to_num(rr)
    return rr


if rec_model == 'gru4rec':
    rr_10 = reciprocal_rank_1(preds, test_true, 10)
    rr_20 = reciprocal_rank_1(preds, test_true, 20)
else:
    rr_10 = reciprocal_rank_2(preds, (test_data[:,-1]-1), 10)
    rr_20 = reciprocal_rank_2(preds, (test_data[:,-1]-1), 20)
    
hit_10 = np.where(rr_10>0,1,0)
hit_20 = np.where(rr_20>0,1,0)

print(np.mean(hit_10))
print(np.mean(rr_10))
print(np.mean(hit_20))
print(np.mean(rr_20))
