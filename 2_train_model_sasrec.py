import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pickle import dump
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import gc
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import sasrec as sr


dataset = '30music'


''' Data '''
# Import.
train = pd.read_csv(dataset+'_train.csv')

# Ordinal encode.
encoder = OrdinalEncoder(dtype='int32', handle_unknown='use_encoded_value', unknown_value=-1)
train['item_id'] = encoder.fit_transform(train[['item_id']])+1

dump(encoder, open('ordinal_encoder_'+dataset+'.pkl', 'wb'))

item_num = len(encoder.categories_[0])


# Train and valid split.
seq_max_len = np.max(train.groupby('session_id').size())
sessions = train[['session_id', 'timestamp']].groupby('session_id', as_index=False)['timestamp'].min()
sessions = sessions.sort_values('timestamp').reset_index(drop=True)

split_index = round(len(sessions)*0.9)
sessions_train = sessions[sessions.index<split_index]
sessions_valid = sessions[sessions.index>=split_index]

valid = train[train['session_id'].isin(sessions_valid['session_id'])]
train = train[train['session_id'].isin(sessions_train['session_id'])]

del sessions
del sessions_train
del sessions_valid
gc.collect()


# Aggregating features.
session_col = 'session_id'
behavior_key_col = 'item_id'

train = train.sort_values(['timestamp', 'session_id']).reset_index(drop=True)
train = train.drop(['timestamp'], axis=1)
train = sr.aggregate_features(train, session_col=session_col, behavior_key_col=behavior_key_col, seq_max_len=seq_max_len)
train_x = tf.keras.preprocessing.sequence.pad_sequences(train['behavior_' + behavior_key_col], padding='pre')
train_y = np.array(train['candidate_' + behavior_key_col])-1

valid = valid.sort_values(['timestamp', 'session_id']).reset_index(drop=True)
valid = valid.drop(['timestamp'], axis=1)

valid_x2 = list(valid.groupby('session_id', sort=False)['item_id'].apply(list))
valid_y2 = []
for i in range(len(valid_x2)):
    valid_y2.append(valid_x2[i][-1]-1)
    valid_x2[i] = valid_x2[i][:-1]
valid_x2 = tf.keras.utils.pad_sequences(valid_x2, padding='pre')

valid = sr.aggregate_features(valid, session_col=session_col, behavior_key_col=behavior_key_col, seq_max_len=seq_max_len)
valid_x1 = tf.keras.preprocessing.sequence.pad_sequences(valid['behavior_' + behavior_key_col], padding='pre')
valid_y1 = np.array(valid['candidate_' + behavior_key_col])-1


del train
del valid
gc.collect()


''' Model '''
def reciprocal_rank(predictions, test_set, k):
    rank = (-predictions).argsort()
    relevant_items = []
    for i in range(len(rank)):
        relevant_items.append(np.where(rank[i]==test_set[i]))
    relevant_items = np.vstack(relevant_items)
    relevant_items1 = np.where(relevant_items >= k, np.nan, relevant_items)
    rr = 1/(relevant_items1+1)
    rr = np.nan_to_num(rr)
    return rr   


tf.keras.utils.set_random_seed(42)
batch_size = 256
units = 128
rate = 0.3

model = sr.sasrec(item_num=item_num, seq_max_len=seq_max_len-1, embedding_dim = units, attention_dim = units, conv_dims = [units, units], dropout_rate = rate)

model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer = 'adam')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('sasrec_model_'+dataset+'.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history = model.fit(train_x, train_y, validation_data=(valid_x1, valid_y1), epochs=100, batch_size=batch_size, callbacks=[es, mc])

saved_model = load_model('sasrec_model_'+dataset+'.h5')
valid_pred = saved_model.predict(valid_x2)
valid_pred = np.exp(valid_pred)/np.sum(np.exp(valid_pred),axis=0)

print(np.mean(reciprocal_rank(valid_pred, valid_y2, 10)))

