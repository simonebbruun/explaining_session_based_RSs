import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pickle import dump
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


dataset = 'diginetica'


''' Data '''
# Import.
train = pd.read_csv(dataset+'_train.csv')


# Ordinal encode.
encoder = OrdinalEncoder(dtype='int32', handle_unknown='use_encoded_value', unknown_value=-1)
train['item_id'] = encoder.fit_transform(train[['item_id']])

dump(encoder, open('ordinal_encoder_'+dataset+'.pkl', 'wb'))

vocab = list(range(len(encoder.categories_[0])))


# Training and validation split.
train = train.sort_values(['timestamp', 'session_id']).reset_index(drop=True)
train = train.drop(['timestamp'], axis=1)

train = list(train.groupby('session_id', sort=False)['item_id'].apply(list))

train, valid = train_test_split(train, test_size=0.1, shuffle=False)


# Padding.
train = tf.keras.utils.pad_sequences(train, value=-1)
valid = tf.keras.utils.pad_sequences(valid, value=-1)

train = np.expand_dims(train, axis=-1)
valid = np.expand_dims(valid, axis=-1)


''' Model '''
def reciprocal_rank(predictions, test_set, k):
    rank = (-predictions).argsort()
    ranked_items = rank.argsort()
    relevant_items = np.where(test_set == 1, ranked_items, np.nan)
    relevant_items1 = np.where(relevant_items >= k, np.nan, relevant_items)
    min_rank = np.nanmin(relevant_items1, axis=1)
    rr = 1/(min_rank+1)
    rr = np.nan_to_num(rr)
    return rr


n_steps = train.shape[1]-1
n_columns = train.shape[2]
n_classes = len(vocab)

one_hot_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.IntegerLookup(mask_token=-1, output_mode='one_hot', vocabulary=vocab, num_oov_indices=0), input_shape=(None,n_columns))
valid_true = one_hot_layer(valid[:,-1:,:])[:,0,:].numpy()


tf.keras.utils.set_random_seed(42)
batch_size, units, rate = 128, 256, 0.3
model = tf.keras.models.Sequential()
model.add(one_hot_layer)
model.add(tf.keras.layers.Masking(mask_value=0))
model.add(tf.keras.layers.GRU(units, return_sequences = True))
model.add(tf.keras.layers.Dropout(rate))
model.add(tf.keras.layers.Dense(units, activation = 'relu'))
model.add(tf.keras.layers.Dense(n_classes, activation = 'linear'))

model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=-1), optimizer = 'adam')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('gru4rec_model_'+dataset+'.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history = model.fit(train[:,:n_steps,:], train[:,1:,:], validation_data=(valid[:,:n_steps,:], valid[:,1:,:]), epochs=100, batch_size=batch_size, callbacks=[es, mc])

saved_model = load_model('gru4rec_model_'+dataset+'.h5')
valid_pred = saved_model.predict(valid[:,:n_steps,:])[:,-1,:]
valid_pred = np.exp(valid_pred)/(np.sum(np.exp(valid_pred),axis=1)[:,None])

print(np.mean(reciprocal_rank(valid_pred, valid_true, 10)))


