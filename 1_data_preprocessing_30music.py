import pandas as pd
import numpy as np



# Import data.
data = pd.read_csv('30music.csv', sep='\t')

# Rename columns, change datatypes, and select columns.
data = data.rename(columns={'Time': 'timestamp_unix', 'ItemId': 'item_id', 'SessionId': 'session_id'})

data['timestamp'] = pd.to_datetime(data['timestamp_unix'], unit='s')

data = data[['session_id', 'timestamp', 'item_id']]



# Removing short sessions and low-frequent items.
min_session = np.min(data['session_id'].value_counts())
min_item = np.min(data['item_id'].value_counts())
while (min_session<5) or (min_item<50):
    data = data.groupby(['session_id']).filter(lambda x: len(x)>=5)

    item_count = (data['item_id'].value_counts()).rename_axis('item_id').reset_index(name='count')
    item_count = item_count[item_count['count']>=50]
    
    data = data[data['item_id'].isin(item_count['item_id'])]
    
    min_session = np.min(data['session_id'].value_counts())
    min_item = np.min(data['item_id'].value_counts())
    

# Truncating the number of items in a session to the 99th percentile.
session_count = data['session_id'].value_counts()
items_max = np.percentile(session_count, 99).astype(int)
data = data.sort_values('timestamp', ascending=False).groupby('session_id').head(items_max)


# Data statistics.
n_interactions = len(data)
n_items = data['item_id'].nunique()
n_sessions = data['session_id'].nunique()

n_repeated = np.sum(data[['session_id', 'item_id']].duplicated())

sessions_with_repeated = data[['session_id', 'item_id']].groupby(['session_id', 'item_id'], as_index=False).size()
sessions_with_repeated = sessions_with_repeated[sessions_with_repeated['size']>1]
sessions_with_repeated = sessions_with_repeated['session_id'].nunique()

session_stats = data['session_id'].value_counts()
print(np.min(session_stats))
print(np.median(session_stats))
print(np.mean(session_stats))
print(np.max(session_stats))


# Train and test split.
sessions = data[['session_id', 'timestamp']].groupby('session_id', as_index=False)['timestamp'].min()
sessions = sessions.sort_values('timestamp').reset_index(drop=True)

split_index = round(len(sessions)*0.9)
sessions_train = sessions[sessions.index<split_index]
sessions_test = sessions[sessions.index>=split_index]

data_train = data[data['session_id'].isin(sessions_train['session_id'])]
data_test = data[data['session_id'].isin(sessions_test['session_id'])]


# Remove test instances where last item is not in train set.
train_items = data_train['item_id'].unique()
test_items = data_test.sort_values('timestamp').drop_duplicates(['session_id'],keep='last')
test_items = test_items[test_items['item_id'].isin(train_items)]

data_test = data_test[data_test['session_id'].isin(test_items['session_id'])]


# Export.
data_train.to_csv('30music_train.csv', index=False)
data_test.to_csv('30music_test.csv', index=False)
