import os
import pickle
import pandas as pd
import time

from shared_parameter import dataset, train_ratio, test_ratio, validation_ratio

start = time.perf_counter()

print('Loading dataset')
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, dataset)

dtype = {
    'userId': str,
    'movieId': int,
    'rating': float,
    'timestamp': int
}

df = pd.read_csv(
    os.path.join(data_path, 'ratings.csv'),
    dtype=dtype
)

print('Sorting dataset by timestamp')
df.sort_values('timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)

total_interactions = len(df)
num_users = df['userId'].nunique()
num_items = df['movieId'].nunique()

print(f'Users: {df["userId"].nunique()}')
print(f'Items: {df["movieId"].nunique()}')
print(f'Interactions: {len(df)}')
print(f'Sparsity: {1.0 - (total_interactions / (num_users * num_items)):.6f}')

print('Filtering dataset')
df = df[df['rating'] == 5].copy()

print(f'Interactions after filtering: {len(df)}')
print(f'Sparsity after filtering: {1.0 - (total_interactions / (num_users * num_items)):.6f}')

print(f'Removing columns')
df = df[['userId', 'movieId', 'timestamp']]

print(f'Building maps')
user_id_list = sorted(df['userId'].unique(), key=lambda x: int(x))
item_id_list = sorted(df['movieId'].unique())

user_id_map = {
    user_id: idx
    for idx, user_id in enumerate(user_id_list)
}

item_id_map = {
    item_id: idx
    for idx, item_id in enumerate(item_id_list)
}

df['user_idx'] = df['userId'].map(user_id_map)
df['item_idx'] = df['movieId'].map(item_id_map)

print(f'Building stream data')
stream_data = []

for row in df.itertuples(index=False):
    stream_data.append((row.user_idx, row.item_idx))

del df

print('Splitting dataset')
n = len(stream_data)

temp_train_size = int(n * train_ratio)
val_size = int(temp_train_size * validation_ratio)
train_size = temp_train_size - val_size
test_size = int(n * test_ratio)

train_data = stream_data[:train_size]
validation_data = stream_data[train_size:temp_train_size]
test_data = stream_data[temp_train_size:temp_train_size + test_size]

print(f'Train: {len(train_data)} | Validation: {len(validation_data)} | Test: {len(test_data)}')

cache_path = os.path.join(current_path, f'{dataset}.pkl')

print(f'Saving dataset to cache')
with open(cache_path, 'wb') as f:
    pickle.dump({
        'stream_data': stream_data,
        'user_id_list': user_id_list,
        'item_id_list': item_id_list,
        'user_id_map': user_id_map,
        'item_id_map': item_id_map,
        'train_data': train_data,
        'test_data': test_data,
        'validation_data': validation_data
    }, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f'Preprocess time: {time.perf_counter() - start:.2f}s')