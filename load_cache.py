import os
import pickle
import time

from shared_parameter import dataset, validation_ratio

start = time.perf_counter()

current_path = os.path.dirname(os.path.abspath(__file__))

cache_path = os.path.join(
    current_path,
    f'{dataset}.pkl'
)

print('Loading dataset')

with open(cache_path, 'rb') as f:
    cached_data = pickle.load(f)

train_data = cached_data['train_data']
validation_data = cached_data['validation_data']
test_data = cached_data['test_data'] if validation_ratio == 0.0 else validation_data

user_id_list = cached_data['user_id_list']
item_id_list = cached_data['item_id_list']
user_id_map = cached_data['user_id_map']
item_id_map = cached_data['item_id_map']

print(f'Loading time: {time.perf_counter() - start:.2f}s')