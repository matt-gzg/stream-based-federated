import time
import numpy as np
import logging
from shared_parameter import *
from load_cache import (user_id_list, item_id_list, train_data, test_data)
from logs_paths import output_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(output_path('isgd', 'isgd.log'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)

def user_item_update(user_vector, item_vector):
    p_ui = 1.0
    for _ in range(iter):
        error = p_ui - np.dot(user_vector, item_vector)
        grad_user = -2 * error * item_vector + 2 * reg_u * user_vector
        grad_item = -2 * error * user_vector + 2 * reg_v * item_vector
        user_vector = user_vector - lr * grad_user
        item_vector = item_vector - lr * grad_item
        return user_vector, item_vector

def process_iter(users_matrix, items_matrix, items_matrix_T, seen_items_online, uid, item_id, evaluate):
    hit = None
    hit_rate_time = 0.0

    if evaluate:
        eval = time.perf_counter()
        p = np.dot(users_matrix[uid], items_matrix_T)
        seen_items = seen_items_online[uid]
        if seen_items:
            p[list(seen_items)] = -np.inf
        top_k = np.argpartition(-p, k)[:k]
        hit = item_id in top_k
        hit_rate_time = time.perf_counter() - eval

    users_matrix[uid], items_matrix[item_id] = user_item_update(users_matrix[uid], items_matrix[item_id])   
    items_matrix_T[:, item_id] = items_matrix[item_id]
    seen_items_online[uid].add(item_id)

    return hit, hit_rate_time

def run_phase(data, phase, evaluate, users_matrix, items_matrix, items_matrix_T, seen_items_online, time_dataset):
    logger.info(f'[{phase}] starting | Time={time.perf_counter() - time_dataset:.2f}s')
    hits = 0
    total_predictions = 0
    total_hit_rate_time = 0.0

    phase_start = time.perf_counter()

    for uid, item_id in data:
        hit, hit_rate_time = process_iter(users_matrix, items_matrix, items_matrix_T, seen_items_online, uid, item_id, evaluate)

        if evaluate:
            if hit:
                hits += 1
            total_predictions += 1
            total_hit_rate_time += hit_rate_time

            #logger.info(f'HR@{k}: {hits / total_predictions:.4f}')

    logger.info(f'[{phase}] Finished | Time={time.perf_counter() - phase_start:.2f}s')
    if evaluate:
        logger.info(f'HR@{k}: {hits / total_predictions:.4f}')
        logger.info(f'Hit Rate Total Time: {total_hit_rate_time:.4f}s')
        logger.info(f'Hit Rate Avg Time: {total_hit_rate_time / total_predictions:.6f}s')

def main():
    np.random.seed(42)
    time_dataset = time.perf_counter()

    scale = 0.1 / np.sqrt(hidden_dim)

    users_matrix = scale * np.random.randn(len(user_id_list), hidden_dim)
    items_matrix = scale * np.random.randn(len(item_id_list), hidden_dim)
    items_matrix_T = items_matrix.T.copy()

    seen_items_online = [set() for _ in user_id_list]

    run_phase(train_data, 'TRAIN', evaluate=False, users_matrix=users_matrix, items_matrix=items_matrix, items_matrix_T=items_matrix_T, seen_items_online=seen_items_online, time_dataset=time_dataset)
    run_phase(test_data, 'TEST', evaluate=True, users_matrix=users_matrix, items_matrix=items_matrix, items_matrix_T=items_matrix_T, seen_items_online=seen_items_online, time_dataset=time_dataset)

    logger.info(f'Total Time: {time.perf_counter() - time_dataset:.4f} seconds')

if __name__ == '__main__':
    train_l = len(train_data)
    test_l = len(test_data)
    n = train_l + test_l

    logger.info(
        f'\n'
        f'hidden_dim: {hidden_dim} | reg: {reg_u} | lr: {lr} | iter: {iter}\n'
        f'Dataset size: {n} | Train size: {train_l} | Test size: {test_l}'
    )

    main()
