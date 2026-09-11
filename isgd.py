import time
import numpy as np
import logging
from shared_parameter import *
from load_cache import (train_data, test_data)
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

def get_init(matrix, key, scale):
    if key not in matrix:
        matrix[key] = scale * np.random.randn(hidden_dim)
    return matrix[key]

def process_iter(users, items, seen_online, uid, item_id, evaluate, scale):
    hit = None
    hit_rate_time = 0.0

    user_vector = get_init(users, uid, scale)
    item_vector = get_init(items, item_id, scale)

    if evaluate:
        eval = time.perf_counter()

        item_ids = list(items.keys())
        items_stack = np.stack([items[iid] for iid in item_ids])
        p = np.dot(items_stack, user_vector)
        seen = seen_online.get(uid, set())
        if seen:
            seen_pos = [pos for pos, iid in enumerate(item_ids) if iid in seen]
            p[seen_pos] = -np.inf
        kk = min(k, len(item_ids))
        topk_pos = np.argpartition(-p, kk-1)[:kk]
        topk_items = {item_ids[pos] for pos in topk_pos}
        hit = item_id in topk_items
        hit_rate_time = time.perf_counter() - eval

    new_user_vector, new_item_vector = user_item_update(user_vector, item_vector)
    users[uid] = new_user_vector
    items[item_id] = new_item_vector

    seen_online.setdefault(uid, set()).add(item_id)

    return hit, hit_rate_time

def run_phase(data, phase, evaluate, users, items, seen_online, scale, time_dataset):
    logger.info(f'[{phase}] starting | Time={time.perf_counter() - time_dataset:.2f}s')
    hits = 0
    total_predictions = 0
    total_hit_rate_time = 0.0

    phase_start = time.perf_counter()

    for uid, item_id in data:
        hit, hit_rate_time = process_iter(users, items, seen_online, uid, item_id, evaluate, scale)

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

    users = {}
    items = {}
    seen_online = {}

    run_phase(train_data, 'TRAIN', evaluate=False, users=users, items=items, seen_online=seen_online, scale=scale, time_dataset=time_dataset)
    run_phase(test_data, 'TEST', evaluate=True, users=users, items=items, seen_online=seen_online, scale=scale, time_dataset=time_dataset)

    logger.info(f'Total Time: {time.perf_counter() - time_dataset:.4f} seconds')
    logger.info(f'Total Users Seen: {len(users)} | Total Items Seen: {len(items)}')

if __name__ == '__main__':
    logger.info(
        f'\n'
        f'hidden_dim: {hidden_dim} | reg: {reg_u} | lr: {lr} | iter: {iter}\n'
        f'Dataset size: {len(train_data) + len(test_data)} | Train size: {len(train_data)} | Test size: {len(test_data)}'
    )

    main()
