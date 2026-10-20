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

def user_update(user_vector, item_id):
    gradient = {}
    p_ui = 1.0
 
    for _ in range(iter):
        error = p_ui - np.dot(user_vector, reg_v)
        user_vector = user_vector - lr * (-2 * error * reg_v + 2 * reg_u * user_vector)
        gradient[item_id] = lr * (-2 * error * user_vector + 2 * reg_v * item_id)
 
    return user_vector, gradient


def get_init(matrix, key, scale):
    if key not in matrix:
        matrix[key] = scale * np.random.randn(hidden_dim)
    return matrix[key]

def agg(items_global, items_local, interact_count, updated_items,
                      item_users_updated, item_users_all):
    for item_id in updated_items:
        total_interacs = 0
        weighted_sum = np.zeros_like(items_global[item_id])
 
        for uid in item_users_updated[item_id]:
            n = interact_count[uid]
            weighted_sum += (n * items_local[uid][item_id])
            total_interacs += n
 
        if total_interacs > 0:
            items_global[item_id] = weighted_sum / total_interacs
 
            for uid in item_users_all[item_id]:
                items_local[uid][item_id] = items_global[item_id].copy()
 
    return items_global, items_local

def process_iter(users, items_global, items_local, seen_online, uid, item_id, evaluate, scale):
    hit = None
    hit_rate_time = 0.0

    user_vector = get_init(users, uid, scale)

    if uid not in items_local:
        items_local[uid] = {}
    if item_id not in items_global:
        items_global[item_id] = get_init(items_global, item_id, scale)
    if item_id not in items_local[uid]:
        items_local[uid][item_id] = items_global[item_id].copy()

    current_item = items_local[uid][item_id]

    if evaluate:
        eval = time.perf_counter()

        item_ids = list(items_global.keys())
        vecs = [
            items_local[uid][iid] if iid in items_local[uid] else items_global[iid]
            for iid in item_ids
        ]
        items_stack = np.stack(vecs)

        p = np.dot(items_stack, user_vector)
        seen = seen_online.get(uid)
        if seen:
            seen_pos = [pos for pos, iid in enumerate(item_ids) if iid in seen]
            p[seen_pos] = -np.inf

        kk = min(k, len(item_ids))
        topk_pos = np.argpartition(-p, kk-1)[:kk]
        topk_items = {item_ids[pos] for pos in topk_pos}
        hit = item_id in topk_items

        hit_rate_time = time.perf_counter() - eval

    new_user_vector, gradient = user_update(user_vector, item_id, current_item)
    users[uid] = new_user_vector
    items_local[uid][item_id] = current_item - gradient[item_id]

    seen_online.setdefault(uid, set()).add(item_id)

    return hit, hit_rate_time

def run_phase(data, phase, evaluate, users, items_global, items_local, seen_online, scale, interact_count, updated_items, item_users_updated, item_users_all, time_dataset, o_offset):
    logger.info(f'[{phase}] starting | Time={time.perf_counter() - time_dataset:.2f}s')

    hits = 0
    total_predictions = 0
    total_hit_rate_time = 0.0
    user_time_list = []
    agg_time_list = []
    o_count = o_offset

    phase_start = time.perf_counter()
    last_log_time = time.perf_counter()
    last_agg_time = time.perf_counter()

    for local_count, (uid, item_id) in enumerate(data, start=1):
        o_count = o_offset + local_count
        t = time.perf_counter()

        hit, hit_rate_time = process_iter(users, items_global, items_local, seen_online, uid, item_id, evaluate, scale)

        updated_items.add(item_id)
        interact_count[uid] = interact_count.get(uid, 0) + 1

        item_users_updated.setdefault(item_id, set()).add(uid)
        item_users_all.setdefault(item_id, set()).add(uid)

        user_time_list.append(time.perf_counter() - t)

        if evaluate:
            if hit:
                hits += 1
            total_predictions += 1
            total_hit_rate_time += hit_rate_time

        if o_count % 10000 == 0:
            current_time = time.perf_counter()
            block_time = current_time - last_log_time
            logger.info(
                f'[{phase}] Processed={o_count} | '
                f'AvgUserTime={np.mean(user_time_list[-10000:]):.6f}s | '
                f'BlockTime={block_time:.6f}s | '
                f'TotalTime={time.perf_counter() - time_dataset:.6f}s | '
                f"HR@{k}={'N/A' if not evaluate or total_predictions == 0 else f'{hits / total_predictions:.4f}'}"
            )
            last_log_time = current_time

        if o_count % aggregations == 0:
            agg_start = time.perf_counter()

            agg(items_global, items_local, interact_count, updated_items, item_users_updated, item_users_all)

            agg_elapsed = time.perf_counter() - agg_start
            agg_time_list.append(agg_elapsed)

            item_users_updated.clear()
            updated_items.clear()
            interact_count.clear()

            agg_time = time.perf_counter() - last_agg_time
            logger.info(
                f'[Agg {o_count // aggregations}] '
                f'AggTime={agg_elapsed:.6f}s | '
                f'Time={agg_time:.2f}s'
            )
            last_agg_time = time.perf_counter()

    logger.info(f'[{phase}] Finished | Time={time.perf_counter() - phase_start:.2f}s')
    if evaluate and total_predictions > 0:
        logger.info(f'HR@{k}: {hits / total_predictions:.4f}')
        logger.info(f'Hit Rate Total Time: {total_hit_rate_time:.4f}s')
        logger.info(f'Hit Rate Avg Time: {total_hit_rate_time / total_predictions:.6f}s')
    if user_time_list:
        logger.info(f'User Average Time: {np.mean(user_time_list):.6f}s')
    logger.info(f'Aggregations in phase: {len(agg_time_list)}')
 
    return o_count


def main():
    np.random.seed(42)
    time_dataset = time.perf_counter()

    scale = 0.1 / np.sqrt(hidden_dim)

    users = {}
    items_global = {}
    items_local = {}
    seen_online = {}

    interact_count = {}
    updated_items = set()
    item_users_updated = {}
    item_users_all = {}

    o_offset = run_phase(train_data, 'TRAIN', evaluate=False, users=users, items_global=items_global, items_local=items_local, seen_online=seen_online, scale=scale, interact_count=interact_count, updated_items=updated_items, item_users_updated=item_users_updated, item_users_all=item_users_all, time_dataset=time_dataset, o_offset=0)

    run_phase(test_data, 'TEST', evaluate=True, users=users, items_global=items_global, items_local=items_local, seen_online=seen_online, scale=scale, interact_count=interact_count, updated_items=updated_items, item_users_updated=item_users_updated, item_users_all=item_users_all, time_dataset=time_dataset, o_offset=o_offset)

    logger.info(f'Total Time: {time.perf_counter() - time_dataset:.4f} seconds')
    logger.info(f'Total Users Seen: {len(users)} | Total Items Seen: {len(items_global)}')

if __name__ == '__main__':
    logger.info(
        f'\n'
        f'hidden_dim: {hidden_dim} | reg: {reg_u} | lr: {lr} | iter: {iter}\n'
        f'Dataset size: {len(train_data) + len(test_data)} | Train size: {len(train_data)} | Test size: {len(test_data)}'
    )

    main()
