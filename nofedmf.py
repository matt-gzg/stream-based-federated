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
        logging.FileHandler(output_path('federated_stream', 'logs', 'training.log'), mode='a'),
    ]
)

logger = logging.getLogger(__name__)


def get_init(matrix, key, scale):
    if key not in matrix:
        matrix[key] = scale * np.random.randn(hidden_dim)
    return matrix[key]


def user_update(user_vector, item_id, v, lr, reg_u, reg_v, iter):
    gradient = {}
    p_ui = 1.0

    for _ in range(iter):
        error = p_ui - np.dot(user_vector, v)
        user_vector = user_vector - lr * (-2 * error * v + 2 * reg_u * user_vector)
        gradient[item_id] = lr * (-2 * error * user_vector + 2 * reg_v * v)

    return user_vector, gradient


def aggregate_fedavg(items_global, items_local, interact_count, updated_items,
                      item_users_updated, item_users_all):
    for item_id in updated_items:
        total_interacs = 0
        weighted_sum = np.zeros(hidden_dim)

        for uid in item_users_updated[item_id]:
            n = interact_count[uid]
            weighted_sum += (n * items_local[uid][item_id])
            total_interacs += n

        if total_interacs > 0:
            items_global[item_id] = weighted_sum / total_interacs

            for uid in item_users_all[item_id]:
                items_local[uid][item_id] = items_global[item_id].copy()

    return items_global, items_local


def process_iter(users, items_global, items_local, seen_online, uid, item_id, evaluate, scale,
                  lr, reg_u, reg_v, iter):
    hit = None
    hit_rate_time = 0.0

    user_vector = get_init(users, uid, scale)

    if uid not in items_local:
        items_local[uid] = {}

    if item_id not in items_local[uid]:
        if item_id in items_global:
            # Item já está no catálogo do servidor: usuário baixa o vetor atual.
            items_local[uid][item_id] = items_global[item_id].copy()
        else:
            # Item ainda não foi admitido no catálogo (será criado na próxima
            # agregação, a partir da média dos vetores locais que o tocaram).
            # Até lá, o cliente inicializa seu próprio vetor local.
            items_local[uid][item_id] = scale * np.random.randn(hidden_dim)

    current_item = items_local[uid][item_id]

    if evaluate:
        eval_start = time.perf_counter()

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
        topk_pos = np.argpartition(-p, kk - 1)[:kk]
        topk_items = {item_ids[pos] for pos in topk_pos}
        hit = item_id in topk_items

        hit_rate_time = time.perf_counter() - eval_start

    new_user_vector, gradient = user_update(user_vector, item_id, current_item, lr, reg_u, reg_v, iter)
    users[uid] = new_user_vector
    items_local[uid][item_id] = current_item - gradient[item_id]

    seen_online.setdefault(uid, set()).add(item_id)

    return hit, hit_rate_time


def run_phase(data, phase, evaluate, users, items_global, items_local, seen_online, scale,
              interact_count, updated_items, item_users_updated, item_users_all,
              time_dataset, obs_offset, aggregation_interval):
    logger.info(f'[{phase}] starting | Time={time.perf_counter() - time_dataset:.2f}s')

    hits = 0
    total_predictions = 0
    total_hit_rate_time = 0.0
    user_time_list = []
    aggregation_time_list = []
    obs_count = obs_offset

    phase_start = time.perf_counter()
    last_log_time = time.perf_counter()
    last_aggregation_time = time.perf_counter()

    for local_count, (uid, item_id) in enumerate(data, start=1):
        obs_count = obs_offset + local_count
        t = time.perf_counter()

        hit, hit_rate_time = process_iter(
            users, items_global, items_local, seen_online, uid, item_id, evaluate, scale,
            lr, reg_u, reg_v, iter
        )

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

        if obs_count % 10000 == 0:
            current_time = time.perf_counter()
            block_time = current_time - last_log_time
            logger.info(
                f'[{phase}] Processed={obs_count} | '
                f'AvgUserTime={np.mean(user_time_list[-10000:]):.6f}s | '
                f'BlockTime={block_time:.6f}s | '
                f'TotalTime={time.perf_counter() - time_dataset:.6f}s | '
                f"HR@{k}={'N/A' if not evaluate or total_predictions == 0 else f'{hits / total_predictions:.4f}'}"
            )
            last_log_time = current_time

        if aggregation_interval > 0 and obs_count % aggregation_interval == 0:
            agg_start = time.perf_counter()

            aggregate_fedavg(items_global, items_local, interact_count, updated_items,
                              item_users_updated, item_users_all)

            agg_elapsed = time.perf_counter() - agg_start
            aggregation_time_list.append(agg_elapsed)

            item_users_updated.clear()
            updated_items.clear()
            interact_count.clear()

            agg_time = time.perf_counter() - last_aggregation_time
            logger.info(
                f'[Agg {obs_count // aggregation_interval}] '
                f'AggTime={agg_elapsed:.6f}s | '
                f'Time={agg_time:.2f}s'
            )
            last_aggregation_time = time.perf_counter()

    logger.info(f'[{phase}] Finished | Time={time.perf_counter() - phase_start:.2f}s')
    if evaluate and total_predictions > 0:
        logger.info(f'HR@{k}: {hits / total_predictions:.4f}')
        logger.info(f'Hit Rate Total Time: {total_hit_rate_time:.4f}s')
        logger.info(f'Hit Rate Avg Time: {total_hit_rate_time / total_predictions:.6f}s')
    if user_time_list:
        logger.info(f'User Average Time: {np.mean(user_time_list):.6f}s')
    logger.info(f'Aggregations in phase: {len(aggregation_time_list)}')

    return obs_count


def main():
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

    n_train = len(train_data)
    n_test = len(test_data)
    n_total = n_train + n_test

    aggregation_interval = n_total // aggregations if aggregations > 0 else 0

    logger.info(
        f'hidden_dim: {hidden_dim} | reg: {reg_u} | lr: {lr} | iter: {iter}\n'
        f'Dataset size: {n_total} | Train: {n_train} | Test: {n_test}\n'
        f'Aggregations: {aggregations} | Aggregation interval: {aggregation_interval}'
    )

    obs_offset = run_phase(
        train_data, 'TRAIN', evaluate=False,
        users=users, items_global=items_global, items_local=items_local, seen_online=seen_online,
        scale=scale, interact_count=interact_count, updated_items=updated_items,
        item_users_updated=item_users_updated, item_users_all=item_users_all,
        time_dataset=time_dataset, obs_offset=0, aggregation_interval=aggregation_interval
    )

    run_phase(
        test_data, 'TEST', evaluate=True,
        users=users, items_global=items_global, items_local=items_local, seen_online=seen_online,
        scale=scale, interact_count=interact_count, updated_items=updated_items,
        item_users_updated=item_users_updated, item_users_all=item_users_all,
        time_dataset=time_dataset, obs_offset=obs_offset, aggregation_interval=aggregation_interval
    )

    logger.info(f'Total Time: {time.perf_counter() - time_dataset:.4f} seconds')
    logger.info(f'Total Users Seen: {len(users)} | Total Items Seen: {len(items_global)}')


if __name__ == '__main__':
    main()