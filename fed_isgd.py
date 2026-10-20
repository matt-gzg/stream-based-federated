import time
import numpy as np
import logging
from params import *
from load_cache import train_data, test_data
from logs_paths import output_path

"""
1. The server initializes the system's global model. When the client connects to the system, a global item model already exists to be used for recommendation.
2. 

"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(output_path('isgd', 'geleia.log'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)

def init_global_model(data):
    users = {}
    items_global = {}
    for uid, item_id in data:
        if uid not in users:
            users[uid] = np.random.normal(0, 0.1, hidden_dim)
        if item_id not in items_global:
            items_global[item_id] = np.random.normal(0, 0.1, hidden_dim)
    return users, items_global


def train_batch(temp_users, items_global, train_data, epochs=b_iter):
    data = list(train_data)
    n = len(data)

    logger.info(f'[TRAIN-SERVER] starting | samples={n} | epochs={epochs}')
    train_start = time.perf_counter()

    for count in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        np.random.shuffle(data)

        for uid, item_id in data:
            au = temp_users[uid]
            bi = items_global[item_id]

            err = 1.0 - np.dot(au, bi)

            new_au = au + lr * (err * bi - reg_u * au)
            new_bi = bi + lr * (err * new_au - reg_v * bi)
            temp_users[uid] = new_au
            items_global[item_id] = new_bi

        logger.info(f'[TRAIN-SERVER] epoch {count}/{epochs} | Time={time.perf_counter() - epoch_start:.2f}s')

    logger.info(f'[TRAIN-SERVER] Finished | Time={time.perf_counter() - train_start:.2f}s')
    return items_global # só mantem os itens


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


def agg_int(n, aggregations):
    if aggregations <= 0 or n <= 0:
        return []
    if n < aggregations:
        return list(range(1, n + 1))
    base = n // aggregations
    rem = n % aggregations
    points = []
    acc = 0
    for i in range(aggregations):
        acc += base + (1 if i < rem else 0)
        points.append(acc)
    return points


def test_incremental(items_global, test_data):
    users = {}
    items_local = {}
    interact_count = {}
    updated_items = set()
    item_users_updated = {}
    item_users_all = {}

    n = len(test_data)
    agg_points = set(agg_int(n, aggregations))
    agg_count = 0
    # Snapshot congelado da matriz global: mesma versão p/ todos os clientes
    # criados dentro do intervalo. Só muda na agregação.
    items_global_copy = {iid: v.copy() for iid, v in items_global.items()}

    logger.info(f'[INCREMENTAL] starting | samples={n} | aggregations={aggregations}')
    inc_start = time.perf_counter()

    for obs_count, (uid, item_id) in enumerate(test_data, start=1):
        # New client
        if uid not in users:
            users[uid] = np.random.normal(0, 0.1, hidden_dim)
            items_local[uid] = {iid: v.copy() for iid, v in items_global_copy.items()}

        # New item
        if item_id not in items_global:
            items_global[item_id] = np.random.normal(0, 0.1, hidden_dim)
            items_global_copy[item_id] = items_global[item_id].copy()
        if item_id not in items_local[uid]:
            items_local[uid][item_id] = items_global_copy[item_id].copy()

        au = users[uid]
        bi = items_local[uid][item_id]

        err = 1.0 - np.dot(au, bi)

        new_au = au + lr * (err * bi - reg_u * au)
        new_bi = bi + lr * (err * new_au - reg_v * bi)
        users[uid] = new_au
        items_local[uid][item_id] = new_bi

        updated_items.add(item_id)
        interact_count[uid] = interact_count.get(uid, 0) + 1
        item_users_updated.setdefault(item_id, set()).add(uid)
        item_users_all.setdefault(item_id, set()).add(uid)

        if obs_count in agg_points:
            agg_count += 1
            agg_start = time.perf_counter()
            aggregate_fedavg(items_global, items_local, interact_count, updated_items,
                             item_users_updated, item_users_all)
            # Nova versão global disponível p/ o próximo intervalo.
            for iid, v in items_global.items():
                items_global_copy[iid] = v.copy()
            item_users_updated.clear()
            updated_items.clear()
            interact_count.clear()
            logger.info(f'[Agg {agg_count}/{len(agg_points)} @obs={obs_count}] Time={time.perf_counter() - agg_start:.6f}s | items={len(items_global)}')

    logger.info(f'[INCREMENTAL] Finished | Time={time.perf_counter() - inc_start:.2f}s')
    return users, items_global, items_local


def main():
    np.random.seed(42)
    time_dataset = time.perf_counter()

    # 1. Init global model (server-only)
    users_init, items_global = init_global_model(train_data)
    items_global = train_batch(users_init, items_global, train_data, epochs=b_iter)

    # 2. Incremental training
    users, items_global, _ = test_incremental(items_global, test_data)

    logger.info(f'Total Time: {time.perf_counter() - time_dataset:.4f} seconds')
    logger.info(f'Total Users (incr): {len(users)} | Total Items (server): {len(items_global)}')

    return items_global


if __name__ == '__main__':
    logger.info(
        f'\n'
        f'hidden_dim: {hidden_dim} | reg_u: {reg_u} | reg_v: {reg_v} | lr: {lr} | b_iter: {b_iter}\n'
        f'Train size: {len(train_data)}'
    )

    main()
