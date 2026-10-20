import time
import numpy as np
import logging
from params import hidden_dim, reg_u, reg_v, lr, n_iter, k
from load_cache import (train_data, test_data)
from logs_paths import output_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(output_path('isgd', 'isgd.log'), mode='w'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

def user_item_update(user_vector, item_vector):
    for _ in range(n_iter):
        error = 1.0 - np.dot(user_vector, item_vector)
        grad_user = -2 * error * item_vector + 2 * reg_u * user_vector
        grad_item = -2 * error * user_vector + 2 * reg_v * item_vector
        user_vector = user_vector - lr * grad_user
        item_vector = item_vector - lr * grad_item
    return user_vector, item_vector

def get_init(matrix, key, scale):
    if key not in matrix:
        matrix[key] = scale * np.random.randn(hidden_dim)
    return matrix[key]

class _ItemCache:
    """Mirrors `items` dict insertion order with a C-contiguous float64 matrix.

    Eliminates per-event `list(keys) + np.stack` (O(I*d) alloc/copy).
    Rows hold value copies in the same order/precision, so
    `mat[:n] @ user` is bit-identical to `stack(...) @ user`.
    """

    def __init__(self, capacity=4096):
        self.ids = []
        self.pos = {}
        self.mat = np.empty((capacity, hidden_dim), dtype=np.float64)
        self.n = 0

    def _grow(self):
        self.mat = np.concatenate([self.mat, np.empty_like(self.mat)], axis=0)

    def get_init(self, items, item_id, scale):
        vec = items.get(item_id)
        if vec is None:
            vec = scale * np.random.randn(hidden_dim)
            items[item_id] = vec
            p = self.n
            self.pos[item_id] = p
            self.ids.append(item_id)
            if p >= self.mat.shape[0]:
                self._grow()
            self.mat[p] = vec
            self.n += 1
        return vec

    def write_back(self, item_id, new_vec):
        self.mat[self.pos[item_id]] = new_vec

def process_iter(users, items, cache, seen_online, uid, item_id, evaluate, scale):
    hit = None
    hit_rate_time = 0.0

    user_vector = get_init(users, uid, scale)
    # Same RNG/order as before (user first, then item); cache includes
    # the current item before eval, exactly like the old `items` dict did.
    item_vector = cache.get_init(items, item_id, scale)

    if evaluate:
        eval = time.perf_counter()

        n = cache.n
        p = cache.mat[:n] @ user_vector
        seen = seen_online.get(uid, set())
        if seen:
            # Same masked positions as the old O(I) scan, in O(|seen_u|).
            seen_pos = [cache.pos[i] for i in seen if i in cache.pos]
            if seen_pos:
                p[seen_pos] = -np.inf
        kk = min(k, n)
        topk_pos = np.argpartition(-p, kk-1)[:kk]
        # Same membership result as `{ids[p] for p in topk}; item in set`.
        _ids = cache.ids
        _target = item_id
        _hit = False
        for pos in topk_pos:
            if _ids[pos] == _target:
                _hit = True
                break
        hit = _hit
        hit_rate_time = time.perf_counter() - eval

    new_user_vector, new_item_vector = user_item_update(user_vector, item_vector)
    users[uid] = new_user_vector
    items[item_id] = new_item_vector
    cache.write_back(item_id, new_item_vector)

    seen_online.setdefault(uid, set()).add(item_id)

    return hit, hit_rate_time

def run_phase(data, phase, evaluate, users, items, cache, seen_online, scale, time_dataset):
    logger.info(f'[{phase}] starting | Time={time.perf_counter() - time_dataset:.2f}s')
    hits = 0
    total_predictions = 0
    total_hit_rate_time = 0.0

    phase_start = time.perf_counter()

    for uid, item_id in data:
        hit, hit_rate_time = process_iter(users, items, cache, seen_online, uid, item_id, evaluate, scale)

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
    cache = _ItemCache()
    seen_online = {}

    run_phase(train_data, 'TRAIN', evaluate=False, users=users, items=items, cache=cache, seen_online=seen_online, scale=scale, time_dataset=time_dataset)
    run_phase(test_data, 'TEST', evaluate=True, users=users, items=items, cache=cache, seen_online=seen_online, scale=scale, time_dataset=time_dataset)

    logger.info(f'Total Time: {time.perf_counter() - time_dataset:.4f} seconds')
    logger.info(f'Total Users Seen: {len(users)} | Total Items Seen: {len(items)}')

if __name__ == '__main__':
    logger.info(
        f'\n'
        f'hidden_dim: {hidden_dim} | reg: {reg_u} | lr: {lr} | n_iter: {n_iter}\n'
        f'Dataset size: {len(train_data) + len(test_data)} | Train size: {len(train_data)} | Test size: {len(test_data)}'
    )

    main()
