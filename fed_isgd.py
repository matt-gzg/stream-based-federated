"""Federated ISGD with streaming prequential evaluation.

Protocol (same as isgd_ot.py):
  TRAIN stream update-only, then TEST stream predict-before-update, HR@k.
  Same seed (42), same init scale/order, same SGD math, same seen-masking.

Federated paradigm:
  - Each user is a client: user vector + per-item LOCAL copies stay client-side.
  - Server holds GLOBAL item vectors (+ _ItemCache mirror for fast scoring).
  - Local SGD update reuses the exact isgd update equations; the item
    gradient is applied to the client's local copy only.
  - N_ROUNDS FedAvg rounds run spread evenly over the whole stream
    (AGG_INTERVAL = total_events // N_ROUNDS events each); each round
    averages per item over clients that touched it (weight = client's
    event count since last aggregation), then broadcasts the averaged
    vector back to the global model and to every local copy holding it.
  - Item *insertion* (new ids) is still global/immediate so the candidate
    set and RNG sequence match the centralized run; only *values* diverge
    after the first aggregation -- by design, federation changes results.
"""

import time
import numpy as np
import logging
import params as _sp
from params import hidden_dim, reg_u, reg_v, lr, n_iter, k
from load_cache import (train_data, test_data)
from logs_paths import output_path

# aggregations = TOTAL number of FedAvg rounds over the whole stream
# (train + test). The rounds spread evenly via fractional boundaries, so
# exactly N_ROUNDS rounds run even when the division is not exact
# (e.g. 35 events / 20 rounds -> boundaries, not 35 modulo rounds).
N_ROUNDS = int(getattr(_sp, 'aggregations', 0) or 0)
_TOTAL_EVENTS = len(train_data) + len(test_data)
if N_ROUNDS > 0 and _TOTAL_EVENTS > 0:
    AGG_INTERVAL = max(1, _TOTAL_EVENTS // N_ROUNDS)  # informational
    AGG_BOUNDARIES = {round(_TOTAL_EVENTS * i / N_ROUNDS) for i in range(1, N_ROUNDS + 1)}
else:
    AGG_INTERVAL = 0  # 0 = no in-stream aggregation (final flush only)
    AGG_BOUNDARIES = set()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(output_path('fed_isgd', 'fed_isgd.log'), mode='w'),
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
    """Mirrors `items_global` insertion order with a C-contiguous matrix."""

    def __init__(self, capacity=4096):
        self.ids = []
        self.pos = {}
        self.mat = np.empty((capacity, hidden_dim), dtype=np.float64)
        self.n = 0

    def _grow(self):
        self.mat = np.concatenate([self.mat, np.empty_like(self.mat)], axis=0)

    def get_init(self, items_global, item_id, scale):
        vec = items_global.get(item_id)
        if vec is None:
            vec = scale * np.random.randn(hidden_dim)
            items_global[item_id] = vec
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


def _aggregate(state):
    """One FedAvg round over items touched since the last aggregation."""
    if not state['updated_items']:
        return
    t0 = time.perf_counter()
    items_global = state['items_global']
    items_local = state['items_local']
    cache = state['cache']
    updated_items = state['updated_items']
    item_users = state['item_users_updated']
    counts = state['pending_counts']
    n_aggs = 0
    for item_id in updated_items:
        users = item_users.get(item_id)
        if not users:
            continue
        total = 0
        acc = np.zeros(hidden_dim, dtype=np.float64)
        for uid in users:
            w = counts.get(uid, 0)
            if w <= 0:
                continue
            acc += w * items_local[uid][item_id]
            total += w
        if total <= 0:
            continue
        avg = acc / total
        items_global[item_id] = avg
        cache.write_back(item_id, avg)
        for uid, local in items_local.items():
            if item_id in local:
                local[item_id] = avg.copy()
        n_aggs += 1
    state['updated_items'] = set()
    state['item_users_updated'] = {}
    state['pending_counts'] = {}
    state['n_aggregations'] += 1
    dt = time.perf_counter() - t0
    state['agg_time'] += dt
    logger.info(f'[Agg {state["n_aggregations"]}] items={n_aggs} AggTime={dt:.6f}s')


def process_iter(state, uid, item_id, evaluate, scale):
    users = state['users']
    items_global = state['items_global']
    items_local = state['items_local']
    cache = state['cache']
    seen_online = state['seen_online']
    hit = None
    hit_rate_time = 0.0

    # Same RNG order as centralized: user first, then global item.
    user_vector = get_init(users, uid, scale)
    cache.get_init(items_global, item_id, scale)

    # Client's personal copy (plain copy of global, no RNG).
    local = items_local.get(uid)
    if local is None:
        local = {}
        items_local[uid] = local
    local_vec = local.get(item_id)
    if local_vec is None:
        local_vec = items_global[item_id].copy()
        local[item_id] = local_vec

    if evaluate:
        eval = time.perf_counter()
        n = cache.n
        p = cache.mat[:n] @ user_vector
        # Personalize: override with local copies where the client has them.
        for iid2, vec in local.items():
            pos = cache.pos.get(iid2)
            if pos is not None:
                p[pos] = np.dot(vec, user_vector)
        seen = seen_online.get(uid, set())
        if seen:
            seen_pos = [cache.pos[i] for i in seen if i in cache.pos]
            if seen_pos:
                p[seen_pos] = -np.inf
        kk = min(k, n)
        topk_pos = np.argpartition(-p, kk - 1)[:kk]
        _ids = cache.ids
        _hit = False
        for pos in topk_pos:
            if _ids[pos] == item_id:
                _hit = True
                break
        hit = _hit
        hit_rate_time = time.perf_counter() - eval

    new_user_vector, new_item_vector = user_item_update(user_vector, local_vec)
    users[uid] = new_user_vector
    local[item_id] = new_item_vector  # local-only; global syncs at aggregation

    state['updated_items'].add(item_id)
    state['item_users_updated'].setdefault(item_id, set()).add(uid)
    state['pending_counts'][uid] = state['pending_counts'].get(uid, 0) + 1

    seen_online.setdefault(uid, set()).add(item_id)

    state['obs_count'] += 1
    if state['obs_count'] in AGG_BOUNDARIES:
        _aggregate(state)

    return hit, hit_rate_time


def run_phase(data, phase, evaluate, state, scale, time_dataset):
    logger.info(f'[{phase}] starting | Time={time.perf_counter() - time_dataset:.2f}s')
    hits = 0
    total_predictions = 0
    total_hit_rate_time = 0.0

    phase_start = time.perf_counter()

    for uid, item_id in data:
        hit, hit_rate_time = process_iter(state, uid, item_id, evaluate, scale)

        if evaluate:
            if hit:
                hits += 1
            total_predictions += 1
            total_hit_rate_time += hit_rate_time

    logger.info(f'[{phase}] Finished | Time={time.perf_counter() - phase_start:.2f}s')
    if evaluate:
        logger.info(f'HR@{k}: {hits / total_predictions:.4f}')
        logger.info(f'Hit Rate Total Time: {total_hit_rate_time:.4f}s')
        logger.info(f'Hit Rate Avg Time: {total_hit_rate_time / total_predictions:.6f}s')


def make_state():
    return {
        'users': {},
        'items_global': {},
        'items_local': {},
        'cache': _ItemCache(),
        'seen_online': {},
        'updated_items': set(),
        'item_users_updated': {},
        'pending_counts': {},
        'obs_count': 0,
        'n_aggregations': 0,
        'agg_time': 0.0,
    }


def main():
    np.random.seed(42)
    time_dataset = time.perf_counter()

    scale = 0.1 / np.sqrt(hidden_dim)
    state = make_state()

    run_phase(train_data, 'TRAIN', evaluate=False, state=state,
              scale=scale, time_dataset=time_dataset)
    run_phase(test_data, 'TEST', evaluate=True, state=state,
              scale=scale, time_dataset=time_dataset)

    # Flush leftovers so all updates reach the global model.
    if state['updated_items']:
        _aggregate(state)

    logger.info(f'Total Time: {time.perf_counter() - time_dataset:.4f} seconds')
    logger.info(f'Aggregations: {state["n_aggregations"]} | Agg Total Time: {state["agg_time"]:.4f}s')
    logger.info(f'Total Users Seen: {len(state["users"])} | Total Items Seen: {len(state["items_global"])}')


if __name__ == '__main__':
    logger.info(
        f'\n'
        f'hidden_dim: {hidden_dim} | reg: {reg_u} | lr: {lr} | n_iter: {n_iter} | agg_rounds: {N_ROUNDS} | agg_interval: {AGG_INTERVAL}\n'
        f'Dataset size: {len(train_data) + len(test_data)} | Train size: {len(train_data)} | Test size: {len(test_data)}'
    )

    main()
