import numpy as np

np.random.seed(42)

hidden_dim = 64
reg_u = 0.015
reg_v = 0.015
lr = 0.03
iter = 10

dataset = 'datasets/ml-32m'

aggregations = 0

k = 20  #para precision e ndcg

train_ratio = 0.2
validation_ratio = 0.2
test_ratio = 0.8
