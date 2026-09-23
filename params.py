import numpy as np

np.random.seed(42)

hidden_dim = 5
reg_u = 0.015
reg_v = 0.015
lr = 0.03
n_iter = 10

dataset = 'datasets/ml-100k'

aggregations = 20

k = 20

train_ratio = 0.2
validation_ratio = 0.0
test_ratio = 0.8
