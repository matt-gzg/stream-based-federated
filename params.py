import numpy as np

np.random.seed(42)

hidden_dim = 5
reg_u = 0.01
reg_v = 0.01
lr = 0.03
iter = 1
b_iter = 1

dataset = 'datasets/ml-test'

aggregations = 10

k = 20  #para precision e ndcg

train_ratio = 0.2
validation_ratio = 0.0
test_ratio = 0.8
