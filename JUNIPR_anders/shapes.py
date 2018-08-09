import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--q_granularity', default=21, type=int)
parser.add_argument('-p', '--p_granularity', default=10, type=int)
parser.add_argument('-b', '--batch_size', default=100, type=int)
parser.add_argument('-n', '--n_events', default=10**4, type=int)
parser.add_argument('-d', '--data_path', default='../../data/junipr/final_reclustered_practice.out')
parser.add_argument('--down_path', default='../../data/junipr/d_jets.out')
parser.add_argument('--up_path', default='../../data/junipr/u_jets.out')
parser.add_argument('--down_model',
                default='../../data/junipr/d_training/p10_q21/.out')
parser.add_argument('--up_model', default='../../data/junipr/u_training/p10_q21/.out')
parser.add_argument('-x', '--times', action='store_true')
args = vars(parser.parse_args())

# Setup parameters
p_granularity = args['p_granularity']
q_granularity = args['q_granularity']
batch_size = args['batch_size']

n_events = args['n_events']

# path to where to save data
path = args['data_path']
# boolean indicating if we are in the p times q or p plus q framework
times = args['times']
from utilities import load_data
import numpy as np


[daughters, endings, mothers, (discrete_p_splittings, discrete_q_splittings), mother_momenta] = load_data(path, 
    n_events=n_events, batch_size=batch_size, split_p_q=True,
    p_granularity=p_granularity, q_granularity=q_granularity)

#print(len(mothers[0]))
print(np.nonzero(mothers[0][0][0,1]))
print(endings[0][0][0,2])
print(discrete_p_splittings[0][0][0,2].shape)

