import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--q_granularity', default=21, type=int)
parser.add_argument('-p', '--p_granularity', default=10, type=int)
parser.add_argument('-n', '--n_events', default=10**5, type=int)
parser.add_argument('-d', '--data_path', default='../../data/junipr/d_0.03_charge')
parser.add_argument('-i', '--input_path', default='../../data/junipr/final_reclustered_d_jets_0.03.out')
args = vars(parser.parse_args())

# Setup parameters
p_granularity = args['p_granularity']
q_granularity = args['q_granularity']

n_events = args['n_events']

# path to where to save data
data_path = args['data_path']

# path to input jets file
input_path = args['input_path']

import JUNIPR_class
from utilities import load_data

def compile_jets(data_path, n_events, p_granularity, q_granularity, batch_size):
  # Load in jets from file
  [daughters, endings, mothers, (discrete_p_splittings, discrete_q_splittings), mother_momenta] = load_data(data_path, 
      n_events=n_events, batch_size=batch_size, split_p_q=True,
      p_granularity=p_granularity, q_granularity=q_granularity)

  # this unpacking is necessary to remove it from the tuple and put it into a
  # list
  x = [[*a] for a in zip(daughters, mother_momenta, [m[1] for m in mothers], [q[0] for q in discrete_q_splittings])]

  # temporary hack having to do with mask values; this will change later.
  for i in range(len(mothers)):
    mothers[i][0][mothers[i][0]==-1] = 0
    discrete_p_splittings[i][0][discrete_p_splittings[i][0]==p_granularity**4] = 0
    discrete_q_splittings[i][0][discrete_q_splittings[i][0]==q_granularity] = 0

  y = [[*a] for a in zip([e[0] for e in endings], [m[0] for m in mothers], [d[0] for d in
      discrete_p_splittings], [q[0] for q in discrete_q_splittings])]
  return x, y

x_10, y_10 = compile_jets(data_path=input_path, n_events=n_events,
    q_granularity=q_granularity, p_granularity=p_granularity, batch_size=10)
#x_100, y_100 = compile_jets(data_path=input_path, n_events=n_events,
#    q_granularity=q_granularity, p_granularity=p_granularity, batch_size=100)

model = JUNIPR_class.JUNIPR_charge(p_granularity, q_granularity)
#model.train([10**(-x) for x in [2, 3, 4, 3, 4, 5]], [5, 1, 1, 1, 1, 1], 
model.train([10**(-x) for x in [2, 3, 4, 5]], [2, 2, 1, 1], 
  [*[x_10]*6], [*[y_10]*6], data_path)
