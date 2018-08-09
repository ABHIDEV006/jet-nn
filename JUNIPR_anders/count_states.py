import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--in_fn', required = True)
parser.add_argument('-o', '--out_fn', required = True)
args = vars(parser.parse_args())

in_fn = args['in_fn']
out_fn = args['out_fn']

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

counts = []
first = True
jets = 0
with open(in_fn) as f:
  count = 0
  for l in f:
    if l[0] == 'J':
      jets += 1
      if first:
        first = False
      else:
        counts.append(count)
        count = 0
    elif l[0] == 'T':
      count += 1
  counts.append(count)


plt.hist(counts, bins=list(range(25)))
plt.title('Total states in tree ({} jets)'.format(jets))
plt.xlabel('number of states')
plt.xticks(list(range(25)))
plt.ylabel('frequency')
plt.savefig(out_fn)
plt.close()
