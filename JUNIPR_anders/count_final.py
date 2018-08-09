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
lastiline = ''
jets = 0
with open(in_fn) as f:
  for l in f:
    if l[0] == 'J':
      jets += 1
    elif l[0] == 'I':
      lastiline = l
    elif lastiline != '' and l[0] == 'P':
      counts.append(len(lastiline.split(' ')) -1)
      lastiline = ''


plt.hist(counts, bins=list(range(50)))
plt.title('Total states in tree ({} jets)'.format(jets))
plt.xlabel('number of states')
plt.xticks(list(range(0, 50, 5)))
plt.ylabel('frequency')
plt.savefig(out_fn)
plt.close()
