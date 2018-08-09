import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', required=True)
parser.add_argument('-o', required=True)
args = vars(parser.parse_args())

in_fn = args['i']
out_fn = args['o']

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import sys
import re

batches = []

with open(in_fn) as f:
  found_beginning = False
  for l in f:
    if not found_beginning:
      if l[:5] == 'Epoch':
        found_beginning = True
      else:
        continue
    elif l[:5] == 'Batch':
      m = re.search('\[.*\]', l)
      batches.append([float(s) for s in [s.rstrip(',') for s in m.group(0)[1:-1].split(' ')]])

batches = list(zip(*batches))
t = list(range(len(batches[0])))

plt.plot(t, batches[0], t, batches[1], t, batches[2], t, batches[3])
plt.legend(['0', '1', '2', '3'])
plt.title('loss')
plt.xlabel('batch number')
plt.savefig('{}_losses.png'.format(out_fn))
plt.close()

#plt.plot(t[1:], np.diff(batches[0]), t[1:], np.diff(batches[1]), t[1:],
#    np.diff(batches[2]), t[1:], np.diff(batches[3]))
#plt.legend(['0', '1', '2', '3'])
#plt.title('loss differences (difference from batch n-1 -> n)')
#plt.xlabel('batch number')
#plt.savefig('{}_diffs.png'.format(out_fn))
#plt.close()
