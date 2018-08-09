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

lrs = []
lr_batches = []
batches = []

with open(in_fn) as f:
  found_beginning = False
  for l in f:
    if l[:5] == 'Using':
      lrs.append(l.split(' ')[-1])
      if found_beginning:
        lr_batches.append(batches)
        batches = []
      else:
        found_beginning = True
    elif (not found_beginning) or (l[:5] == 'Epoch'):
      continue
    else:
      batches.append(float(l.split(' ')[2][:-1]))
  lr_batches.append(batches)


for i, lr in enumerate(lrs):
  t = list(range(len(lr_batches[i])))

  # loss plot
  plt.figure(1)
  plt.plot(t, lr_batches[i], label=lr)

  # loss diffs plot
  plt.figure(2)
  plt.plot(t[1:], np.diff(lr_batches[i]), label=lr)

plt.figure(1)
plt.title('loss')
plt.xlabel('batch_number')
plt.ylabel('loss')
plt.legend()
plt.savefig('{}_loss.png'.format(out_fn))
plt.close()

plt.figure(2)
plt.title('loss diffs')
plt.xlabel('batch_number')
plt.ylabel('marginal loss')
plt.legend()
plt.savefig('{}_loss_diff.png'.format(out_fn))
plt.close()
