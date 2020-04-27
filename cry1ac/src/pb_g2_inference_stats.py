# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, pickle
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util, compbio
import pandas as pd

import _fitness_from_reads_pt_multi

# Default params
inp_dir = _config.OUT_PLACE + f'pb_e2_dataset_multi/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

##
#
##
def get_stats(log_fn):
  '''
    ----------
    Epoch 0/999 at 2020-01-23 18:42:00.122453
    Mar 3.076E-01 Fit 9.652E-01 Fq -2.034E-06 FSp 6.367E+01 FSk -3.940E-01: 2.503E+00

    - Last epoch (999 or converged earlier?)
    - Time per epoch
    - 

  '''

  dd = defaultdict(list)
  dd['Epoch'] = []
  take = False
  with open(log_fn) as f:
    for i, line in enumerate(f):
      w = line.split()

      if len(w) == 0:
        continue

      if w[0] == 'Epoch':
        epoch_num = int(w[1].split('/')[0])
        if epoch_num == len(dd['Epoch']):
          take = True
        else:
          continue
        dd['Epoch'].append(epoch_num)
        dd['Date'].append(w[-2])
        dd['Time'].append(w[-1])

      if take:
        if w[0] == 'Mar':
          dd['Marginal loss'].append(float(w[1]))
          dd['Fitness loss'].append(float(w[3]))
          dd['Skew loss'].append(float(w[5]))
          dd['Epoch loss'].append(float(w[-1]))
        take = False

  stats = {
    'Last epoch': max(dd['Epoch']),
  }

  from datetime import datetime, timedelta
  epoch_times = []
  time_format = '%H:%M:%S.%f'
  for idx in range(len(dd['Time']) - 1):
    t1 = dd['Time'][idx]
    t2 = dd['Time'][idx + 1]
    tdelta = datetime.strptime(t2, time_format) - datetime.strptime(t1, time_format)
    num_ms = tdelta / timedelta(microseconds = 1)
    epoch_times.append(num_ms)

  if len(set(dd['Date'])) == 1:
    stats['Mean epoch time (microseconds)'] = np.mean(epoch_times)
    stats['Median epoch time (microseconds)'] = np.median(epoch_times)
  else:
    stats['Mean epoch time (microseconds)'] = np.nan
    stats['Median epoch time (microseconds)'] = np.nan

  try:
    log_df = pd.DataFrame(dd)
  except:
    import code; code.interact(local=dict(globals(), **locals()))
  return log_df, stats


##
# Main
##
@util.time_dec
def main(argv):
  print(NAME)

  modelexp_nm = argv[0]
  print(modelexp_nm)

  exp_design = pd.read_csv(_config.DATA_DIR + f'{modelexp_nm}.csv')
  hyperparam_cols = [col for col in exp_design.columns if col != 'Name']

  new_out_dir = out_dir + f'{modelexp_nm}/'
  util.ensure_dir_exists(new_out_dir)

  print(f'Collating experiments...')

  model_out_dir = _config.OUT_PLACE + f'_fitness_from_reads_pt_multi/{modelexp_nm}/'

  stats_dd = defaultdict(list)
  master_log_df = pd.DataFrame()

  timer = util.Timer(total = len(exp_design))
  for idx, row in exp_design.iterrows():
    int_nm = row['Name']
    real_nm = row['dataset']

    log_fn = model_out_dir + f'_log_{int_nm}.out'

    try:
      log_df, stats = get_stats(log_fn)
    except:
      print(log_fn)
      import code; code.interact(local=dict(globals(), **locals()))

    # Annotate exp log df
    log_df['Experiment name'] = real_nm
    log_df['Experiment index'] = int_nm
    master_log_df = master_log_df.append(log_df, ignore_index = True, sort = False)

    # Save single stats
    stats_dd['Experiment name'].append(real_nm)
    stats_dd['Experiment index'].append(int_nm)
    for key in stats:
      stats_dd[key].append(stats[key])

    timer.update()

  stats_df = pd.DataFrame(stats_dd)
  stats_df.to_csv(out_dir + f'stats_{modelexp_nm}.csv')

  master_log_df.to_csv(out_dir + f'master_log_df_{modelexp_nm}.csv')

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    print(f'Usage: python x.py <modelexp_nm>')
