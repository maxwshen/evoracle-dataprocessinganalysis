# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, math, pickle, imp
sys.path.append('/home/unix/maxwshen/')
import fnmatch
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd

# Default params
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

'''
  Intention: Use this only to vary model parameters on a limited number of datasets (manually specify here).
  To run experiments on a large set of datasets, refer to exp dfs produced by pb_e2_dataset_multi.py
'''
hparams = {
  # 'random_seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  # 'random_seed': [0, 1, 2, 3, 4, 5, 6, 7, 8],
  'random_seed': [0],
  # 'random_seed': list(range(0, 50 + 1)),
  # 'random_seed': [0, 1, 2, 3, 4],

  # 'alpha_marginal': [
  #   0.1, 
  #   0.3, 
  #   0.5,
  #   1, 
  #   3, 
  #   5,  # preferred
  #   10,
  #   30,
  #   50,
  #   100,
  # ],

  # 'beta_skew': [
  #   0,
  #   0.0001,
  #   0.0003,
  #   0.001,
  #   0.003,
  #   0.01, # preferred
  #   0.03,
  #   0.1,
  #   0.3,
  #   1,
  # ],
}

##
# 
##
def load_data_group(group_nm):
  df = pd.read_csv(_config.DATA_DIR + f'datagroup_{group_nm}.csv', index_col = 0)

  global hparams
  hparams['dataset'] = list(df['dataset'])

  return

def gen_recursive(hparams, keys):
  if len(keys) == 1:
    return [[s] for s in hparams[keys[0]]]
  else:
    ll = gen_recursive(hparams, keys[:-1])
    curr_key = keys[-1]
    new_ll = []
    for l in ll:
      for item in hparams[curr_key]:
        new_ll.append(l + [item])
    return new_ll

def gen_modeling_exp(dataset_nm):

  dd = defaultdict(list)
  keys = list(hparams.keys())
  print(keys)
  ll = gen_recursive(hparams, keys)

  num_exps = len(ll)
  df = pd.DataFrame(ll, index = list(range(num_exps)))
  # df.columns = keys

  print(f'Generated {len(df)} experiments.')
  df.columns = keys
  df['Name'] = df.index

  if dataset_nm == 'varynoisev2_rs':
    print(f'Using noiserep as random seed')
    df['random_seed'] = list(range(50)) * 17

  print(f'Generated modelexp_{dataset_nm}.csv with {len(df)} experiments')
  df.to_csv(_config.DATA_DIR + f'modelexp_{dataset_nm}.csv', index = False)

  return


@util.time_dec
def main(dataset_nm = 'temp'):
  print(NAME)

  # Function calls
  print(dataset_nm)

  # group_nm = 'simple'
  # group_nm = 'varynoise'
  # group_nm = 'varyproposals'
  # load_data_group(group_nm)
  load_data_group(dataset_nm)

  gen_modeling_exp(f'{dataset_nm}_rs')

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(dataset_nm = sys.argv[1])
  else:
    print(f'Usage: python _gen_modeling_exp.py <dataset_nm>')

