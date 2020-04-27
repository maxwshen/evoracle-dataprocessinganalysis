#
import _config
import sys, os, fnmatch, datetime, subprocess, pickle
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util, compbio
import pandas as pd


true_df_memoizer = {}

##
# Utility
##
def load_true_df_from_row(row):
  options = {}
  dnm = row['dataset']
  ws = dnm.split('--')
  for w in ws:
    cs = w.split('-')
    if len(cs) == 2:
      options[cs[0]] = cs[1]
      # print(cs[0], cs[1])

  min_gt_fq = options['min_gt_frequency']
  threshold = options['threshold']
  pace_num = options['pace_num']

  memoize_key = f'{min_gt_fq}_{threshold}_{pace_num}'
  
  if memoize_key in true_df_memoizer:
    true_df, idx_to_col = true_df_memoizer[memoize_key]
  else:
    print(f'Loading true df ...')
    true_df = pd.read_csv(_config.OUT_PLACE + f'data_multi/pv_groundtruth_{min_gt_fq}pct_t{threshold}_p{pace_num}.csv')
    true_df['Abbrev genotype'] = [full_to_abbrev(s) for s in true_df['Full genotype']]

    t_cols = [col for col in true_df.columns if 'genotype' not in col]
    idx_to_col = {str(idx): col for idx, col in enumerate(t_cols)}
    true_df_memoizer[memoize_key] = (true_df, idx_to_col)
  return {
    'true_df': true_df, 
    'idx_to_col': idx_to_col, 
    'memoize_key': memoize_key,
  }


def full_to_abbrev(full_gt):
  '''
    1 I,23 L,26 .,48 .,74 .,77 .,82 .,86 .,87 .,
    IL.......
  '''
  ws = full_gt.split(',')
  agt = ''
  for w in ws:
    agt += w.split()[-1]
  return agt

def abbrev_to_full(abbrev_gt, true_df):
  '''
    IL.......
    1 I,23 L,26 .,48 .,74 .,77 .,82 .,86 .,87 .,
  '''
  ex_gt = true_df['Full genotype'].iloc[0]
  ws = ex_gt.split(',')
  full_gt = []
  for idx, char in enumerate(abbrev_gt):
    w = ws[idx]
    pos = w.split()[0]
    full_gt.append(f'{pos} {char}')
  return ','.join(full_gt)