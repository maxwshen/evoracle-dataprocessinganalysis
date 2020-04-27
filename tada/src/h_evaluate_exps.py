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
inp_dir = _config.OUT_PLACE + f'g_collate_exps/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

true_df = None
idx_to_col = None

memoizer = {}

##
# 
##
def get_timewise_pearsonr(pred_df, true_df):
  time_cols = [col for col in true_df.columns if 'genotype' not in col]
  id_cols = [col for col in true_df.columns if 'genotype' in col]

  stats_dd = defaultdict(list)
  for t in time_cols:
    tdf = true_df[['Abbrev genotype', t]]
    pdf = pred_df[['Abbrev genotype', t]]

    mdf = tdf.merge(pdf, on = 'Abbrev genotype', how = 'outer', suffixes = ['_true', '_pred'])
    mdf = mdf.fillna(value = 0)

    from scipy.stats import pearsonr
    stats_dd['t'].append(t)
    stats_dd['r'].append(pearsonr(mdf[f'{t}_true'], mdf[f'{t}_pred'])[0])

  stats_df = pd.DataFrame(stats_dd)
  return stats_df


def get_overall_pearsonr(pred_df, true_df):
  time_cols = [col for col in true_df.columns if 'genotype' not in col]
  id_cols = [col for col in true_df.columns if 'genotype' in col]

  stats_d = dict()

  pdf = pred_df.melt(id_vars = id_cols, var_name = 'Time', value_name = 'Frequency')
  tdf = true_df.melt(id_vars = id_cols, var_name = 'Time', value_name = 'Frequency')

  mdf = tdf.merge(pdf, on = id_cols + ['Time'], how = 'outer', suffixes = ['_true', '_pred'])
  mdf = mdf.fillna(value = 0)

  from scipy.stats import pearsonr
  stats_d['r'] = pearsonr(mdf[f'Frequency_true'], mdf[f'Frequency_pred'])[0]

  return stats_d


def load_true_df(row):
  global true_df
  global idx_to_col

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
  
  if memoize_key in memoizer:
    true_df, idx_to_col = memoizer[memoize_key]
  else:
    print(f'Loading true df ...')
    true_df = pd.read_csv(_config.OUT_PLACE + f'data_multi/pv_groundtruth_{min_gt_fq}pct_t{threshold}_p{pace_num}.csv')
    true_df['Abbrev genotype'] = [full_to_abbrev(s) for s in true_df['Full genotype']]

    t_cols = [col for col in true_df.columns if 'genotype' not in col]
    idx_to_col = {str(idx): col for idx, col in enumerate(t_cols)}
    memoizer[memoize_key] = (true_df, idx_to_col)  
  return


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

def abbrev_to_full(abbrev_gt):
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


##
# Primary
##
def run_all(modelexp_nm, start, end, split_idx):
  print(modelexp_nm)

  tw_out_dir = out_dir + f'timewiseR_{modelexp_nm}/'
  util.ensure_dir_exists(tw_out_dir)

  exp_design = pd.read_csv(_config.DATA_DIR + f'{modelexp_nm}.csv')
  dg_nm = modelexp_nm.replace('modelexp_', '').replace('_rs', '')
  datagroup_df = pd.read_csv(_config.DATA_DIR + f'datagroup_{dg_nm}.csv', index_col = 0)
  hyperparam_cols = [col for col in exp_design.columns if col != 'Name']

  # For splitting
  exp_design = exp_design.iloc[start : end + 1]

  stats_dd = defaultdict(list)
  print(f'Calculating stats...')
  timer = util.Timer(total = len(exp_design))
  for idx, row in exp_design.iterrows():

    # Load true_df
    load_true_df(row)

    nm = row['Name']
    df_fn = f'{inp_dir}/{modelexp_nm}/genotype_matrix_{nm}.csv'
    if os.path.isfile(df_fn):

      stats_dd['Int name'].append(nm)
      stats_dd['Random seed'].append(row['random_seed'])
      stats_dd['Dataset'].append(row['dataset'])

      pred_df = pd.read_csv(df_fn, index_col = 0)
      pred_df['Abbrev genotype'] = pred_df.index
      pred_df['Full genotype'] = [abbrev_to_full(s) for s in list(pred_df['Abbrev genotype'])]

      pred_df = pred_df.rename(columns = idx_to_col)

      op_d = get_overall_pearsonr(pred_df, true_df)
      tw_df = get_timewise_pearsonr(pred_df, true_df)

      stats_dd['Pearsonr (overall)'].append(op_d['r'])

      summary_stats = tw_df['r'].describe()
      for stat in summary_stats.index:
        val = summary_stats[stat]
        stats_dd[f'TimewiseR {stat}'].append(val)

      tw_df.to_csv(tw_out_dir + f'tw_{nm}.csv')

    timer.update()

  stats_df = pd.DataFrame(stats_dd)

  if len(stats_df) == 0:
    print(f'Error: Found no completed experiments in this split')
    sys.exit(1)

  # Annotate with datagroup
  print(f'Annotating...')
  cols = [s for s in datagroup_df.columns if s != 'dataset']
  for col in cols:
    print(col)
    nm_to_col = {dataset: s for dataset, s in zip(datagroup_df['dataset'], datagroup_df[col])}
    stats_df[col] = [nm_to_col[dataset] for dataset in stats_df['Dataset']]

  stats_df.to_csv(out_dir + f'{modelexp_nm}_{split_idx}.csv')
  return


'''
  qsub
'''
def gen_qsubs(modelexp_nm = ''):
  # Generate qsub shell scripts and commands for easy parallelization
  print('Generating qsub scripts...')
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  if modelexp_nm == '':
    modelexp_nm = 'modelexp_simple_rs'

  print(f'Writing qsubs for {modelexp_nm}.')

  exp_design = pd.read_csv(_config.DATA_DIR + f'{modelexp_nm}.csv')
  num_splits = 60
  n = len(exp_design)
  split_size = (n // num_splits) + 1

  # Generate qsubs
  num_scripts = 0
  for idx in range(num_splits):
    start = idx * split_size
    end = (idx + 1) * split_size

    command = f'python {NAME}.py {modelexp_nm} {start} {end} {idx}'
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + f'q_{modelexp_nm}_{idx}.sh'
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append(f'qsub -V -P regevlab -l h_rt=16:00:00,h_vmem=4G -l os=RedHat7 -wd {_config.SRC_DIR} {sh_fn} &')

  # Save commands
  commands_fn = qsubs_dir + '_commands.sh'
  with open(commands_fn, 'w') as f:
    f.write('\n'.join(qsub_commands))

  subprocess.check_output(f'chmod +x {commands_fn}', shell = True)
  print(f'Wrote {num_scripts} shell scripts to {qsubs_dir}')
  return


##
# Main
##
@util.time_dec
def main(argv):
  print(NAME)

  [modelexp_nm, start, end, split_idx] = argv
  start, end, split_idx = int(start), int(end), int(split_idx)

  run_all(modelexp_nm, start, end, split_idx)

  return


if __name__ == '__main__':
  if len(sys.argv) > 2:
    main(sys.argv[1:])
  else:
    gen_qsubs(modelexp_nm = sys.argv[1])
    # print('Qsub not supported yet')
