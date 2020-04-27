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
inp_dir_pb_e3 = _config.OUT_PLACE + f'pb_e3_interpolate_multi/'
inp_dir_infer = _config.OUT_PLACE + '_fitness_from_reads_pt_multi/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

fitness_df = pd.read_csv('/ahg/regevdata/projects/CRISPR-libraries/prj2/evolution/badran/out/_fitness_pt/fullgt_fitness.csv', index_col = 0)

##
# Functions
##
def eval(modelexp_nm, model_nm, fq_df, row, split_idx):
  '''
    Call rising stars
  '''
  model_fold = inp_dir_infer + f'{modelexp_nm}'

  pred_fitness_df = pd.read_csv(f'{model_fold}/model_{model_nm}/_final_fitness.csv', index_col = 0)
  pred_fq_df = pd.read_csv(f'{model_fold}/model_{model_nm}/_final_genotype_matrix.csv', index_col = 0)
  if len(pred_fq_df) == 0:
    return None

  last_t_col = pred_fq_df.columns[-1]
  pfqdfs = pred_fq_df.sort_values(by = last_t_col, ascending = False)
  consensus_gt = pfqdfs.iloc[0].name
  consensus_fq = pfqdfs[last_t_col].iloc[0]

  pred_consensus_fitness = pred_fitness_df[pred_fitness_df['Genotype'] == consensus_gt]['Inferred fitness'].iloc[0]

  # Predict consensus gt as rising star if fq < 0.75
  second_last_t_col = pred_fq_df.columns[-2]
  is_rising = pred_fq_df.loc[consensus_gt, last_t_col] - pred_fq_df.loc[consensus_gt, second_last_t_col]
  not_consensus = bool(pred_fq_df.loc[consensus_gt, last_t_col] < 0.75)
  if is_rising and not_consensus:
    pred_fitness_df['Rising star, predicted'] = (pred_fitness_df['Inferred fitness'] >= pred_consensus_fitness)
  else:
    pred_fitness_df['Rising star, predicted'] = (pred_fitness_df['Inferred fitness'] > pred_consensus_fitness)

  # Get observed rising stars
  last_t_col = pred_fq_df.columns[-1]
  second_last_t_col = pred_fq_df.columns[-2]

  ofqdfs = fq_df.sort_values(by = last_t_col, ascending = False)
  obs_consensus_gt = ofqdfs.iloc[0].name
  obs_is_rising = fq_df.loc[obs_consensus_gt, last_t_col] - fq_df.loc[obs_consensus_gt, second_last_t_col]
  obsnot_consensus = bool(fq_df.loc[obs_consensus_gt, last_t_col] < 0.75)

  obs_consensus_fitness = fitness_df[fitness_df['Genotype'] == obs_consensus_gt]['Fitness'].iloc[0]
  int_tp = row['data_risingstar_num']

  # Rising in last two timepoints
  crit = (fq_df[str(int_tp)] - fq_df[str(int_tp - 1)] > 0)
  keep_gts_crit3 = set(fq_df[crit].index)

  # Above 3% frequency in last timepoint
  last_timepoint_min_fq = 0.03
  keep_gts_crit4 = set(fq_df[fq_df[str(int_tp)] >= last_timepoint_min_fq].index)

  keep_gts = keep_gts_crit3 & keep_gts_crit4
  crit = (fitness_df['Genotype'].isin(keep_gts))
  dfs = fitness_df[crit]

  # Label most frequent gt as rising star if fq < 0.50 and fq > 0.50 later
  if obs_is_rising and obsnot_consensus:
    dfs['Rising star, observed'] = (dfs['Fitness'] >= obs_consensus_fitness)
  else:
    dfs['Rising star, observed'] = (dfs['Fitness'] > obs_consensus_fitness)
          
  eval_df = pred_fitness_df.merge(dfs, on = 'Genotype', how = 'outer')
  eval_df['Rising star, observed'] = eval_df['Rising star, observed'].fillna(value = False)
  eval_df['Rising star, predicted'] = eval_df['Rising star, predicted'].fillna(value = False)

  for item in row.index:
    eval_df[item] = row[item]
  eval_df.to_csv(out_dir + f'evals_{model_nm}.csv')
  return eval_df

##
# master
##
def eval_risingstar(modelexp_nm, start, end, split_idx):
  global out_dir
  out_dir = out_dir + f'{modelexp_nm}/'
  util.ensure_dir_exists(out_dir)

  exp_design = pd.read_csv(_config.DATA_DIR + f'{modelexp_nm}.csv')
  me_nm = modelexp_nm.replace('modelexp_', '').replace('_rs', '')
  datagroup_df = pd.read_csv(_config.DATA_DIR + f'datagroup_{me_nm}.csv')
  exp_design = exp_design.merge(datagroup_df, on = 'dataset')
  exp_design = exp_design.iloc[start : end + 1]

  fq_df = pd.read_csv(inp_dir_pb_e3 + 'badran_pacbio_pivot_1pct.csv')
  time_cols = [col for col in fq_df.columns if 'hrs' in col]
  fq_df = fq_df.rename(columns = {col: str(idx) for idx, col in enumerate(time_cols)})
  fq_df = fq_df.set_index('Abbrev genotype')

  stats_dd = defaultdict(list)
  timer = util.Timer(total = len(exp_design))
  for idx, row in exp_design.iterrows():
    model_nm = row['Name']

    eval_df = eval(modelexp_nm, model_nm, fq_df, row, split_idx)
    if eval_df is None:
      continue

    # Record stats
    for item in row.index:
      stats_dd[item].append(row[item])

    stats_dd['Num pred rising stars'].append(sum(eval_df['Rising star, predicted']))
    stats_dd['Num true rising stars'].append(sum(eval_df['Rising star, observed']))
    
    crit = (eval_df['Rising star, predicted'] == True) & (eval_df['Rising star, observed'] == True)
    stats_dd['True positive'].append(sum(crit))

    crit = (eval_df['Rising star, predicted'] == True) & (eval_df['Rising star, observed'] == False)
    stats_dd['False positive'].append(sum(crit))
    
    crit = (eval_df['Rising star, predicted'] == False) & (eval_df['Rising star, observed'] == False)
    stats_dd['True negative'].append(sum(crit))
    
    crit = (eval_df['Rising star, predicted'] == False) & (eval_df['Rising star, observed'] == True)
    stats_dd['False negative'].append(sum(crit))
    
    # AUROC, AUPRC
    eval_df['Inferred fitness'] = eval_df['Inferred fitness'].fillna(value = 0)

    from sklearn.metrics import roc_auc_score
    try:
      auroc = roc_auc_score(
        eval_df['Rising star, observed'],
        eval_df['Inferred fitness'],
      )
    except ValueError:
      auroc = np.nan
    stats_dd['AUROC'].append(auroc)

    from sklearn.metrics import average_precision_score
    try:
      aps = average_precision_score(
        eval_df['Rising star, observed'],
        eval_df['Inferred fitness'],
      )
    except ValueError:
      auroc = np.nan
    stats_dd['Average precision score'].append(aps)

    timer.update()

  # Last stats and save
  stats_df = pd.DataFrame(stats_dd)
  if len(stats_df) == 0:
    return

  stats_df['Sensitivity'] = stats_df['True positive'] / (stats_df['True positive'] + stats_df['False negative'])
  stats_df['Specificity'] = stats_df['True negative'] / (stats_df['True negative'] + stats_df['False positive'])
  stats_df['Precision'] = stats_df['True positive'] / (stats_df['True positive'] + stats_df['False positive'])
  stats_df['Recall'] = stats_df['True positive'] / (stats_df['True positive'] + stats_df['False negative'])
  stats_df.to_csv(out_dir + f'aggstats_{split_idx}.csv')

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
    modelexp_nm = 'modelexp_readlen_by_seed'

  print(f'Writing qsubs for {modelexp_nm}. OK?')
  input()

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

  eval_risingstar(modelexp_nm, start, end, split_idx)

  return


if __name__ == '__main__':
  if len(sys.argv) > 2:
    main(sys.argv[1:])
  else:
    # print(f'Usage: python x.py <modelexp_nm>')
    gen_qsubs(modelexp_nm = sys.argv[1])