# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util, compbio
import pandas as pd

# Default params
inp_dir = _config.OUT_PLACE + f'f2_majormuts_trajectory/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

design_df = pd.read_csv(_config.DATA_DIR + 'exp_design.csv')

params = {
  'major_thresholds': [5, 8, 10],  
}

##
# Merging
##
aligned_highpance = {
    '1': [41, 42, 45, 44, 46, 47, 48, 49, 50],
    '2': [51, 52, 55, 54, 56, 57, 58, 59, 60],
    '3': [61, 62, 65, 64, 65, 66, 67, 68, 69],
    '4': [70, 71, 72, 73, 74, 75, 76, 77, 78],
}
def merge_highpances(df):
  num_samples = len(aligned_highpance['1'])
  
  new_df = pd.DataFrame()
  timer = util.Timer(total = num_samples)
  for idx in range(num_samples):
    new_samplenm = f'Fq {41 + idx}'

    samples = [f'Fq {aligned_highpance[highpance][idx]}' for highpance in aligned_highpance]
    dfs = df[df['Sample name'].isin(samples)]
    
    pv_df = dfs.pivot(index = 'Full genotype', columns = 'Sample name', values = 'Frequency')
    pv_df = pv_df.fillna(value = 0)
    pv_df['Mean fq'] = pv_df.apply(np.mean, axis = 'columns')
    pv_df['Mean fq'] /= sum(pv_df['Mean fq'])
    pv_df['Full genotype'] = pv_df.index
    pv_df = pv_df[['Full genotype' , 'Mean fq']]
    
    dfm = pv_df.melt(id_vars = ['Full genotype'], value_name = 'Frequency')
    dfm['Sample name'] = new_samplenm
    dfm['Sample'] = 41 + idx
    
    new_df = new_df.append(dfm, ignore_index = True)
    timer.update()

  return new_df


aligned_lowpance = {
    '1': list(range(1, 20 + 1)),
    '2': list(range(21, 40 + 1)),
}
def merge_lowpances(df):
  num_samples = len(aligned_lowpance['1'])
  
  new_df = pd.DataFrame()
  timer = util.Timer(total = num_samples)
  for idx in range(num_samples):
    new_samplenm = f'Fq {1 + idx}'

    samples = [f'Fq {aligned_lowpance[pance][idx]}' for pance in aligned_lowpance]
    dfs = df[df['Sample name'].isin(samples)]
    
    pv_df = dfs.pivot(index = 'Full genotype', columns = 'Sample name', values = 'Frequency')
    pv_df = pv_df.fillna(value = 0)
    pv_df['Mean fq'] = pv_df.apply(np.mean, axis = 'columns')
    pv_df['Mean fq'] /= sum(pv_df['Mean fq'])
    pv_df['Full genotype'] = pv_df.index
    pv_df = pv_df[['Full genotype' , 'Mean fq']]
    
    dfm = pv_df.melt(id_vars = ['Full genotype'], value_name = 'Frequency')
    dfm['Sample name'] = new_samplenm
    dfm['Sample'] = 1 + idx
    
    new_df = new_df.append(dfm, ignore_index = True)
    timer.update()
    
  return new_df



##
# Primary
##
def merge_regimes(major_threshold):
  df = pd.read_csv(inp_dir + f'pv_trajectory_t{major_threshold}.csv', index_col = 0)
  dfm = df.melt(id_vars = 'Full genotype', var_name = 'Sample name', value_name = 'Frequency')
  dfm['Sample'] = [int(s.split()[-1]) for s in dfm['Sample name']]

  pace_regimes = {
    '1': list(range(79, 85 + 1)),
    '2': list(range(86, 92 + 1)),
    '3': list(range(93, 99 + 1)),
  }

  for pace_idx in pace_regimes:
    print(f'PACE idx: {pace_idx}')
    pace_range = pace_regimes[pace_idx]

    low_df = merge_lowpances(dfm)
    high_df = merge_highpances(dfm)
    pace_df = dfm[dfm['Sample'].isin(pace_range)]

    merged_df = low_df.append(high_df, ignore_index = True, sort = False)
    merged_df = merged_df.append(pace_df, ignore_index = True, sort = False)
    merged_df['Sample'] = [str(s).zfill(2) for s in merged_df['Sample']]

    merged_df.to_csv(out_dir + f'mel_trajectory_t{major_threshold}_p{pace_idx}.csv')

    pv_df = merged_df.pivot(index = 'Full genotype', columns = 'Sample', values = 'Frequency')
    pv_df.to_csv(out_dir + f'pv_trajectory_t{major_threshold}_p{pace_idx}.csv')

  return


##
# qsub
##
def gen_qsubs():
  # Generate qsub shell scripts and commands for easy parallelization
  print('Generating qsub scripts...')
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  num_scripts = 0
  for idx, row in design_df.iterrows():
    nm = row['Short name']

    command = f'python {NAME}.py {nm}'
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + f'q_{script_id}_{nm}.sh'
    with open(sh_fn, 'w') as f:
      f.write(f'#!/bin/bash\n{command}\n')
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append(f'qsub -V -P regevlab -l h_rt=10:00:00,h_vmem=4G -wd {_config.SRC_DIR} {sh_fn} &')

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
def main(args):
  print(NAME)
  
  # Function calls
  for major_threshold in params['major_thresholds']:
    print(f'Threshold %: {major_threshold}')
    merge_regimes(major_threshold)

  return


if __name__ == '__main__':
  # if len(sys.argv) > 1:
  #   main(sys.argv[1:])
  # else:
  #   gen_qsubs()
  main([])
