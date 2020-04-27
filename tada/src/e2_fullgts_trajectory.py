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
inp_dir = _config.OUT_PLACE + f'e_all_fullgts_stats/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

design_df = pd.read_csv(_config.DATA_DIR + 'exp_design.csv')

params = {
  
}

##
# Primary
##
def get_trajectory():

  mdf = pd.DataFrame()
  timer = util.Timer(total = len(design_df))
  for nm in design_df['Short name']:
    df = pd.read_csv(inp_dir + f'{nm}.csv', index_col = 0)    

    # Filter
    df = df[df['Count'] >= 5]

    fq_col = f'Fq {nm}'
    df[fq_col] = df['Count'] / sum(df['Count'])
    df = df[['Full genotype', fq_col]]

    if len(mdf) == 0:
      mdf = df
    else:
      mdf = mdf.merge(df, on = 'Full genotype', how = 'outer')

    timer.update()

  mdf = mdf.fillna(value = 0)
  mdf.to_csv(out_dir + f'pv_fullgts_trajectory.csv')

  dfm = mdf.melt(id_vars = 'Full genotype', var_name = 'Sample name', value_name = 'Frequency')
  dfm['Sample'] = [int(s.split()[-1]) for s in dfm['Sample name']]
  dfm.to_csv(out_dir + f'mel_fullgts_trajectory.csv')


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
  get_trajectory()
  return


if __name__ == '__main__':
  # if len(sys.argv) > 1:
  #   main(sys.argv[1:])
  # else:
  #   gen_qsubs()
  main([])
