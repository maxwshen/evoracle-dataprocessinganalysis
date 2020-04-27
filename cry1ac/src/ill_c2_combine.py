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
inp_dir = _config.OUT_PLACE + f'ill_c_convert_aa/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

exp_design = pd.read_csv(_config.DATA_DIR + f'Badran2015_SraRunTable.csv')
wt_gt = open(_config.DATA_DIR + f'SP055-rpoZ-cMyc-Cry1Ac1-d123.fa').readlines()[1].strip()
rc_wt_gt = compbio.reverse_complement(wt_gt)

params = {
  'num_splits': 10,
}

##
# Primary
##
def merge_n_paired_reads(nm):
  mdf = pd.DataFrame()
  for split in range(params['num_splits']):
    df = pd.read_csv(inp_dir + f'{nm}_{split}.csv', index_col = 0)
    mdf = mdf.append(df, ignore_index = True, sort = False)

  mdf.to_csv(out_dir + f'{nm}.csv')
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
  for nm in ill_nms:
    command = f'python {NAME}.py {nm}'
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + f'q_{script_id}_{nm}.sh'
    with open(sh_fn, 'w') as f:
      f.write(f'#!/bin/bash\n{command}\n')
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append(f'qsub -V -P regevlab -l h_rt=10:00:00 -wd {_config.SRC_DIR} {sh_fn} &')

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
  # [nm] = args

  ill_nms = exp_design[exp_design['Instrument'] == 'Illumina MiSeq']['Library Name']

  for nm in ill_nms:
    print(nm)
    merge_n_paired_reads(nm)

  return


if __name__ == '__main__':
  # if len(sys.argv) > 1:
  #   main(sys.argv[1:])
  # else:
  #   gen_qsubs()
  main([])
