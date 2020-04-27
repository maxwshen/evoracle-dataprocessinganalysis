# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd

# Default params
inp_dir = _config.DATA_DIR
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

exp_design = pd.read_csv(_config.DATA_DIR + f'Badran2015_SraRunTable.csv')

##
# Functions
##
def run_blasr(srr, nm):
  inp_fn = inp_dir + f'{srr}.fastq'
  genome_fn = inp_dir + 'SP055-rpoZ-cMyc-Cry1Ac1-d123.fa'

  out_fn = out_dir + f'{nm}.txt'
  command = f'blasr {inp_fn} {genome_fn} -m 0 > {out_fn}'

  subprocess.check_output(command, shell = True)

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

  pacbio_srrs = exp_design[exp_design['Instrument'] == 'PacBio RS II']['Run']
  pacbio_nms = exp_design[exp_design['Instrument'] == 'PacBio RS II']['Library Name']

  num_scripts = 0
  for srr, nm in zip(pacbio_srrs, pacbio_nms):
    command = f'python {NAME}.py {srr} {nm}'
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + f'q_{script_id}_{srr}.sh'
    with open(sh_fn, 'w') as f:
      f.write(f'#!/bin/bash\n{command}\n')
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append(f'qsub -V -P regevlab -l h_rt=4:00:00 -wd {_config.SRC_DIR} {sh_fn} &')

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
  [srr, nm] = args

  run_blasr(srr, nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    gen_qsubs()
