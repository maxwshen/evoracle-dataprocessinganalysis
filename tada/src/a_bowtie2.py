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
inp_dir = _config.OUT_PLACE + f'pr_pandaseq/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

design_df = pd.read_csv(_config.DATA_DIR + 'exp_design.csv')
ref = 'tada_context'

##
# Functions
##
def run_align_bowtie2(sample_name, nm, ref):
  r_fn = inp_dir + f'm_{nm}.fa'
  
  ref_out_dir = out_dir + f'{ref}/'
  util.ensure_dir_exists(ref_out_dir)
  out_fn = ref_out_dir + f'{nm}.sam'

  bowtie2 = '/ahg/regevdata/projects/CRISPR-libraries/tools/bowtie2-2.3.5.1-linux-x86_64/bowtie2'
  bt_index = _config.DATA_DIR + f'bowtie2_index/{ref}'

  command = f'{bowtie2} -x {bt_index} -f {r_fn} -S {out_fn}'
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

  num_scripts = 0
  for idx, row in design_df.iterrows():
    sample_id = row['Name']

    command = f'python {NAME}.py {sample_id}'
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + f'q_{script_id}_{sample_id}.sh'
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
def main(argv):
  print(NAME)
  
  # Function calls
  sample_name = argv[0]

  nm = design_df[design_df['Name'] == sample_name]['Short name'].iloc[0]
  print(nm, ref)

  run_align_bowtie2(sample_name, nm, ref)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    gen_qsubs()