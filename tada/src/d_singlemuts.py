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
inp_dir_c = _config.OUT_PLACE + f'c2_combine/'
inp_dir_b2 = _config.OUT_PLACE + f'b22_combine/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

design_df = pd.read_csv(_config.DATA_DIR + 'exp_design.csv')
wt_gt = open(_config.DATA_DIR + f'tada_context.fa').readlines()[1].strip()
rc_wt_gt = compbio.reverse_complement(wt_gt)

params = {
  'num_splits': 8,
}

##
# Primary
##
def obtain_single_muts(nm):
  mut_df = pd.read_csv(inp_dir_c + f'{nm}.csv', index_col = 0)

  # print('Forming pos to ref dict ...')
  # pos_to_ref = {row['Position']: row['Reference amino acid'] for idx, row in mut_df.iterrows()}

  print('Pivoting ...')
  pvm_df = mut_df.pivot(index = 'Read name', columns = 'Position', values = 'Mutated amino acid')

  dd = defaultdict(list)
  timer = util.Timer(total = len(pvm_df.columns))
  for pos in pvm_df.columns:
    col = pvm_df[pos]

    from collections import Counter
    counts = Counter(col.astype(str))

    n = len(pvm_df)

    all_muts = [s for s in counts.keys() if s != 'nan']
    for mut in all_muts:
      mut_count = counts[mut]
      mut_frac = mut_count / n if n > 0 else np.nan

      # ref_aa = pos_to_ref[pos]
      # dd['Mutation'].append(f'{ref_aa}{pos}{mut}')
      dd['Mutation'].append(f'{pos}{mut}')
      dd['Frequency'].append(mut_frac)
      dd['Total count'].append(n)
      dd['Allele count'].append(mut_count)

    timer.update()

  df = pd.DataFrame(dd)
  df.to_csv(out_dir + f'{nm}.csv')

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
  [nm] = args
  obtain_single_muts(nm)

  # for nm in ill_nms:
  #   print(nm)
  #   obtain_single_muts(nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    gen_qsubs()
  # main([])
