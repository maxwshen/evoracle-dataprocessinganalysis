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
inp_dir = _config.OUT_PLACE + f'pb_e2_dataset_multi/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

exp_design = pd.read_csv(_config.DATA_DIR + f'Badran2015_SraRunTable.csv')
pacbio_nms = exp_design[exp_design['Instrument'] == 'PacBio RS II']['Library Name']
pacbio_nms = sorted(pacbio_nms)

##
# Functions
##
def load_proposed_genotypes(read_len, nm = 'smart'):
  inp_fn = inp_dir + f'propose_genotypes_{nm}_readlen_{read_len}.txt'
  with open(inp_fn) as f:
    lines = f.readlines()
  gts = [s.strip() for s in lines]
  return gts


##
# qsub
##
def gen_qsubs():
  # Generate qsub shell scripts and commands for easy parallelization
  print('Generating qsub scripts...')
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  sim_read_lens = [
    1,
    50,
    75,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
  ]

  num_scripts = 0
  for read_len in sim_read_lens:
    for random_seed in range(10):
      command = f'python {NAME}.py {read_len} {random_seed}'
      script_id = NAME.split('_')[0]

      # Write shell scripts
      sh_fn = qsubs_dir + f'q_{script_id}_{read_len}_{random_seed}.sh'
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

  read_len = argv[0]
  random_seed = int(argv[1])
  print(read_len)
  df = pd.read_csv(inp_dir + f'obs_reads_pivot_readlen_{read_len}.csv')

  genotypes_set = 'smart'
  proposed_genotypes = load_proposed_genotypes(read_len, nm = genotypes_set)

  read_groups = pickle.load(open(inp_dir + f'read_groups_readlen_{read_len}.pkl', 'rb'))

  exp_nm = f'pb_f2_{read_len}_{random_seed}'
  package = _fitness_from_reads_pt_multi.infer_fitness(
    df, 
    proposed_genotypes, 
    read_groups,
    exp_nm,
    random_seed = random_seed,
  )

  d = {
    'fitness': package[0],
    'fq_mat': package[1],
    'pred_marginals': package[2],
    'Proposed genotypes': proposed_genotypes,
  }

  out_dfs = {
    'fitness': pd.DataFrame({'Genotype': list(proposed_genotypes), 'Inferred fitness': d['fitness']}),
    'genotype_matrix': pd.DataFrame(d['fq_mat'].T, index = proposed_genotypes),
  }
  for df_nm in out_dfs:
    dfs = out_dfs[df_nm]
    d[df_nm] = dfs
    dfs.to_csv(out_dir + f'{df_nm}_readlen_{read_len}_{random_seed}.csv')

  with open(out_dir + f'readlen_{read_len}_{random_seed}.pkl', 'wb') as f:
    pickle.dump(d, f)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    gen_qsubs()

