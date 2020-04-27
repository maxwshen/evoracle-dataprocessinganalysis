# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, pickle
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util, compbio
import pandas as pd

# Default params
inp_dir = _config.OUT_PLACE + f'd_singlemuts/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

design_df = pd.read_csv(_config.DATA_DIR + 'exp_design.csv')
wt_gt = open(_config.DATA_DIR + f'tada_context.fa').readlines()[1].strip()
rc_wt_gt = compbio.reverse_complement(wt_gt)

major_single_muts = {
  '-76V': 'A-76V',
  '-73I': 'M-73I',
  '15W': 'C15W',
  '68S': 'F68S',
  '198G': 'R198G',
  '286D': 'G286D',
  '304N': 'T304N',
  '332G': 'E332G',
  '344E': 'A344E',
  '347R': 'Q347R',
  '361I': 'T361I',
  '363P': 'S363P',
  '384Y': 'D384Y',
  '404C': 'S404C',
  '417D': 'N417D',
  '461K': 'E461K',
  '463S': 'N463S',
  '515K': 'E515K',
  '582L': 'S582L',
}

ordered_time_strings = [
  '0hrs',
  '12hrs',
  '24hrs',
  '36hrs',
  '48hrs',
  '60hrs',
  '72hrs',
  '84hrs',
  '96hrs',
  '108hrs',
  '120hrs',
  '132hrs',
  '144hrs',
  '156hrs',
  '168hrs',
  '180hrs',
  '192hrs',
  '204hrs',
  '216hrs',
  '228hrs',
  '240hrs',
  '264hrs',
  '276hrs',
  '300hrs',
  '324hrs',
  '348hrs',
  '372hrs',
  '396hrs',
  '408hrs',
  '432hrs',
  '456hrs',
  '480hrs',
  '504hrs',
  '528hrs',
]

##
# Helper
##
def propose_genotypes(obs_marginals, groups, dataset_nm, style = 'smart'):
  '''
    Proposes genotypes.
    -> proposed_genotypes.txt
  '''
  from pb_e2_dataset_multi import get_smart_genotypes

  gt_sets = {
    'smart': get_smart_genotypes(obs_marginals, groups),
  }

  set_smart = set(gt_sets['smart'])

  gts = gt_sets['smart']

  print(f'Writing genotypes -- {len(gts)}')
  with open(out_dir + f'propose_genotypes_{dataset_nm}.txt', 'w') as f:
    for gt in gts:
      f.write(f'{gt}\n')
  return len(gts)


def get_obs_marginal_df():
  dd = defaultdict(list)
  print(f'Forming obs. marginal df ...')
  timer = util.Timer(total = len(ordered_time_strings))
  for t_idx, t in enumerate(ordered_time_strings):
    df = pd.read_csv(inp_dir + f'PACE_Cry1Ac_{t}_Illumina.csv', index_col = 0)

    # df = df[df['Mutation'].isin(major_single_muts)]
    df['Position'] = [s[:-1] for s in df['Mutation']]

    for mut_idx, mut in enumerate(major_single_muts):
      pos = mut[:-1]
      dfs = df[df['Position'] == pos]

      # Ignore other mutations: calculate (Specific mutation / (specific mutation + wt))
      mut_fq = dfs[dfs['Mutation'] == mut]['Frequency'].iloc[0]
      all_mut_fq = sum(dfs['Frequency'])
      wt_fq = 1 - all_mut_fq
      mut_fq = mut_fq / (mut_fq + wt_fq)

      mut_aa = mut[-1]

      dd['Time index'].append(t_idx)
      dd['Nucleotide and position'].append(f'{mut_aa} {mut_idx}')
      dd['Frequency'].append(mut_fq)

      dd['Time index'].append(t_idx)
      dd['Nucleotide and position'].append(f'. {mut_idx}')
      dd['Frequency'].append(1 - mut_fq)

    timer.update()

  om_df = pd.DataFrame(dd)
  om_df = om_df.pivot(index = 'Nucleotide and position', columns = 'Time index', values = 'Frequency')

  pref_order = [
    '. 0',
    'V 0',
    '. 1',
    'I 1',
    '. 2',
    'W 2',
    '. 3',
    'S 3',
    '. 4',
    'G 4',
    '. 5',
    'D 5',
    '. 6',
    'N 6',
    '. 7',
    'G 7',
    '. 8',
    'E 8',
    '. 9',
    'R 9',
    '. 10',
    'I 10',
    '. 11',
    'P 11',
    '. 12',
    'Y 12',
    '. 13',
    'C 13',
    '. 14',
    'D 14',
    '. 15',
    'K 15',
    '. 16',
    'S 16',
    '. 17',
    'K 17',
    '. 18',
    'L 18',
  ]
  om_df = om_df.loc[pref_order]

  om_df['Nucleotide and position'] = om_df.index
  return om_df


##
# Primary
##
def form_dataset():
  '''
    obs_reads_pivot
    proposed_genotypes
    read_groups
  '''

  dataset_nm = 'ill_rl_1'

  # Form read groups
  read_groups = [[s] for s in range(len(major_single_muts))]
  with open(out_dir + f'read_groups_{dataset_nm}.pkl', 'wb') as f:
    pickle.dump(read_groups, f)

  # Form obs_reads_pivot
  om_df = get_obs_marginal_df()
  om_df.to_csv(out_dir + f'obs_reads_pivot_{dataset_nm}.csv', index = False)

  # Form proposed genotypes
  propose_genotypes(om_df, read_groups, dataset_nm)

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
  # [nm] = args
  form_dataset()

  return


if __name__ == '__main__':
  # if len(sys.argv) > 1:
  #   main(sys.argv[1:])
  # else:
  #   gen_qsubs()
  main([])
