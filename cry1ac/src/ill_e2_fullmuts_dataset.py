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
inp_dir = _config.OUT_PLACE + f'ill_e_fullmuts/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

exp_design = pd.read_csv(_config.DATA_DIR + f'Badran2015_SraRunTable.csv')
wt_gt = open(_config.DATA_DIR + f'SP055-rpoZ-cMyc-Cry1Ac1-d123.fa').readlines()[1].strip()
rc_wt_gt = compbio.reverse_complement(wt_gt)

params = {
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
  mdf = pd.DataFrame()
  for t_idx, t in enumerate(ordered_time_strings):
    df = pd.read_csv(inp_dir + f'PACE_Cry1Ac_{t}_Illumina.csv', index_col = 0)

    df['Nucleotide and position'] = df['Mutation'] + ' ' + df['Group'].astype(str)

    dfs = df[['Nucleotide and position', 'Frequency']]
    dfs = dfs.rename(columns = {'Frequency': t_idx})

    if len(mdf) == 0:
      mdf['Nucleotide and position'] = df['Nucleotide and position']
      mdf[t_idx] = df['Frequency']
    else:
      mdf = mdf.merge(dfs, on = 'Nucleotide and position', how = 'outer')

    timer.update()

  om_df = mdf.fillna(value = 0)
  return om_df

##
# Primary
##
def form_simple_dataset():

  dataset_nm = 'ill_rl_100'

  # Form read groups
  read_groups = [
    [0, 1],
    [2],
    [3],
    [4],
    [5, 6],
    [7, 8, 9, 10, 11],
    [12, 13, 14],
    [15, 16],
    [17],
    [18],
  ]
  with open(out_dir + f'read_groups_{dataset_nm}.pkl', 'wb') as f:
    pickle.dump(read_groups, f)

  # Form obs_reads_pivot
  om_df = get_obs_marginal_df()
  om_df.to_csv(out_dir + f'obs_reads_pivot_{dataset_nm}.csv', index = False)  

  # Form proposed genotypes
  propose_genotypes(om_df, read_groups, dataset_nm)
  return

def form_risingstar_datasets():
  om_df = get_obs_marginal_df()

  read_groups = [
    [0, 1],
    [2],
    [3],
    [4],
    [5, 6],
    [7, 8, 9, 10, 11],
    [12, 13, 14],
    [15, 16],
    [17],
    [18],
  ]

  num_tcols = len(ordered_time_strings)

  for final_tcol in range(3, num_tcols):
    print(final_tcol)
    dataset_nm = f'ill_rl100_rs{final_tcol}'

    tidxs = list(range(0, final_tcol + 1))
    keep_cols = ['Nucleotide and position'] + tidxs

    om_dfs = om_df[keep_cols]
    om_dfs.to_csv(out_dir + f'obs_reads_pivot_{dataset_nm}.csv', index = False)

    propose_genotypes(om_dfs, read_groups, dataset_nm)

    with open(out_dir + f'read_groups_{dataset_nm}.pkl', 'wb') as f:
      pickle.dump(read_groups, f)

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

  ill_nms = exp_design[exp_design['Instrument'] == 'Illumina MiSeq']['Library Name']

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
  # form_simple_dataset()
  form_risingstar_datasets()

  # ill_nms = exp_design[exp_design['Instrument'] == 'Illumina MiSeq']['Library Name']

  # for nm in ill_nms:
  #   print(nm)
  #   form_simple_dataset(nm)

  return


if __name__ == '__main__':
  # if len(sys.argv) > 1:
  #   main(sys.argv[1:])
  # else:
  #   gen_qsubs()
  main([])
