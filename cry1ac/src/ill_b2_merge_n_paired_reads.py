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
inp_dir = _config.OUT_PLACE + f'ill_b_call_mutations/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

exp_design = pd.read_csv(_config.DATA_DIR + f'Badran2015_SraRunTable.csv')
wt_gt = open(_config.DATA_DIR + f'SP055-rpoZ-cMyc-Cry1Ac1-d123.fa').readlines()[1].strip()
rc_wt_gt = compbio.reverse_complement(wt_gt)

params = {
}

##
#
##
def build_nm_to_idxs(df):
  '''
    Exploits ordered structure
  '''
  print(f'Building index ...')
  d = dict()
  curr_nm = ''
  start_idx = -1
  timer = util.Timer(total = len(df))
  for idx, row in df.iterrows():
    nm = row['Read name']
    if nm != curr_nm:
      if curr_nm != '':
        d[curr_nm] = {
          'start_idx': start_idx,
          'end_idx': idx,
        }
      start_idx = idx
      curr_nm = nm
    timer.update()

  # Last load
  d[curr_nm] = {
    'start_idx': start_idx,
    'end_idx': idx,
  }

  return d


##
# Primary
##
def merge_n_paired_reads(nm, split):
  df = pd.read_csv(inp_dir + f'{nm}_read_idxs.csv', index_col = 0)

  read_nms = list(sorted(set(df['Read name'])))

  # Parallelization: split by read names
  num_nms = len(read_nms)
  tot_splits = 10
  split_size = int(num_nms * (1 / tot_splits)) + 1
  begin_idx = split_size * int(split)
  end_idx = split_size * (int(split) + 1)
  read_nms = read_nms[begin_idx : end_idx]

  num_non_overlapping = 0

  ds = build_nm_to_idxs(df)

  mdf = pd.DataFrame()
  dd = defaultdict(list)
  timer = util.Timer(total = len(read_nms))
  for read_nm in read_nms:
    timer.update()
    # dfs = df[df['Read name'] == read_nm]
    idxer = ds[read_nm]
    dfs = df.iloc[idxer['start_idx'] : idxer['end_idx']]

    if len(dfs) != 2:
      mdf = mdf.append(dfs, ignore_index = True, sort = False)
      continue

    r1 = dfs.iloc[0]
    r2 = dfs.iloc[1]

    if r1['Read start idx'] <= r2['Read start idx'] <= r1['Read end idx']:
      dd['Read name'].append(r1['Read name'])
      read_len = r2['Read end idx'] - r1['Read start idx']
      dd['Read length'].append(read_len)
      dd['Read start idx'].append(r1['Read start idx'])
      dd['Read end idx'].append(r2['Read end idx'])
    elif r2['Read start idx'] <= r1['Read start idx'] <= r2['Read end idx']:
      dd['Read name'].append(r1['Read name'])
      read_len = r1['Read end idx'] - r2['Read start idx']
      dd['Read length'].append(read_len)
      dd['Read start idx'].append(r2['Read start idx'])
      dd['Read end idx'].append(r1['Read end idx'])
    else:
      mdf = mdf.append(dfs, ignore_index = True, sort = False)
      num_non_overlapping += 1

    pass

  print(f'Among {len(read_nms)} reads, found {num_non_overlapping} not overlapping: {num_non_overlapping / len(read_nms):.2%}')

  ndf = pd.DataFrame(dd)
  mdf = mdf.append(ndf, ignore_index = True, sort = False)

  mdf.to_csv(out_dir + f'{nm}_{split}_read_idxs.csv')

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
    for split in range(10):
      command = f'python {NAME}.py {nm} {split}'
      script_id = NAME.split('_')[0]

      # Write shell scripts
      sh_fn = qsubs_dir + f'q_{script_id}_{nm}_{split}.sh'
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
  [nm, split] = args

  merge_n_paired_reads(nm, split)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    gen_qsubs()
