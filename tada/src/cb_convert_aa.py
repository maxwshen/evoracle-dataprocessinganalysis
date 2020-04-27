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
inp_dir = _config.OUT_PLACE + f'b2_convert_sam_to_text/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

design_df = pd.read_csv(_config.DATA_DIR + 'exp_design.csv')
wt_gt = open(_config.DATA_DIR + f'tada_context.fa').readlines()[1].strip()
rc_wt_gt = compbio.reverse_complement(wt_gt)

params = {
}

##
# Functions
##
def create_gt_with_mutations(dfs):
  plus_strand = True
  gt = list(wt_gt) if plus_strand else list(rc_wt_gt)

  for idx, row in dfs.iterrows():
    try:
      assert gt[row['Position (0 based)']] == row['Reference nt'], 'Error: Probably bad strand'
    except:
      import code; code.interact(local=dict(globals(), **locals()))
    gt[row['Position (0 based)']] = row['Mutated nt']
  gt = ''.join(gt)
  return gt if plus_strand else compbio.reverse_complement(gt)

def translate_tada(gt):
  from Bio.Seq import Seq
  start_pos = 64
  end_pos = 565
  gt = gt[start_pos : end_pos]
  return compbio.translate(gt)


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
def convert(nm, split):
  df = pd.read_csv(inp_dir + f'tada_context/{nm}.csv', index_col = 0)

  # Drop duplicates: we edit list gt in place. Paired end reads can yield duplicate mutation calls
  print(f'Starting with {len(df)} mutations ...')
  df = df.drop_duplicates(subset = ['Position (0 based)', 'Read name'])
  print(f'Subsetted to {len(df)} mutations ...')
  df = df.reset_index(drop = True)

  read_nms = list(sorted(set(df['Read name'])))

  # Parallelization: split by read names
  num_nms = len(read_nms)
  tot_splits = 10
  split_size = int(num_nms * (1 / tot_splits)) + 1
  begin_idx = split_size * int(split)
  end_idx = split_size * (int(split) + 1)
  read_nms = read_nms[begin_idx : end_idx]

  all_aas = []

  ds = build_nm_to_idxs(df)

  print(f'Translating reads with mutations to amino acid sequences...')
  timer = util.Timer(total = len(read_nms))
  for read_nm in read_nms:
    # dfs = df[df['Read name'] == read_nm]
    idxer = ds[read_nm]
    dfs = df.iloc[idxer['start_idx'] : idxer['end_idx']]

    try:
      assert len(set(dfs['Read name'])) == 1 and read_nm in set(dfs['Read name'])
    except:
      import code; code.interact(local=dict(globals(), **locals()))

    gt = create_gt_with_mutations(dfs)
    aas = translate_tada(gt)
    all_aas.append(aas)
    timer.update()

  wt_aa = translate_tada(wt_gt)

  print(f'Summarizing non-synonymous amino acid mutations...')
  dd = defaultdict(list)
  idx_to_relative_pos = lambda idx: idx + 1
  timer = util.Timer(total = len(all_aas))
  for read_nm, obs_seq in zip(read_nms, all_aas):
    for idx, (obs_aa, ref_aa) in enumerate(zip(obs_seq, wt_aa)):
      pos = idx_to_relative_pos(idx)
      dd['Position'].append(pos)
      dd['Read name'].append(read_nm)
      if obs_aa != ref_aa:
        dd['Reference amino acid'].append(ref_aa)
        dd['Mutated amino acid'].append(obs_aa)
      else:
        dd['Reference amino acid'].append(ref_aa)
        dd['Mutated amino acid'].append('.')
    timer.update()

  df = pd.DataFrame(dd)
  df.to_csv(out_dir + f'{nm}_{split}.csv')

  df['Reference and position'] = df['Position'].astype(str) + df['Reference amino acid']
  pv_df = df.pivot(index = 'Read name', columns = 'Reference and position', values = 'Mutated amino acid')
  pv_df.to_csv(out_dir + f'{nm}_{split}_pivot.csv')
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
    for split in range(8):
      command = f'python {NAME}.py {nm} {split}'
      script_id = NAME.split('_')[0]

      # Write shell scripts
      sh_fn = qsubs_dir + f'q_{script_id}_{nm}_{split}.sh'
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
  [nm, split] = args

  convert(nm, split)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    gen_qsubs()
