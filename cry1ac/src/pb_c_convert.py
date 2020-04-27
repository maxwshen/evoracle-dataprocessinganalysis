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
inp_dir = _config.OUT_PLACE + f'pb_b_parse/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

exp_design = pd.read_csv(_config.DATA_DIR + f'Badran2015_SraRunTable.csv')
wt_gt = open(_config.DATA_DIR + f'SP055-rpoZ-cMyc-Cry1Ac1-d123.fa').readlines()[1].strip()
rc_wt_gt = compbio.reverse_complement(wt_gt)

params = {
}

##
# Functions
##
def create_gt_with_mutations(dfs):
  plus_strand = bool(dfs['Target strand'].iloc[0] == 0)
  gt = list(wt_gt) if plus_strand else list(rc_wt_gt)

  for idx, row in dfs.iterrows():
    try:
      assert gt[row['Position']] == row['Reference nucleotide'], 'Error: Probably bad strand'
    except:
      import code; code.interact(local=dict(globals(), **locals()))
    gt[row['Position']] = row['Mutated nucleotide']
  gt = ''.join(gt)
  return gt if plus_strand else compbio.reverse_complement(gt)

def translate_cry1ac(gt):
  from Bio.Seq import Seq
  start_pos = 2910
  end_pos = 8327
  gt = gt[start_pos : end_pos]
  return compbio.translate(gt)

##
# Primary
##
def convert(nm):
  df = pd.read_csv(inp_dir + f'{nm}.csv', index_col = 0)

  read_nms = list(sorted(set(df['Read name'])))

  all_aas = []

  print(f'Translating reads with mutations to amino acid sequences...')
  timer = util.Timer(total = len(read_nms))
  for read_nm in read_nms:
    dfs = df[df['Read name'] == read_nm]

    gt = create_gt_with_mutations(dfs)
    aas = translate_cry1ac(gt)
    all_aas.append(aas)
    timer.update()

  wt_aa = translate_cry1ac(wt_gt)

  print(f'Summarizing non-synonymous amino acid mutations...')
  dd = defaultdict(list)
  idx_to_relative_pos = lambda idx: idx - 76
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
  df.to_csv(out_dir + f'{nm}.csv')

  df['Reference and position'] = df['Position'].astype(str) + df['Reference amino acid']
  pv_df = df.pivot(index = 'Read name', columns = 'Reference and position', values = 'Mutated amino acid')
  pv_df.to_csv(out_dir + f'{nm}_pivot.csv')
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

  pacbio_nms = exp_design[exp_design['Instrument'] == 'PacBio RS II']['Library Name']

  num_scripts = 0
  for nm in pacbio_nms:
    command = f'python {NAME}.py {nm}'
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + f'q_{script_id}_{nm}.sh'
    with open(sh_fn, 'w') as f:
      f.write(f'#!/bin/bash\n{command}\n')
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append(f'qsub -V -P regevlab -l h_rt=4:00:00,h_vmem=4G -wd {_config.SRC_DIR} {sh_fn} &')

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

  convert(nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    gen_qsubs()
