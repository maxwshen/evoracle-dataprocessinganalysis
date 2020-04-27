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
inp_dir = _config.OUT_PLACE + 'a2_bowtie2_paired/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

design_df = pd.read_csv(_config.DATA_DIR + 'exp_design.csv')
ref = 'tada_context'

##
# Functions
##
def get_alignment(mut_dd, sam, nd, ref_seq):
  calls = parse_cigar(sam['cigar'])
  read_start_idx = sam['1-based pos'] - 1
  read = sam['seq']
  qs = sam['qs']

  query_consumed = list('MIS=X')
  ref_consumed = list('MDN=X')

  obs_ref_idxs = set()

  ref_idx = 0
  read_idx = 0
  for idx in range(len(calls)):
    op = calls[idx]
    curr_idx_in_ref = read_start_idx + ref_idx
    ref_nt = ref_seq[curr_idx_in_ref]
    obs_nt = read[read_idx]
    obs_q = ord(qs[read_idx]) - 33

    # Record mismatches
    if bool(op == 'X') or bool(op == 'M' and ref_nt != obs_nt):
      if obs_q >= 30:
        mut_dd['Position (0 based)'].append(curr_idx_in_ref)
        mut_dd['Reference nt'].append(ref_nt)
        mut_dd['Mutated nt'].append(obs_nt)
        mut_dd['Read name'].append(sam['read_nm'])

    # Advance indices
    if op in query_consumed:
      read_idx += 1
    if op in ref_consumed:
      ref_idx += 1
      obs_ref_idxs.add(curr_idx_in_ref)

  import code; code.interact(local=dict(globals(), **locals()))
  for ori in obs_ref_idxs:
    nd[ori] += 1

  return


def parse_cigar(cigar):
  '''
    148M1X1M -> ['M', 'M', ..., 'X', 'M']
  '''
  ops = list('MIDNSHP=X')
  calls = []
  trailing_idx = 0
  for idx in range(len(cigar)):
    if cigar[idx] in ops:
      op = cigar[idx]
      length = int(cigar[trailing_idx : idx])      
      calls += [op] * length
      trailing_idx = idx + 1
  return calls

## 
# Primary
##
def convert_sam_to_text(ref, sample_id):
  inp_fn = inp_dir + f'{ref}/{sample_id}.sam'

  ref_fn = _config.DATA_DIR + f'{ref}.fa'
  ref_seq = open(ref_fn).readlines()[-1].strip()

  # Parse SAM
  mut_dd = defaultdict(list)
  nd = {idx: 0 for idx in range(len(ref_seq))}
  timer = util.Timer(total = util.line_count(inp_fn))
  with open(inp_fn) as f:
    for i, line in enumerate(f):
      timer.update()

      if line[0] == '@':
        continue

      w = line.split()
      sam = {
        'read_nm': w[0],
        'target': w[2],
        '1-based pos': int(w[3]),
        'cigar': w[5],
        'seq': w[9],
        'qs': w[10],
      }

      if sam['cigar'] == '*':
        continue

      get_alignment(mut_dd, sam, nd, ref_seq)

  mut_df = pd.DataFrame(mut_dd)
  ref_out_dir = out_dir + f'{ref}/'
  util.ensure_dir_exists(ref_out_dir)
  mut_df.to_csv(ref_out_dir + f'{sample_id}.csv')

  ndd = {
    'Position (0 based)': list(nd.keys()),
    'Read count': list(nd.values()),
  }
  ndf = pd.DataFrame(ndd)
  ndf.to_csv(ref_out_dir + f'n_{sample_id}.csv')

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
    sample_id = row['Short name']

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
  [sample_id] = argv
  convert_sam_to_text(ref, sample_id)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    gen_qsubs()
