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
inp_dir = _config.OUT_PLACE + 'ill_a_align/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

exp_design = pd.read_csv(_config.DATA_DIR + f'Badran2015_SraRunTable.csv')

ref_fn = _config.DATA_DIR + f'SP055-rpoZ-cMyc-Cry1Ac1-d123.fa'
ref = open(ref_fn).readlines()[1].strip()

##
# 
##
def add_mutations(mut_dd, n_d, n_d2, sam):
  calls = parse_cigar(sam['cigar'])
  read_start_idx = sam['1-based pos'] - 1
  read = sam['seq']

  query_consumed = list('MIS=X')
  ref_consumed = list('MDN=X')

  ref_idx = 0
  read_idx = 0
  for idx in range(len(calls)):
    op = calls[idx]
    curr_idx_in_ref = read_start_idx + ref_idx
    ref_nt = ref[curr_idx_in_ref]
    obs_nt = read[read_idx]

    # Record mismatches
    if bool(op == 'X') or bool(op == 'M' and ref_nt != obs_nt):
      mut_dd['Position (0 based)'].append(curr_idx_in_ref)
      mut_dd['Reference nt'].append(ref_nt)
      mut_dd['Mutated nt'].append(obs_nt)
      mut_dd['Read name'].append(sam['read_nm'])


    # Record number of reads aligning to each ref position
    n_d[curr_idx_in_ref] += 1

    # Advance indices
    if op in query_consumed:
      read_idx += 1
    if op in ref_consumed:
      ref_idx += 1

  # Record start and end of reads
  n_d2['Read name'].append(sam['read_nm'])
  n_d2['Read start idx'].append(read_start_idx)
  ref_len = sum([bool(s in ref_consumed) for s in calls])
  n_d2['Read end idx'].append(read_start_idx + ref_len)
  n_d2['Read length'].append(ref_len)
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
def call_mutations(nm):
  inp_fn = inp_dir + f'{nm}.sam'

  mut_dd = defaultdict(list)
  n_d = defaultdict(lambda: 0)
  n_d2 = defaultdict(list)
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
      }

      if sam['target'] != 'SP055-rpoZ-cMyc-Cry1Ac1-d123':
        continue

      if sam['cigar'] == '*':
        continue

      # Call mutation and Track total readcount per position
      add_mutations(mut_dd, n_d, n_d2, sam)


  mut_df = pd.DataFrame(mut_dd)
  mut_df.to_csv(out_dir + f'{nm}.csv')

  n_dd = defaultdict(list)
  for pos in range(len(ref)):
    n_dd['Position (0 based)'].append(pos)
    n_dd['Read count'].append(n_d[pos])

  n_df = pd.DataFrame(n_dd)
  n_df.to_csv(out_dir + f'{nm}_readcounts.csv')

  ndf2 = pd.DataFrame(n_d2)
  ndf2.to_csv(out_dir + f'{nm}_read_idxs.csv')

  '''
    Important note on ndf2:
    - Many paired reads appear to have sequenced the same molecule. Mutations observed on paired reads are combined; overlapping paired reads are also expected to be combined.

    This is done in ill_b2_merge_n_paired_reads.py
  '''

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
  [nm] = args

  call_mutations(nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    gen_qsubs()
