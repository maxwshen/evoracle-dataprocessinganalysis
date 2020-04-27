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
inp_dir = _config.OUT_PLACE + f'pb_a_blasr/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

exp_design = pd.read_csv(_config.DATA_DIR + f'Badran2015_SraRunTable.csv')

params = {
  'min nMatch': 2000,
  # 'SNP calling req': 'no neighboring indel',
  'SNP calling req': 'any',
  'primer_len': 30,
}

##
# Functions
##
def parse_header(header_lines):
  '''
        nMatch: 2105
     nMisMatch: 17
          nIns: 0
          nDel: 71
          %sim: 95.9872
         Score: -10068
             Query: SRR3458162.1
            Target: SP055-rpoZ-cMyc-Cry1Ac1-d123
             Model: a hybrid of global/local non-affine alignment
         Raw score: -10068
            Map QV: 254
      Query strand: 0
     Target strand: 1
       QueryRange: 31 -> 2153 of 2184
      TargetRange: 3301 -> 5494 of 8326
  '''
  d = dict()
  for line in header_lines:
    kw = line.split(':')[0].strip()
    val = line.split(':')[1].strip()
    d[kw] = val
  return d

def parse_single_alignment(align_segment, processed_queries):
  header_lines = align_segment[:15]
  header_stats = parse_header(header_lines)

  align_lines = align_segment[15:]

  if header_stats['Query'] in processed_queries:
    # ignore that blasr sometimes returns multiple alignments for same read 
    return None

  if int(header_stats['nMatch']) <= params['min nMatch']:
    # print('nMatch too low')
    return None

  '''
    Parse in groups of 4 lines:
      130  GT-CCACTAAAATTTCTAACA-CTACTATATTA-CTAATAAAGATGTAAA
           || |||||||||||||||||| ||||||||||| |||||*||||||||||
     3401  GTCCCACTAAAATTTCTAACACCTACTATATTACCTAATGAAGATGTAAA

  '''
  align_start_idx = 6
  len_align_lines = 4
  dd = defaultdict(list)
  for idx in range(0, len(align_lines), len_align_lines):
    [query, mline, target, _] = align_lines[idx : idx + len_align_lines]
    mline = mline[align_start_idx:]

    target_start = int(target[:align_start_idx])
    target = target[align_start_idx:]

    query_start = int(query[:align_start_idx])
    query = query[align_start_idx:]

    mut_symbols = ['*', ' ']
    mut_idxs = [i for i, e in enumerate(mline) if e in mut_symbols]
    for mut_idx in mut_idxs:

      if params['SNP calling req'] == 'no neighboring indel':
        if mline[mut_idx - 1] == ' ' or mline[mut_idx + 1] == ' ':
          continue

      if params['SNP calling req'] == 'any':
        pass

      num_ins = target[:mut_idx].count('-')
      star_pos = target_start + mut_idx - num_ins

      num_dels = query[:mut_idx].count('-')
      query_pos = query_start + mut_idx - num_dels
      if query_pos <= params['primer_len']:
        continue

      ref_nt = target[mut_idx]
      obs_nt = query[mut_idx]

      if ref_nt == '-':
        # Ignore insertions
        continue

      dd['Position'].append(star_pos)
      dd['Reference nucleotide'].append(ref_nt)
      dd['Mutated nucleotide'].append(obs_nt)
      dd['Read name'].append(header_stats['Query'])
      dd['Target strand'].append(header_stats['Target strand'])

  return dd


##
# Aggregator
##
def parse_blasr_output(nm):
  inp_fn = inp_dir + f'{nm}.txt'

  with open(inp_fn) as f:
    lines = f.readlines()    

  # Split into individual alignments
  align_idxs = [idx for idx, line in enumerate(lines) if 'nMatch:' in line]
  n_alignments = len(align_idxs)
  print(f'Found {n_alignments} alignments')

  processed_queries = set()
  align_idxs.append(len(lines))
  dd = defaultdict(list)
  timer = util.Timer(total = len(align_idxs) - 1)
  for idx in range(len(align_idxs) - 1):
    start_idx = align_idxs[idx]
    end_idx = align_idxs[idx + 1]

    align_segment = lines[start_idx : end_idx]

    ans_dd = parse_single_alignment(align_segment, processed_queries)
    if ans_dd is not None:
      if len(ans_dd['Read name']) > 0:
        processed_queries.add(ans_dd['Read name'][0])
      for col in ans_dd:
        dd[col] += ans_dd[col]

    timer.update()

  df = pd.DataFrame(dd)
  df.to_csv(out_dir + f'{nm}.csv')

  # Invalid because of inconsistent target strand from blasr
  # df['Reference and position'] = df['Position'].astype(str) + df['Reference nucleotide']
  # pv_df = df.pivot(index = 'Read name', columns = 'Reference and position', values = 'Mutated nucleotide')
  # pv_df.to_csv(out_dir + f'{nm}_pivot.csv')
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

  parse_blasr_output(nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    gen_qsubs()
