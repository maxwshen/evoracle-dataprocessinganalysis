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
inp_dir_c = _config.OUT_PLACE + f'ill_c2_combine/'
inp_dir_b2 = _config.OUT_PLACE + f'ill_b22_combine/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

exp_design = pd.read_csv(_config.DATA_DIR + f'Badran2015_SraRunTable.csv')
wt_gt = open(_config.DATA_DIR + f'SP055-rpoZ-cMyc-Cry1Ac1-d123.fa').readlines()[1].strip()
rc_wt_gt = compbio.reverse_complement(wt_gt)

params = {
  'num_splits': 10,
}

pos_groups = [
  [-76, -73],
  [15],
  [68],
  [198],
  [286, 304],
  [332, 344, 347, 361, 363],
  [384, 404, 417],
  [461, 463],
  [515],
  [582],
]

pos_to_mut = {
  -76: 'V',
  -73: 'I',
  15: 'W',
  68: 'S',
  198: 'G',
  286: 'D',
  304: 'N',
  332: 'G',
  344: 'E',
  347: 'R',
  361: 'I',
  363: 'P',
  384: 'Y',
  404: 'C',
  417: 'D',
  461: 'K',
  463: 'S',
  515: 'K',
  582: 'L',
}
major_positions = set(pos_to_mut.keys())

##
# Helper
##
def get_reads(ndf, start_pos, end_pos):
  '''
    ndf indices are nucleotide indices, while pos is amino acid index.
  '''
  conv_start_pos = 2910 + (start_pos + 76) * 3
  conv_end_pos = 2910 + (end_pos + 76) * 3

  crit = (ndf['Read start idx'] <= conv_start_pos) & \
         (ndf['Read end idx'] >= conv_end_pos)
  reads = ndf[crit]['Read name']
  return reads

##
# Primary
##
def obtain_full_muts(nm):
  mut_df = pd.read_csv(inp_dir_c + f'{nm}.csv', index_col = 0)
  ndf = pd.read_csv(inp_dir_b2 + f'{nm}.csv', index_col = 0)

  # print('Forming pos to ref dict ...')
  # pos_to_ref = {row['Position']: row['Reference amino acid'] for idx, row in mut_df.iterrows()}

  mut_df = mut_df[mut_df['Position'].isin(major_positions)]

  print('Pivoting ...')
  pvm_df = mut_df.pivot(index = 'Read name', columns = 'Position', values = 'Mutated amino acid')

  dd = defaultdict(list)
  timer = util.Timer(total = len(pos_groups))
  for pos_idx, pos_group in enumerate(pos_groups):
    start_pos, end_pos = min(pos_group), max(pos_group)
    reads = get_reads(ndf, start_pos, end_pos)
    n = len(reads)

    dfs = pvm_df.loc[reads]

    # Form "nucleotides" by order of important positions
    dfs = dfs[pos_group].fillna(value = '.')

    # Filter out other reads
    for pos in pos_group:
      mut = pos_to_mut[pos]
      wt = '.'
      crit = (dfs[pos].isin([mut, wt]))
      dfs = dfs[crit]

    dfs['Abbrev genotype'] = dfs.apply(lambda row: ''.join(row.values.astype(str)), axis = 'columns')

    from collections import Counter
    counts = Counter(dfs['Abbrev genotype'])
    total_n = sum(counts.values())

    # For each mutation, record statistics
    for mut in counts:
      count = counts[mut]
      freq = count / total_n

      dd['Group'].append(pos_idx)
      dd['Mutation'].append(mut)
      dd['Frequency'].append(freq)
      dd['Total count'].append(total_n)
      dd['Allele count'].append(count)

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
  [nm] = args
  obtain_full_muts(nm)

  # ill_nms = exp_design[exp_design['Instrument'] == 'Illumina MiSeq']['Library Name']

  # for nm in ill_nms:
  #   print(nm)
  #   obtain_full_muts(nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    gen_qsubs()
  # main([])
