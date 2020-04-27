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
inp_dir = _config.OUT_PLACE + f'pb_c_convert/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

exp_design = pd.read_csv(_config.DATA_DIR + f'Badran2015_SraRunTable.csv')
pacbio_nms = exp_design[exp_design['Instrument'] == 'PacBio RS II']['Library Name']
pacbio_nms = sorted(pacbio_nms)

params = {
  # 21

  'major_positions': [
    -76,
    -73,
    # -69,
    # 15,
    # 61,
    68,
    198,
    286,
    304,
    332,
    344,
    347,
    361,
    363,
    384,
    404,
    417,
    461,
    463,
    515,
    582,
  ],
}

pos_to_ref = {
  -76: 'A',
  -73: 'M',
  # -69: 'G',
  # 15: 'C',
  # 61: 'V',
  68: 'F',
  198: 'R',
  286: 'G',
  304: 'T',
  332: 'E',
  344: 'A',
  347: 'Q',
  361: 'T',
  363: 'S',
  384: 'D',
  404: 'S',
  417: 'N',
  461: 'E',
  463: 'N',
  515: 'E',
  582: 'S',
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
# Functions
##
def get_short_genotypes(dfs):
  short_gts = []
  for read_nm in set(dfs['Read name']):
    df = dfs[dfs['Read name'] == read_nm]

    obs_pos_to_mut = {pos: mut for pos, mut in zip(df['Position'], df['Mutated amino acid'])}

    short_gt = ''.join([obs_pos_to_mut[pos] if pos in obs_pos_to_mut else '.' for pos in params['major_positions']])
    # short_gt = ''.join([obs_pos_to_mut[pos] if pos in obs_pos_to_mut else pos_to_ref[pos] for pos in params['major_positions']])

    short_gts.append(short_gt)

  # Filter genotypes with amino acid 'e' representing a deletion
  print(f'Found {len(short_gts)} genotypes')
  short_gts = [s for s in short_gts if 'e' not in s]
  print(f'Filtered out e, leaving {len(short_gts)} genotypes')
  return short_gts

def major_subset():

  get_time_from_nm = lambda nm: nm.split('_')[2]

  dd = defaultdict(list)
  for nm in pacbio_nms:
    print(nm)
    df = pd.read_csv(inp_dir + f'{nm}.csv', index_col = 0)
    dfs = df[df['Position'].isin(params['major_positions'])]

    short_gts = get_short_genotypes(dfs)
    time = get_time_from_nm(nm)

    dd['Abbrev genotype'] += short_gts
    dd['Timepoint'] += [time] * len(short_gts)

  df = pd.DataFrame(dd)

  # Add stats
  df['Read count'] = 1
  dfs = df.groupby(['Abbrev genotype', 'Timepoint']).agg(sum).reset_index()
  sums = dfs.groupby(['Timepoint'])['Read count'].sum()
  time_to_sum = {time: ct for time, ct in zip(sums.index, list(sums))}
  dfs['Total count'] = [time_to_sum[t] for t in dfs['Timepoint']]
  dfs['Frequency'] = dfs['Read count'] / dfs['Total count']
  dfs.to_csv(out_dir + f'badran_pacbio.csv')

  pv_df = dfs.pivot(index = 'Abbrev genotype', columns = 'Timepoint', values = 'Frequency')
  pv_df = pv_df.fillna(value = 0)
  pv_df = pv_df[ordered_time_strings]
  pv_df.to_csv(out_dir + f'badran_pacbio_pivot.csv')

  # Subset to > 1% fq and renormalize
  t = pv_df.apply(max, axis = 'columns')
  gt_to_max_fq = {gt: max_fq for gt, max_fq in zip(t.index, list(t))}
  keep_gts = [gt for gt, max_fq in zip(t.index, list(t)) if max_fq > 0.01]
  print(f'Filtered {len(pv_df)} to {len(keep_gts)} genotypes with >1% fq in any timepoint')

  # Normalize
  pv_df = pv_df.loc[keep_gts]
  pv_df /= pv_df.apply(sum)
  pv_df = pv_df.sort_values(by = '528hrs', ascending = False)
  pv_df.to_csv(out_dir + f'badran_pacbio_pivot_1pct.csv')

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
def main():
  print(NAME)
  
  # Function calls
  major_subset()

  return


if __name__ == '__main__':
  # if len(sys.argv) > 1:
  #   main(sys.argv[1:])
  # else:
  #   gen_qsubs()
  main()