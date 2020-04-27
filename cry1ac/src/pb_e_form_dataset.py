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
inp_dir = _config.OUT_PLACE + f'pb_d_major_subset/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

exp_design = pd.read_csv(_config.DATA_DIR + f'Badran2015_SraRunTable.csv')
pacbio_nms = exp_design[exp_design['Instrument'] == 'PacBio RS II']['Library Name']
pacbio_nms = sorted(pacbio_nms)

params = {
  'mutations': 'VIWSGDNGERIPYCDKSKL',
  'wt':        'AACFRGTEAQTSDSNENES',
  'wt_dots':   '...................',
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
def is_canonical_mutant(gt):
  for idx, c in enumerate(gt):
    if c != '.' and gt[idx] != params['mutations'][idx]:
      return False
  return True


def get_true_genotype_matrix(df):
  '''
    Ground truth genotype matrix: badran_pacbio_pivot_1pct.csv
  '''

  # Subset mutations to only 2^19 = 500k possibilities
  gts = set(df['Abbrev genotype'])
  print(f'Starting from {len(gts)} genotypes...')
  canonical_gts = [gt for gt in gts if is_canonical_mutant(gt)]
  df = df[df['Abbrev genotype'].isin(canonical_gts)]
  print(f'Subsetted to {len(canonical_gts)} canonical genotypes.')
  dfs = df

  # Pivot
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

  # Melt
  pv_df = pv_df.reset_index()
  df = pv_df.melt(id_vars = 'Abbrev genotype', value_name = 'Frequency')
  df.to_csv(out_dir + f'badran_pacbio_melt_1pct.csv')
  return df


def get_obs_marginals(df):
  '''
    Output: obs_reads.csv, obs_reads_pivot.csv

    Single nucleotides for now (most convenient)
  '''

  len_gt = len(params['mutations'])
  for pos in range(len_gt):
    df[f'Pos{pos}'] = [s[pos] for s in df['Abbrev genotype']]

  pos_cols = [col for col in df.columns if 'Pos' in col]

  mdf = pd.DataFrame()
  for pos in range(len_gt):
    wt_nt = params['wt_dots'][pos]
    mut_nt = params['mutations'][pos]
    dfs = df.groupby(['Timepoint', f'Pos{pos}'])['Frequency'].apply(sum).reset_index().pivot(index = f'Pos{pos}', columns = 'Timepoint')  
    dfs.columns = dfs.columns.droplevel(0)
    dfs.index = [f'{wt_nt}{pos}', f'{mut_nt}{pos}']

    dfs = dfs[ordered_time_strings]
    dfs.columns = [ordered_time_strings.index(col) if col in ordered_time_strings else col for col in dfs.columns]
    dfs['Nucleotide and position'] = dfs.index

    mdf = mdf.append(dfs, ignore_index = True)

  mdf.to_csv(out_dir + f'obs_reads_pivot.csv', index = False)

  return mdf


##
# Propose genotypes
##
def recursive_make_genotypes(all_nts, idx):
  # all_nts is a list of lists
  if idx == 0:
    return all_nts[idx]
  else:
    gts = recursive_make_genotypes(all_nts, idx - 1)
    new_gts = []
    for gt in gts:
      for nt in all_nts[idx]:
        new_gts.append(gt + nt)
    return new_gts

def get_all_genotypes():
  all_mutations = params['mutations']
  all_wt = params['wt_dots']
  all_nts = [[s, wt_nt] for s, wt_nt in zip(all_mutations, all_wt)]
  all_gts = recursive_make_genotypes(all_nts, len(all_nts) - 1)
  return all_gts

def form_gt(changers, majorities):
  len_gt = len(params['mutations'])
  wt_symbol = '.'

  changer_pos_to_nt = {int(s[1:]): s[0] for s in changers}
  majorities_pos_to_nt = {int(s[1:]): s[0] for s in majorities}

  gt = ''
  for pos in range(len_gt):
    if pos in changer_pos_to_nt:
      gt += changer_pos_to_nt[pos]
    elif pos in majorities_pos_to_nt:
      gt += majorities_pos_to_nt[pos]
    else:
      gt += wt_symbol
  return gt

def get_smart_genotypes(om_df):
  change_threshold = 0.025
  majority_threshold = 0.5

  nt_pos = om_df['Nucleotide and position']
  gts = set()

  time_cols = sorted([col for col in om_df if col != 'Nucleotide and position'])  
  for idx in range(len(time_cols) - 1):
    t0, t1 = time_cols[idx], time_cols[idx + 1]

    diff = om_df[t1] - om_df[t0]

    uppers = nt_pos[diff > change_threshold]
    downers = nt_pos[diff < -1 * change_threshold]
    majorities = om_df[om_df[t1] >= majority_threshold]['Nucleotide and position']

    for gt in [uppers, downers]:
      gts.add(form_gt(gt, majorities))

  return gts

def propose_genotypes(df, obs_marginals):
  '''
    Proposes genotypes.
    -> proposed_genotypes.txt
  '''

  gt_sets = {
    'easy': sorted(list(set(df['Abbrev genotype']))),
    'smart': get_smart_genotypes(obs_marginals),
    'all': get_all_genotypes(),
  }

  print(f'Overlap b/w smart and easy')
  calc_set_overlap(gt_sets['easy'], gt_sets['smart'])

  for gt_nm in gt_sets:
    gts = gt_sets[gt_nm]
    print(f'Writing {gt_nm} genotypes -- {len(gts)}')
    with open(out_dir + f'propose_genotypes_{gt_nm}.txt', 'w') as f:
      for gt in gts:
        f.write(f'{gt}\n')

  return

def calc_set_overlap(a, b):
  a, b = set(a), set(b)
  print(len(a), len(b), len(a & b))
  print('In both sets')
  for gt in a:
    if gt in b:
      print(f'  {gt}')
  print('Only in first set')
  for gt in a:
    if gt not in b:
      print(f'  {gt}')
  print('Only in second set')
  for gt in b:
    if gt not in a:
      print(f'  {gt}')
  return

##
# Main
##
@util.time_dec
def main():
  print(NAME)

  df = pd.read_csv(inp_dir + f'badran_pacbio.csv', index_col = 0)
  df = get_true_genotype_matrix(df)

  obs_marginals = get_obs_marginals(df)

  propose_genotypes(df, obs_marginals)


  return


if __name__ == '__main__':
  main()

