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
  # 'mutations': 'VIWASDNGERIPYCDKSKL',
  # 'wt':        'AACVFGTEAQTSDSNENES',
  'mutations': 'VIWSGDNGERIPYCDKSKL',
  'wt':        'AACFRGTEAQTSDSNENES',
  'wt_dots':   '...................',
  'sim_read_lens': [
    1,
    50,
    75,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
  ],
}

import pb_d_major_subset
params['major_positions'] = pb_d_major_subset.params['major_positions']

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


def get_interpolated_true_genotype_matrix(df):
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

  # Interpolate
  pv_df = interpolate(pv_df)

  # Save
  pv_df = pv_df.sort_values(by = '528hrs', ascending = False)
  pv_df.to_csv(out_dir + f'badran_pacbio_pivot_1pct.csv')

  # Melt
  pv_df = pv_df.reset_index()
  df = pv_df.melt(id_vars = 'Abbrev genotype', value_name = 'Frequency')
  df.to_csv(out_dir + f'badran_pacbio_melt_1pct.csv')

  return df


def get_read_groups(read_len):
  '''
    groups = list of lists
  '''
  positions = params['major_positions']
  start_pos = positions[0]

  groups = []
  for pos in range(positions[0], positions[-1] + read_len + 1, read_len):
    group = [idx for idx in range(len(positions)) if bool(pos <= positions[idx] < pos + read_len)]
    if len(group) != 0:
      groups.append(group)

  print(groups)
  return groups


def get_obs_marginals(df, read_len, dataset_nm, noise = 0):
  '''
    Output: obs_reads.csv, obs_reads_pivot.csv

    Single nucleotides for now (most convenient)
  '''

  # Adjust since mutations are amino acids
  adj_read_len = max(1, read_len // 3)
  groups = get_read_groups(adj_read_len)

  for gidx, group in enumerate(groups):
    df[f'Group {gidx}'] = [s[group[0] : group[-1] + 1] for s in df['Abbrev genotype']]

  mdf = pd.DataFrame()
  for gidx, group in enumerate(groups):
    dfs = df.groupby(['Timepoint', f'Group {gidx}'])['Frequency'].apply(sum).reset_index().pivot(index = f'Group {gidx}', columns = 'Timepoint')  
    dfs.columns = dfs.columns.droplevel(0)
    dfs.index = [f'{s} {gidx}' for s in list(dfs.index)]

    # import code; code.interact(local=dict(globals(), **locals()))
    # dfs = dfs[ordered_time_strings]
    dfs.columns = [ordered_time_strings.index(col) if col in ordered_time_strings else col for col in dfs.columns]
    sorted_cols = sorted(list(dfs.columns))
    dfs = dfs[sorted_cols]

    # Optional: add synthetic noise
    # Find N such that binomial std = specified noise

    def resample(p):
      '''
        Add binomial noise with std = noise

        if p is 0 or 1, add half-gaussian noise with modified std = noise.
        Half-gaussian = |normal(mu, sigma)| has std = 0.6 * sigma.
        Therefore, we rescale sigma by 5/3 to ensure sigma = intended stdev.
      '''
      std = noise
      if std == 0: return p
      
      scaling_factor = (1 - 2/np.pi)**(-1/2)
      
      if p <= 0.01:
        draw = abs(np.random.normal(0, std * scaling_factor))
        new_p = 0 + draw
      elif p >= 0.99:
        draw = abs(np.random.normal(0, std * scaling_factor))
        new_p = 1 - draw
      else:
        q = 1 - p

        '''
          Find N
          std = sqrt(p*q) / sqrt(n)
          n = (sqrt(p*q)/std)^2
        '''
        n = (np.sqrt(p * q) / std)**2
        
        '''
          If n is too low, values are too discretized. Use normal noise instead
        '''
        if n > 20:
          new_p = np.random.binomial(n, p) / n
        else:
          new_p = np.random.normal(p, std)
          while new_p < 0 or new_p > 1:
            new_p = np.random.normal(p, std)

      return new_p

    new_dfs = dfs.applymap(resample)

    # Renormalize to sum to 1
    new_dfs /= new_dfs.apply(sum, axis = 'rows')

    while sum(np.isnan(new_dfs.iloc[0])) > 0:
      new_dfs = dfs.applymap(resample)
      new_dfs /= new_dfs.apply(sum, axis = 'rows')

    dfs = new_dfs

    dfs['Nucleotide and position'] = dfs.index

    mdf = mdf.append(dfs, ignore_index = True)

  mdf.to_csv(out_dir + f'obs_reads_pivot_{dataset_nm}.csv', index = False)

  return mdf, groups


##
# Interpolation
##
def interpolate(pv_df):
  '''
    columns: hrs
  '''
  tres = 6

  # Convert string columns to int
  tcols = [col for col in pv_df.columns if 'hrs' in col]
  tcols_ints = [int(col.replace('hrs', '')) for col in pv_df.columns if 'hrs' in col]
  pv_df = pv_df.rename(columns = {tcol: int(tcol.replace('hrs', '')) for tcol in tcols})

  for idx in range(len(tcols_ints) - 1):
    tnow, tnext = tcols_ints[idx], tcols_ints[idx + 1]

    interpolate_between(pv_df, tnow, tnext, tres)

  # Rename back to hrs
  tcols = [col for col in pv_df.columns if col != 'Abbrev genotype']
  stcols = sorted(tcols)
  pv_df = pv_df[stcols]
  pv_df = pv_df.rename(columns = {tcol: f'{tcol}hrs' for tcol in tcols})

  global ordered_time_strings
  ordered_time_strings = [f'{tcol}hrs' for tcol in stcols]

  return pv_df


def interpolate_between(pv_df, tnow, tnext, tres):
  '''
    Edit pv_df in place
  '''
  epsilon = 1e-6

  num_t_steps = (tnext - tnow) // tres

  dfs = pv_df[[tnow, tnext]]

  # Fill in zeros that occur exactly once, normalize
  crit = (dfs[tnow] == 0) & (dfs[tnext] != 0)
  dfs.loc[crit, tnow] = epsilon
  crit = (dfs[tnow] != 0) & (dfs[tnext] == 0)
  dfs.loc[crit, tnext] = epsilon
  dfs /= dfs.apply(sum)

  # Calculate fitness
  fitness = dfs[tnext] / dfs[tnow]
  fitness = fitness.fillna(value = 0)

  # Adjust fitness for finer time resolution
  fitness = fitness**(1/num_t_steps)

  # Add interpolated data
  curr_ps = dfs[tnow]
  for step in range(1, num_t_steps):
    new_t = tnow + tres * step
    mean_fitness = np.dot(curr_ps, fitness)
    new_ps = curr_ps * fitness / mean_fitness
    pv_df[new_t] = new_ps
    curr_ps = new_ps

  return


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


def form_gt(changers, majorities, groups):
  len_gt = len(params['mutations'])
  wt_symbol = '.'

  changer_pos_to_nt = {int(s.split()[1]): s.split()[0] for s in changers}
  majorities_pos_to_nt = {int(s.split()[1]): s.split()[0] for s in majorities}

  gt = ''
  for gidx in range(len(groups)):
    if gidx in changer_pos_to_nt:
      gt += changer_pos_to_nt[gidx]
    elif gidx in majorities_pos_to_nt:
      gt += majorities_pos_to_nt[gidx]
    else:
      gt += wt_symbol * len(groups[gidx])
  return gt


def smart_subgroup(group, diffs, split_threshold):
  subgroups = []
  used = set()
  for idx in range(len(group)):
    mut = group[idx]
    if mut in used:
      continue

    curr_group = [mut]
    used.add(mut)

    for jdx in range(len(group)):
      j_mut = group[jdx]
      if j_mut not in used:
        if diffs[idx] - split_threshold <= diffs[jdx] <= diffs[idx] + split_threshold:
          curr_group.append(j_mut)
          used.add(j_mut)
    subgroups.append(curr_group)

  return subgroups


def get_smart_genotypes(om_df, groups):
  '''
  '''
  change_threshold = 0.025
  majority_threshold = 0.5
  split_threshold = 0.05

  nt_pos = om_df['Nucleotide and position']
  gts = set()

  time_cols = sorted([col for col in om_df if col != 'Nucleotide and position'])  
  for idx in range(len(time_cols) - 1):
    t0, t1 = time_cols[idx], time_cols[idx + 1]

    diff = om_df[t1] - om_df[t0]

    uppers = list(nt_pos[diff > change_threshold])
    downers = list(nt_pos[diff < -1 * change_threshold])
    majorities = list(om_df[om_df[t1] >= majority_threshold]['Nucleotide and position'])

    up_diffs = list(diff[diff > change_threshold])
    down_diffs = list(diff[diff < -1 * change_threshold])
    covarying_groups = smart_subgroup(uppers, up_diffs, split_threshold) + smart_subgroup(downers, down_diffs, split_threshold)

    for gt in covarying_groups:
      gts.add(form_gt(gt, majorities, groups))

  return gts


def propose_combinatorial(df):
  '''
    Identify top single mutations with highest average frequency across timepoints.
    Build all combinations of mutations.
  '''
  soft_limit_total = 10000
  soft_limit_n = 14
  # 2^14 = 16k. Assumes binary mutations

  # use single nucleotide marginals
  om_df, groups = get_obs_marginals(df, 1, 'temp', noise = 0)
  mut_rows = [s for s in list(om_df['Nucleotide and position']) if '.' not in s]
  om_df = om_df.set_index('Nucleotide and position')
  om_df = om_df.loc[mut_rows]

  # prioritize single mutations by average frequency across timepoints
  mean_fqs = list(om_df.apply(np.mean, axis = 'columns'))
  mut_to_mean_fq = {mut_row: mean_fq for mut_row, mean_fq in zip(mut_rows, mean_fqs)}

  top_muts = sorted(mut_to_mean_fq, key = mut_to_mean_fq.get, reverse = True)

  if len(top_muts) < soft_limit_n:
    pass
  else:
    top_muts = top_muts[:soft_limit_n]

  top_poss = [int(s.split()[1]) for s in top_muts]

  # Combinatorially form all genotypes
  all_nts = []
  all_mutations = params['mutations']
  all_wt = params['wt_dots']
  for pos in range(len(all_mutations)):
    if pos in top_poss:
      all_nts.append([all_mutations[pos], all_wt[pos]])
    else:
      all_nts.append([all_wt[pos]])
  all_gts = recursive_make_genotypes(all_nts, len(all_nts) - 1)

  return all_gts


def propose_genotypes(df, obs_marginals, groups, dataset_nm, style = 'smart'):
  '''
    Proposes genotypes.
    -> proposed_genotypes.txt
  '''

  gt_sets = {
    # 'easy': sorted(list(set(df['Abbrev genotype']))),
    'smart': get_smart_genotypes(obs_marginals, groups),
    # 'all': get_all_genotypes(),
  }

  # print(f'Overlap b/w smart and easy')
  # calc_set_overlap(gt_sets['easy'], gt_sets['smart'])
  set_smart = set(gt_sets['smart'])

  gts = gt_sets['smart']
  if style == 'smart':
    pass
  else:
    size_factor = len(set_smart) * int(style.replace('x', ''))
    add_factor = size_factor - 1
    add_gts = propose_combinatorial(df)

    add_gts = [s for s in add_gts if s not in set_smart]

    if len(add_gts) < add_factor:
      pass
    else:
      np.random.shuffle(add_gts)
      add_gts = add_gts[:add_factor]

    for add_gt in add_gts:
      gts.add(add_gt)

  print(f'Writing genotypes -- {len(gts)}')
  with open(out_dir + f'propose_genotypes_{dataset_nm}.txt', 'w') as f:
    for gt in gts:
      f.write(f'{gt}\n')
  return len(gts)


def calc_set_overlap(a, b):
  a, b = set(a), set(b)
  print(f'First: {len(a)}, Second: {len(b)}, Overlap: {len(a & b)}')
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
  df = get_interpolated_true_genotype_matrix(df)

  timepoints = ordered_time_strings
  num_ts = len(timepoints)

  ##
  # Group 1a
  ##
  group_nm = 'intsimple'
  proposal_type = 'smart'
  exp_dd = defaultdict(list)
  for read_len in [1, 100]:
    noises = [0, 0.03, 0.05] if read_len == 1 else [0]
    for noise in noises:
      for final_t_idx in range(3, num_ts):
        dfs = df[df['Timepoint'].isin(timepoints[:final_t_idx])]

        dataset_nm = f'{group_nm}_{proposal_type}_rl_{read_len}_noise_{noise}_star_{final_t_idx}'
        print(dataset_nm)

        obs_marginals, groups = get_obs_marginals(
          dfs, read_len, dataset_nm,
          noise = 0,
        )
        num_proposed_gts = propose_genotypes(
          dfs, obs_marginals, groups, dataset_nm,
          style = proposal_type,
        )

        import pickle
        with open(out_dir + f'read_groups_{dataset_nm}.pkl', 'wb') as f:
          pickle.dump(groups, f)

        exp_dd['data_readlen'].append(read_len)
        exp_dd['data_noise'].append(noise)
        exp_dd['data_num_proposed_gts'].append(num_proposed_gts)
        exp_dd['data_proposal_type'].append(proposal_type)
        exp_dd['data_num_groups'].append(len(groups))
        exp_dd['dataset'].append(dataset_nm)
        exp_dd['data_risingstar'].append(final_t_idx)

  exp_df = pd.DataFrame(exp_dd)
  exp_df.to_csv(_config.DATA_DIR + f'datagroup_{group_nm}.csv')

  return


if __name__ == '__main__':
  main()

