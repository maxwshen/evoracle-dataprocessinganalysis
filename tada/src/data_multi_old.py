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
inp_dir = _config.OUT_PLACE + f'f3_merge_regimes/'
inp_dir_d = _config.OUT_PLACE + f'd_singlemuts/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

design_df = pd.read_csv(_config.DATA_DIR + 'exp_design.csv')

params = {
  'sim_read_lens': [
    1,
    50,
    75,
    100,
    150,
    200,
    250,
    300,
  ],
  'threshold': 8,
  'min_gt_frequency': 0,

  'major_positions': None,
  'mutations': None,
}


##
# Functions
##
# def is_canonical_mutant(gt):
#   for idx, c in enumerate(gt):
#     if c != '.' and gt[idx] != params['mutations'][idx]:
#       return False
#   return True

def load_major_mutations():
  threshold = params['threshold']
  # Load from file output from Jupyter notebook
  # Format: {pos}{mut_aa}
  major_muts_df = pd.read_csv(inp_dir_d + f'_above{threshold}.csv')
  major_muts = list(major_muts_df['Mutation'])
  major_pos = [int(s[:-1]) for s in major_muts]

  # Sort both lists by position
  major_pos, major_muts = zip(*sorted(zip(major_pos, major_muts)))
  params['mutations'] = major_muts
  params['major_positions'] = major_pos
  return



def get_true_genotype_matrix(df):
  '''
    Ground truth genotype matrix: badran_pacbio_pivot_1pct.csv
  '''
  threshold = params['threshold']

  # Subset mutations to only 2^19 = 500k possibilities
  # gts = set(df['Full genotype'])
  # print(f'Starting from {len(gts)} genotypes...')
  # canonical_gts = [gt for gt in gts if is_canonical_mutant(gt)]
  # df = df[df['Full genotype'].isin(canonical_gts)]
  # print(f'Subsetted to {len(canonical_gts)} canonical genotypes.')
  dfs = df

  # Sanitize time column
  num_digits = max([len(str(s)) for s in set(dfs['Sample'])])
  dfs['Sample name sanitized'] = [f't{str(s).zfill(num_digits)}' for s in dfs['Sample']]

  # Pivot
  pv_df = dfs.pivot(index = 'Full genotype', columns = 'Sample name sanitized', values = 'Frequency')
  pv_df = pv_df.fillna(value = 0)
  pv_df.to_csv(out_dir + f'pv_groundtruth_{threshold}.csv')

  # Subset to > 1% fq and renormalize
  t = pv_df.apply(max, axis = 'columns')
  gt_to_max_fq = {gt: max_fq for gt, max_fq in zip(t.index, list(t))}
  min_gt_fq = params['min_gt_frequency']
  keep_gts = [gt for gt, max_fq in zip(t.index, list(t)) if max_fq > min_gt_fq]
  print(f'Filtered {len(pv_df)} to {len(keep_gts)} genotypes with >{min_gt_fq}% fq in any timepoint')

  # Normalize
  pv_df = pv_df.loc[keep_gts]
  pv_df /= pv_df.apply(sum)
  pv_df.to_csv(out_dir + f'pv_groundtruth_{min_gt_fq}pct_{threshold}.csv')

  # Melt
  pv_df = pv_df.reset_index()
  df = pv_df.melt(id_vars = 'Full genotype', value_name = 'Frequency')
  df.to_csv(out_dir + f'mel_groundtruth_{min_gt_fq}pct_{threshold}.csv')
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
  print(f'\tGetting read groups ...')
  groups = get_read_groups(adj_read_len)

  # Create columns for each position with the observed aa
  print(f'\tExpanding full genotype into positions ...')
  full_gts = list(df['Full genotype'])
  pos_from_df = [s.split()[0] for s in df['Full genotype'].iloc[0].split(',')]
  for idx, pos in enumerate(pos_from_df):
    df[f'pos {pos}'] = [s.split(',')[idx].split()[-1] for s in full_gts]

  print(f'\tGetting positions per read group ...')
  for gidx, group in enumerate(groups):
    # Get positions for group
    group_muts = params['mutations'][group[0] : group[-1] + 1]
    group_poss = sorted(list(set([int(s[:-1]) for s in group_muts])))
    pos_first_idx = params['major_positions'].index(group_poss[0])
    pos_last_idx = params['major_positions'].index(group_poss[-1])

    group_pos_cols = [f'pos {s}' for s in group_poss]

    df[f'Group {gidx}'] = df[group_pos_cols].apply(''.join, axis = 'columns')

  df = df.rename(columns = {'Sample name sanitized': 'Timepoint'})

  print(f'\tGetting mutation frequencies per read group ...')
  mdf = pd.DataFrame()
  for gidx, group in enumerate(groups):
    dfs = df.groupby(['Timepoint', f'Group {gidx}'])['Frequency'].apply(sum).reset_index().pivot(index = f'Group {gidx}', columns = 'Timepoint')  
    dfs.columns = dfs.columns.droplevel(0)

    dfs.index = [f'{s} {gidx}' for s in list(dfs.index)]

    ## 
    def resample(p):
      '''
        Add binomial noise with std = noise

        if p is 0 or 1, add half-gaussian noise with modified std = noise.
        Half-gaussian = |normal(mu, sigma)| has std = 0.6 * sigma.
        Therefore, we rescale sigma by 5/3 to ensure sigma = intended stdev.
      '''
      std = noise
      if std == 0: return p
      
      if p <= 0.01:
        draw = abs(np.random.normal(0, std * (5/3)))
        new_p = 0 + draw
      elif p >= 0.99:
        draw = abs(np.random.normal(0, std * (5/3)))
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

    # Optional: add synthetic noise
    # Find N such that binomial std = specified noise
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
  len_gt = len(params['major_positions'])
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
      # Group can contain multiple mutations for a single position; add wt_symbol only for num. unique positions in group
      group = groups[gidx]
      group_muts = params['mutations'][group[0] : group[-1] + 1]
      group_poss = sorted(list(set([int(s[:-1]) for s in group_muts])))
      gt += wt_symbol * len(group_poss)
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
    # 'easy': sorted(list(set(df['Full genotype']))),
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

  params['threshold'] = 10
  threshold = params['threshold']
  print(f'Using threshold {threshold}')
  load_major_mutations()

  df = pd.read_csv(inp_dir + f'mel_trajectory_t{threshold}.csv', index_col = 0)
  df = get_true_genotype_matrix(df)

  ## 
  # Group 1
  ##
  group_nm = f'simple_{threshold}'
  proposal_type = 'smart'
  exp_dd = defaultdict(list)
  for read_len in params['sim_read_lens']:
    dataset_nm = f'{group_nm}_{proposal_type}_rl_{read_len}_t{threshold}'
    print(dataset_nm)

    print(f'Getting obs marginals ...')
    obs_marginals, groups = get_obs_marginals(
      df, read_len, dataset_nm,
      noise = 0,
    )
    print(f'Proposing genotypes ...')
    num_proposed_gts = propose_genotypes(
      df, obs_marginals, groups, dataset_nm,
      style = proposal_type,
    )

    print(f'Saving ...')
    import pickle
    with open(out_dir + f'read_groups_{dataset_nm}.pkl', 'wb') as f:
      pickle.dump(groups, f)

    exp_dd['data_readlen'].append(read_len)
    exp_dd['data_num_proposed_gts'].append(num_proposed_gts)
    exp_dd['data_proposal_type'].append(proposal_type)
    exp_dd['data_num_groups'].append(len(groups))
    exp_dd['dataset'].append(dataset_nm)
  exp_df = pd.DataFrame(exp_dd)
  exp_df.to_csv(_config.DATA_DIR + f'datagroup_{group_nm}.csv')

  ## 
  # Group 2
  ##
  group_nm = 'varynoise'
  proposal_type = 'smart'
  exp_dd = defaultdict(list)
  read_lens = [1, 150, 300]
  for read_len in read_lens:
    for noise in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0]:
      dataset_nm = f'{group_nm}_{proposal_type}_rl_{read_len}_noise_{noise}'
      print(dataset_nm)

      obs_marginals, groups = get_obs_marginals(
        df, read_len, dataset_nm,
        noise = noise,
      )
      num_proposed_gts = propose_genotypes(
        df, obs_marginals, groups, dataset_nm,
        style = proposal_type,
      )

      import pickle
      with open(out_dir + f'read_groups_{dataset_nm}.pkl', 'wb') as f:
        pickle.dump(groups, f)

      exp_dd['data_readlen'].append(read_len)
      exp_dd['data_num_proposed_gts'].append(num_proposed_gts)
      exp_dd['data_proposal_type'].append(proposal_type)
      exp_dd['data_num_groups'].append(len(groups))
      exp_dd['data_noise'].append(noise)
      exp_dd['dataset'].append(dataset_nm)
  exp_df = pd.DataFrame(exp_dd)
  exp_df.to_csv(_config.DATA_DIR + f'datagroup_{group_nm}.csv')

  ## 
  # Group 3
  ##
  group_nm = 'varyproposals'
  proposal_types = [
    'smart',
    'x2',
    'x3',
    'x4',
    'x5',
    'x6',
    'x7',
    'x8',
    'x9',
    'x10',
    'x20',
    'x30',
    'x40',
    'x50',
    'x100',
  ]
  exp_dd = defaultdict(list)
  read_lens = [1, 150, 300, 600]
  for read_len in read_lens:
    for proposal_type in proposal_types:
      dataset_nm = f'{group_nm}_{proposal_type}_rl_{read_len}'
      print(dataset_nm)

      obs_marginals, groups = get_obs_marginals(
        df, read_len, dataset_nm,
        noise = 0,
      )
      num_proposed_gts = propose_genotypes(
        df, obs_marginals, groups, dataset_nm,
        style = proposal_type,
      )

      import pickle
      with open(out_dir + f'read_groups_{dataset_nm}.pkl', 'wb') as f:
        pickle.dump(groups, f)

      exp_dd['data_readlen'].append(read_len)
      exp_dd['data_num_proposed_gts'].append(num_proposed_gts)
      exp_dd['data_num_groups'].append(len(groups))
      exp_dd['data_proposal_type'].append(proposal_type)
      exp_dd['dataset'].append(dataset_nm)
  exp_df = pd.DataFrame(exp_dd)
  exp_df.to_csv(_config.DATA_DIR + f'datagroup_{group_nm}.csv')

  return


if __name__ == '__main__':
  
  main()

