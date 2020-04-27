# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util, compbio
import pandas as pd

import _fitness_from_reads_pt

# Default params
inp_dir = _config.DATA_DIR
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

params = {
}

##
# Functions
##
def form_gt(changers, majorities, len_gt, positions):
  wt_symbol = '_'

  changer_pos_to_nt = {int(s[1:]): s[0] for s in changers}
  majorities_pos_to_nt = {int(s[1:]): s[0] for s in majorities}

  gt = ''
  for pos in positions:
    if pos in changer_pos_to_nt:
      gt += changer_pos_to_nt[pos]
    elif pos in majorities_pos_to_nt:
      gt += majorities_pos_to_nt[pos]
    else:
      gt += wt_symbol
  return gt


def subgroup(group, diffs):
  split_threshold = 0.05

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


def get_smart_genotypes(om_df):
  '''
    Proposes genotypes by looking at mutations that change in the same direction with approximately the same magnitude between any pair of timepoints.
  '''
  change_threshold = 0.025
  majority_threshold = 0.5

  nt_pos = om_df['Nucleotide and position']
  gts = set()
  len_gt = len(om_df) // 2
  positions = sorted(list(set([int(s[1:]) for s in nt_pos])))

  time_cols = sorted([col for col in om_df if col != 'Nucleotide and position'])  
  for idx in range(len(time_cols) - 1):
    t0, t1 = time_cols[idx], time_cols[idx + 1]

    diff = om_df[t1] - om_df[t0]

    uppers = list(nt_pos[diff > change_threshold])
    downers = list(nt_pos[diff < -1 * change_threshold])
    majorities = list(om_df[om_df[t1] >= majority_threshold]['Nucleotide and position'])

    up_diffs = list(diff[diff > change_threshold])
    down_diffs = list(diff[diff < -1 * change_threshold])
    covarying_groups = subgroup(uppers, up_diffs) + subgroup(downers, down_diffs)
    # covarying_groups = [uppers] + [downers]

    for gt in covarying_groups:
      gts.add(form_gt(gt, majorities, len_gt, positions))

  print(gts)

  return list(gts)

##
# Main
##
@util.time_dec
def main():
  print(NAME)

  datasets = [
    '3c',
    '3d',
    's5a',
  ]

  for dataset in datasets:
    df = pd.read_csv(inp_dir + f'data_{dataset}.csv')

    ## Convert nucleotide and real position to fake position
    nt_pos = df['Nucleotide and position']
    positions = sorted(list(set([int(s[1:]) for s in nt_pos])))
    pos_to_new = {pos: idx for idx, pos in enumerate(positions)}
    df['Nucleotide and position'] = [f'{s[0]}{pos_to_new[int(s[1:])]}' for s in nt_pos]

    ## Propose genotypes and infer
    proposed_genotypes = get_smart_genotypes(df)
    # continue
    package = _fitness_from_reads_pt.infer_fitness(df, proposed_genotypes)

    d = {
      'fitness': package[0],
      'fq_mat': package[1],
      'pred_marginals': package[2],
      'Proposed genotypes': proposed_genotypes,
    }

    fitness_df = pd.DataFrame({'Genotype': list(proposed_genotypes), 'Inferred fitness': d['fitness']})

    d['fitness_df'] = fitness_df

    import pickle
    with open(out_dir + f'{dataset}.pkl', 'wb') as f:
      pickle.dump(d, f)

    fitness_df.to_csv(out_dir + f'inferred_fitnesses_{dataset}.csv')

  return


if __name__ == '__main__':
  main()

