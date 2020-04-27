# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, pickle
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util, compbio
import pandas as pd

import _fitness_from_reads_pt_multi

# Default params
inp_dir = _config.OUT_PLACE + f'pb_f2_infer_multi/'
inp_dir_e = _config.OUT_PLACE + f'pb_e_form_dataset/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

exp_design = pd.read_csv(_config.DATA_DIR + f'Badran2015_SraRunTable.csv')
pacbio_nms = exp_design[exp_design['Instrument'] == 'PacBio RS II']['Library Name']
pacbio_nms = sorted(pacbio_nms)

params = {
  'read_lens': [
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
  'random_seeds': list(range(10)),
}


def get_stats():
  '''
    Calculate pearsonr all together (collapse timepoints and genotypes)
  '''

  true_df = pd.read_csv(inp_dir_e + 'badran_pacbio_pivot_1pct.csv')
  true_df = true_df.rename(columns = {col: int(col.replace('hrs', '')) for col in true_df.columns if 'hrs' in col})

  hrs_cols = [col for col in true_df.columns if col != 'Abbrev genotype']
  idx_to_hr = {str(idx): col for idx, col in enumerate(hrs_cols)}

  stats_dd = defaultdict(list)
  timer = util.Timer(total = len(params['read_lens']))
  for read_len in params['read_lens']:
    for random_seed in params['random_seeds']:
      pred_df = pd.read_csv(inp_dir + f'genotype_matrix_readlen_{read_len}_{random_seed}.csv', index_col = 0)
      pred_df['Abbrev genotype'] = pred_df.index
      pred_df = pred_df.rename(columns = idx_to_hr)

      # Compare predictions to observed
      pdf = pred_df.melt(id_vars = 'Abbrev genotype', var_name = 'Time', value_name = 'Frequency')
      tdf = true_df.melt(id_vars = 'Abbrev genotype', var_name = 'Time', value_name = 'Frequency')

      mdf = tdf.merge(pdf, on = ['Abbrev genotype', 'Time'], how = 'outer', suffixes = ['_true', '_pred'])
      mdf = mdf.fillna(value = 0)

      from scipy.stats import pearsonr
      stats_dd['Read length'].append(read_len)
      stats_dd['Random seed'].append(random_seed)
      stats_dd['r'].append(pearsonr(mdf[f'Frequency_true'], mdf[f'Frequency_pred'])[0])

    timer.update()

  stats_df = pd.DataFrame(stats_dd)
  stats_df.to_csv(out_dir + f'stats.csv')


  return



##
# Main
##
@util.time_dec
def main():
  print(NAME)

  get_stats()
  return


if __name__ == '__main__':
  main()
