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
inp_dir = _config.OUT_PLACE + f'data_multi/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)


##
# Main
##
@util.time_dec
def main(argv):
  print(NAME)

  modelexp_nm = argv[0]
  print(modelexp_nm)

  exp_design = pd.read_csv(_config.DATA_DIR + f'{modelexp_nm}.csv')
  hyperparam_cols = [col for col in exp_design.columns if col != 'Name']

  new_out_dir = out_dir + f'{modelexp_nm}/'
  util.ensure_dir_exists(new_out_dir)

  print(f'Collating experiments...')

  model_out_dir = _config.OUT_PLACE + f'_fitness_from_reads_pt_multi/{modelexp_nm}/'
  num_fails = 0
  timer = util.Timer(total = len(exp_design))
  for idx, row in exp_design.iterrows():
    int_nm = row['Name']
    real_nm = row['dataset']

    try:
      command = f'cp {model_out_dir}/model_{int_nm}/_final_fitness.csv {new_out_dir}/fitness_{int_nm}.csv'
      subprocess.check_output(command, shell = True)

      command = f'cp {model_out_dir}/model_{int_nm}/_final_genotype_matrix.csv {new_out_dir}/genotype_matrix_{int_nm}.csv'
      subprocess.check_output(command, shell = True)
    except:
      num_fails += 1

    timer.update()

  print(f'Collated {len(exp_design)} experiments with {num_fails} failures')

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    print(f'Usage: python x.py <modelexp_nm>')
