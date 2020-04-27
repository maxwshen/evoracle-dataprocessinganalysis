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
inp_dir = _config.OUT_PLACE + f'pb_e_form_dataset/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

exp_design = pd.read_csv(_config.DATA_DIR + f'Badran2015_SraRunTable.csv')
pacbio_nms = exp_design[exp_design['Instrument'] == 'PacBio RS II']['Library Name']
pacbio_nms = sorted(pacbio_nms)

##
# Functions
##
def load_proposed_genotypes(nm = 'easy'):
  inp_fn = inp_dir + f'propose_genotypes_{nm}.txt'
  with open(inp_fn) as f:
    lines = f.readlines()
  gts = [s.strip() for s in lines]
  return gts


##
# Main
##
@util.time_dec
def main():
  print(NAME)

  df = pd.read_csv(inp_dir + f'obs_reads_pivot.csv')

  # genotypes_set = 'easy'
  genotypes_set = 'smart'
  # genotypes_set = 'all'
  proposed_genotypes = load_proposed_genotypes(nm = genotypes_set)

  _fitness_from_reads_pt.infer_fitness(df, proposed_genotypes)

  return


if __name__ == '__main__':
  main()

