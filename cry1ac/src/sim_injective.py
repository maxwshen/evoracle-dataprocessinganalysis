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

params = {
  't': 15,
}


##
# Functions
##
def sim_trajectories(gts_to_fitness, spikeins, gt_to_start, num_timepoints, noise = 0):
  '''
      Simulate trajectories.
      Output:
      - Full length genotype matrix over time
      - Marginal (single mutations) matrix over time
      -- Requires all genotypes to be the same length
  '''
  fitnesses = np.array(list(gts_to_fitness.values()))
  all_fqs = []
  gts = list(gts_to_fitness.keys())
  curr_fqs = np.array(list(gt_to_start.values()))
  for t in range(num_timepoints):
    all_fqs.append(curr_fqs)
    mean_fitness = np.dot(fitnesses, curr_fqs)
    changes = fitnesses / mean_fitness
    curr_fqs = changes * curr_fqs
    
    # drift (for plot legibility)
    if noise > 0:
      drift_fqs = np.random.normal(curr_fqs, scale = noise)
      drift_fqs = [d_fq if c_fq > 0 else 0 for d_fq, c_fq in zip(drift_fqs, curr_fqs) ]
      drift_fqs = np.minimum(drift_fqs, 1)
      drift_fqs = np.maximum(drift_fqs, 0)
      
      curr_fqs = drift_fqs / sum(drift_fqs)

    # spike ins at fixed negligible frequency
    for si_gt in spikeins:
      sit = spikeins[si_gt]
      if sit == t:
        si_fq = 1e-4
        gt_idx = gts.index(si_gt)
        
        reduce_factor = 1 - si_fq
        curr_fqs = curr_fqs * reduce_factor
        curr_fqs[gt_idx] = si_fq
      
  gt_df = pd.DataFrame(np.array(all_fqs).T, index = list(gts_to_fitness.keys()))
  # gt_df *= 100
  
  gts = list(gts_to_fitness.keys())
  gt_len = len(gts[0])
  
  import copy
  om_df = copy.copy(gt_df)
  alphabet = dict()
  for gt_idx in range(gt_len):
    nts = [s[gt_idx] for s in gts]
    om_df[f'nt {gt_idx}'] = nts
    alphabet[gt_idx] = set(nts)
  
  from collections import defaultdict
  om_dd = defaultdict(list)
  for t in range(num_timepoints):
    for gt_idx in range(gt_len):
      for nt in alphabet[gt_idx]:
        om_dd['Time'].append(t)
        # om_dd['Position'].append(gt_idx)
        # om_dd['Nucleotide'].append(nt)
        om_dd['Nucleotide and position'].append(f'{nt} {gt_idx}')
        
        fq = sum(om_df[om_df[f'nt {gt_idx}'] == nt][t])
        om_dd['Frequency'].append(fq)
  om_df = pd.DataFrame(om_dd)
  
  # Reshape gt_df
  gt_df['Genotype'] = gt_df.index
  gt_df = gt_df.melt(id_vars = 'Genotype', var_name = 'Time', value_name = 'Frequency')
    
  return {
    'gt_df': gt_df,
    'om_df': om_df,
  }


def gen_exps():
  '''
  '''
  gts = [
    '--', 'a-', '-b', 'ab',
  ]

  min_fitness = 1
  max_fitness = 10
  fitness_resolution = 0.2

  num_timepoints = 15

  '''
    Only a few starting fractions yield the same t0 marginal
  '''
  starting_fracs_setup1 = {
    '--': 0.9,
    'a-': 0.05,
    '-b': 0.05,
    'ab': 0,
  }
  spike_template1 = {
    'ab': 1,
  }

  starting_fracs_setup2 = {
    '--': 0.9,
    'a-': 0,
    '-b': 0,
    'ab': 0.05,
  }
  spike_template2 = {
    'a-': 1,
    '-b': 1,
  }
  starting_fracs = [starting_fracs_setup1, starting_fracs_setup2]
  spike_templates = [spike_template1, spike_template2]

  '''
    Generate
  '''
  exp_num = 0
  total_exps = 1000

  all_marginals = pd.DataFrame()
  all_inputs = pd.DataFrame()

  for starting_frac, spike_template in zip(starting_fracs, spike_templates):

    timer = util.Timer(total = total_exps)
    for iter in range(total_exps):
      fitness = {gt: (np.random.uniform(low = min_fitness, high = max_fitness) // fitness_resolution) * fitness_resolution for gt in gts}
      fitness['--'] = 1
      spikes = {gt: np.random.randint(low = 1, high = num_timepoints) for gt in spike_template}

      package = sim_trajectories(
        fitness,
        spikes,
        starting_frac,
        num_timepoints,
      )

      '''
        Save input

        fitness is an ordered dict
      '''
      full_spikes = {gt: spikes[gt] if gt in spikes else -1 for gt in gts}
      input_vec = list(fitness.values()) + list(starting_frac.values()) + list(full_spikes.values())

      # Could do: annotate feature names
      all_inputs[f'exp {exp_num}'] = input_vec

      '''
        Save output

        om_df is always in the same order given num. timepoints, num positions, and alphabet
      '''
      om_df = package['om_df']
      # Could do: annotate feature names
      # if len(all_marginals) == 0:
      #   all_marginals['Time'] = om_df['Time']
      #   all_marginals['Nucleotide and position'] = om_df['Nucleotide and position']
      all_marginals[f'exp {exp_num}'] = om_df['Frequency']

      exp_num += 1
      timer.update()

  all_marginals.to_csv(out_dir + f'all_marginals.csv')
  all_inputs.to_csv(out_dir + f'all_inputs.csv')

  om_df.to_csv(out_dir + f'example_marginal.csv')

  return


def simulate_data(exp_nm):
  '''
    Simulate many input/output tuples from a model (fitness, starting vec, instantiation) -> marginals.

  '''

  global out_dir
  out_dir = out_dir + f'{exp_nm}/'
  util.ensure_dir_exists(out_dir)

  gen_exps()

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

  exp_design = pd.read_csv(_config.DATA_DIR + f'Badran2015_SraRunTable.csv')
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
def main(argv):
  print(NAME)
  
  exp_nm = argv[0]

  # Function calls
  simulate_data(exp_nm = exp_nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    print(f'Usage: python x.py <exp_nm>')
    # gen_qsubs()
  # main()