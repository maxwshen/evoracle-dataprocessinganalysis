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
inp_dir = _config.DATA_DIR + '_illumina/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

exp_design = pd.read_csv(_config.DATA_DIR + f'Badran2015_SraRunTable.csv')

##
# Functions
##
def run_align_bowtie2(srx, nm):
  r1_fn = inp_dir + f'{srx}_1.fastq'
  r2_fn = inp_dir + f'{srx}_2.fastq'
  out_fn = out_dir + f'{nm}.sam'

  bowtie2 = '/ahg/regevdata/projects/CRISPR-libraries/tools/bowtie2-2.3.5.1-linux-x86_64/bowtie2'
  bt_index = '/ahg/regevdata/projects/CRISPR-libraries/prj2/evolution/badran/data/cry1ac_bowtie2_index/cry1ac_bowtie2_index'

  command = f'{bowtie2} -x {bt_index} -1 {r1_fn} -2 {r2_fn} -S {out_fn}'
  # command = f'{bowtie2} -x {bt_index} -1 {r1_fn} -2 {r2_fn} --local -S {out_fn}'
  subprocess.check_output(command, shell = True)

  return


def run_align_needleman_wunsch(srr, nm):
  inp_fn = inp_dir + f'{srr}.fastq'
  genome_fn = inp_dir + 'SP055-rpoZ-cMyc-Cry1Ac1-d123.fa'

  target = open(genome_fn).readlines()[1].strip()

  seq_align_tool = '/ahg/regevdata/projects/CRISPR-libraries/tools/seq-align/bin/needleman_wunsch'

  out_fn = out_dir + f'{nm}.fa'
  with open(out_fn, 'w') as f:
    pass

  alignment_buffer = []

  timer = util.Timer(total = util.line_count(inp_fn))
  with open(inp_fn) as f:
    for i, line in enumerate(f):
      if i % 4 == 0:
        header = line.strip()
        read_nm = header.split()[0].replace('@', '')
      if i % 4 == 1:
        read = line.strip()
      if i % 4 == 3:
        qs = [ord(s) - 33 for s in line.strip()]
        if np.mean(qs) >= 30:

          read = compbio.reverse_complement(read)

          command = f'{seq_align_tool} --match 1 --mismatch -1 --gapopen -5 --gapextend -1 --freestartgap --freeendgap {read} {target}'
          align = subprocess.check_output(command, shell = True).decode('utf-8')
          align = align[:-2]

          alignment_buffer.append(f'>{read_nm}\n{align}\n')

          if len(alignment_buffer) > 100:
            print(f'Dumping alignment buffer...')
            with open(out_fn, 'a') as f:
              for item in alignment_buffer:
                f.write(item)
            alignment_buffer = []

      timer.update()

  print(f'Dumping alignment buffer...')
  with open(out_fn, 'a') as f:
    for item in alignment_buffer:
      f.write(item)
  alignment_buffer = []

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

  ill_srxs = exp_design[exp_design['Instrument'] == 'Illumina MiSeq']['Experiment']
  ill_nms = exp_design[exp_design['Instrument'] == 'Illumina MiSeq']['Library Name']

  num_scripts = 0
  for srx, nm in zip(ill_srxs, ill_nms):
    command = f'python {NAME}.py {srx} {nm}'
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + f'q_{script_id}_{srx}.sh'
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
def main(args):
  print(NAME)
  
  # Function calls
  [srx, nm] = args

  run_align_bowtie2(srx, nm)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(sys.argv[1:])
  else:
    gen_qsubs()
