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
NAME = util.get_fn(__file__)

target_df = pd.read_csv(_config.DATA_DIR + f'targets.csv', index_col = 0)


##
# Functions
##
def make_bt_index():
  bt_fold = _config.DATA_DIR + f'bowtie2_index/'
  util.ensure_dir_exists(bt_fold)

  for idx, row in target_df.iterrows():
    nm = row['Name']
    # seq = row['Sequence context']
    assembly = row['Assembly']
    chrom = row['Chromosome']
    strand = row['Strand']
    start = row['Start']
    end = row['End']

    twobit = '/ahg/regevdata/projects/CRISPR-libraries/tools/2bit/twoBitToFa'
    twobit_ref = f'/ahg/regevdata/projects/CRISPR-libraries/tools/2bit/{assembly}.2bit'

    # Radius = 1000 needs to be longer than any single read for bowtie2 to work without local alignment
    command = f'{twobit} -seq={chrom} -start={start - 1001} -end={end + 1000} {twobit_ref} temp.fa; cat temp.fa'
    seq = subprocess.check_output(command, shell = True).decode('utf-8')
    seq = ''.join(seq.split()[1:])
    seq = seq.upper()

    if strand == '-':
      seq = compbio.reverse_complement(seq)

    try:
      assert seq.index(row['Spacer (20 nt)']) == 1000
    except:
      print(seq.index(row['Spacer (20 nt)']))
      import code; code.interact(local=dict(globals(), **locals()))

    print(len(seq))
    print(nm)

    ref_fn = _config.DATA_DIR + f'{nm}.fa'
    with open(ref_fn, 'w') as f:
      f.write(f'>{nm}\n{seq}\n')

    bt2_build = f'/ahg/regevdata/projects/CRISPR-libraries/tools/bowtie2-2.3.5.1-linux-x86_64/bowtie2-build'
    command = f'{bt2_build} {ref_fn} {bt_fold}/{nm}'
    result = subprocess.check_output(command, shell = True)

  return


##
# Main
##
@util.time_dec
def main(argv):
  print(NAME)
  
  # Function calls
  make_bt_index()

  return


if __name__ == '__main__':
  # if len(sys.argv) > 1:
  #   main(sys.argv[1:])
  # else:
  #   gen_qsubs()
  main([])