#!/usr/bin/env python
import re
import os
from os import chdir, listdir

base_dir = '/Users/nav/Programming/obamadb_results/results/epoch_results_after_refactor/vary_model_size_and_density'
result_filename_re = re.compile('output_(\d)_m(\d+\.0)_d(0\.\d+)_(\d+).txt')
result_line_re = re.compile('/users/navsan/Programming/obamadb/data/_synth_svm.*')
schema_line_re = re.compile('train_file,.*')

print 'run_num, density, train_file, rows, features, sparsity, nnz, num_threads, trial, epoch, train_time'
filenames = listdir(base_dir)
for fn in filenames:
  s = result_filename_re.search(fn)
  if s:
    run_num = int(s.groups()[0])
    num_features = float(s.groups()[1])
    density = float(s.groups()[2])
    threads = int(s.groups()[3])
    with open(base_dir + '/' + fn) as f:
      for line in f:
        if result_line_re.match(line):
          l = line.split('/')[-1]
          output_line = [str(run_num), str(density), l]
          print ', '.join(output_line),


