#! /usr/bin/env python

# script to test process scalability of HogWild!
#

import subprocess as sp
import time

EXE = "/fastdisk/obamadb/build/obamadb_main"
DATA = "/fastdisk/obamadb/data/"

for dataset in ['s0']:
  for num_proc in xrange(1,11):
    for num_threads in xrange(1,11):
      if num_proc * num_threads != 10:
        continue
      procs = []
      print time.time(), " Starting ", num_proc, " parallel runs with ", num_threads, " threads each for dataset ", dataset
      for proc_idx in xrange(0,num_proc):
        pin_start = proc_idx * num_threads
        pin_end = (proc_idx + 1) * num_threads
        pin = ','.join(map(str,xrange(pin_start, pin_end)))
        fname = dataset + "_proc{}_t{}_{}.out".format(num_proc, num_threads, proc_idx)
        exe_str = \
          "{} -num_trials 1 -verbose true -num_epochs 20 -rank {} -threads {} -core_affinities {} -train_file {} -test_file {} -algorithm mc >> {}".format(
          EXE,
          6,  # rank will effect the size of the model
          num_threads,
          pin,
          DATA + "_synth_mc_" + dataset + ".train.tsv",
          DATA + "_synth_mc_" + dataset + ".probe.tsv",
          fname)
        with open(fname, 'w') as fout:
          fout.write(exe_str + "\n")
        procs.append(sp.Popen(exe_str, shell=True))

      for proc in procs:
        proc.wait()
        if proc.returncode != 0:
            print "failed during run of cmd:\n{}\nexiting".format(exe_str)
            exit(-1)

      print time.time(), "Done with ", num_proc, " parallel runs with ", num_threads, " threads each for dataset ", dataset
