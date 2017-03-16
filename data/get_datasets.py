#!/usr/bin/env python

import os
import subprocess as sp
import sys

# Assumes datasets are all in bz2 format
datasets = {
  "news20":"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2",
  "covtype":"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2",
  "webspam":"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2",
  "rcv1":"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2",
  "epsilon":"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2"
}

def file_in_list(file, file_list):
  for f in file_list:
    if file in f.lower():
      return True
  return False

def download(f_name, url):
  out_name = url.split('/')[-1]
  out_file = open(out_name, 'w')
  cmd = ["curl", url]
  rc = -1
  try:
    rc = sp.call(cmd, stdout=out_file)
  except Exception as ex:
    print "download call excepted: {}".format(ex)
  out_file.close()
  return (rc, out_name)

def unzip(f_name):
  out_name = f_name.split(".")[0] + ".data"
  out_file = open(out_name, 'w')
  cmd = ["bzip2", "--stdout", "-d", f_name]
  rc = sp.call(cmd, stdout=out_file)
  out_file.close()
  return rc

def usage():
  print("usage: {} [get|test|parse]".format(sys.argv[0]))
  print("\tget: downloads datasets")
  print("\ttest: looks for downloaded datasets and runs a speedup test")
  print("\tparse: parses the output files from the speedup test and summarizes them")
  exit(0)

def get():
  working_files = [f for f in os.listdir('.') if os.path.isfile(f)]

  for ds_name, url in datasets.iteritems():
    if not file_in_list(ds_name, working_files):
      print "Downloading {}".format(ds_name)
      rc, zipped_file_name = download(ds_name, url)
      if rc != 0:
        print "download of {} failed, skipping dataset".format(ds_name)
        continue
      rc = unzip(zipped_file_name)
      if rc != 0:
        print "failed to unzip {}, skipping dataset".format(ds_name)
    else:
      print "Found a file containing the name {}, not attempting to download data set".format(ds_name)

DEFAULT_ARGS = {
  "algorithm":"svm",
  "num_epochs":"50",
  "num_trials":"10",
  "threads":"4",
  "train_file":"",
  "verbose":"true"
}

VALID_ARGS = ["algorithm","core_affinities","measure_convergence","num_epochs",
              "num_trials","rank","test_file","threads","train_file","verbose"]

def arg_str(**kwargs):
  # Gets an argument string which can be passed to obamadb
  # kwargs have key of the corresponding argument
  #
  args = ""
  for arg in kwargs.keys():
    if arg not in VALID_ARGS:
      print("argument {} not recognized!".format(arg))
      exit(-1)
    else:
      args += " -{}={}".format(arg, kwargs[arg])
  for arg in DEFAULT_ARGS.keys():
    if arg not in kwargs:
      args += " -{}={}".format(arg, DEFAULT_ARGS[arg])
  return args


def svmTest(filename, **kwargs):
  out_name = filename + "." + "-".join([str(k) + "=" + str(v) for k, v in kwargs.items()]) + ".out"
  out_file = open(out_name, 'w')
  kwargs['train_file'] = filename
  cmd = "../build/obamadb_main " + arg_str(**kwargs)
  rc = -1
  try:
    rc = sp.call(cmd, stdout=out_file, shell=True)
  except Exception as e:
    print "{} test excepted: {}".format(filename, e)
  out_file.close()
  if rc != 0:
    print "command was: {}".format("".join(cmd))
  return rc

def find_data_file(data_set):
  working_files = [f for f in os.listdir('.') if os.path.isfile(f)]
  for f in working_files:
    if data_set in f.split('.')[0] and f.endswith('.data'):
      return f
  return ""

def test():
  # Looks through all of the datasets in the datasets map and
  # tests them with obamadb and the given params
  #
  for dataset in datasets.keys():
    fname = find_data_file(dataset)
    if fname != "":
      for thread in [1,2,4,8,10]:
        if svmTest(fname, threads=str(thread)) != 0:
          print "{} test failed".format(dataset)
    else:
      print("did not find data file for {}".format(dataset))

def parse_file(f):
  # we expect the files to look like dataname.data.properties.out
  #
  trial_summary = {}
  fsplit = f.split('.')
  trial_summary['dataset'] = fsplit[0]
  props = fsplit[2].split('-')
  for prop in props:
    psplit = prop.split('=')
    trial_summary[psplit[0]] = psplit[1]

  with open(f, 'r') as fh:
    lines = fh.read().split('\n')
    keys = lines[-3].split(',')
    values = lines[-2].split(',')
    for i in range(len(keys)):
      trial_summary[keys[i]] = values[i]

  return trial_summary

def parse():
  # Looks for all the .out files. Grabs their summary line, aggregates these and
  # prints the results
  #
  working_files = [f for f in os.listdir('.') if os.path.isfile(f)]
  summaries = []
  for f in working_files:
    if f.endswith("out"):
      try:
        summaries.append(parse_file(f))
      except Exception as e:
        print "failed to read file {} : {}".format(f,e)
  keys = ['dataset', 'threads','mean', 'variance', 'stddev']
  print ",".join(keys)
  for s in summaries:
    sstr = ""
    for k in keys:
      sstr += "," if k not in s else s[k] + ","
    print sstr[:-1]

def main():
  if len(sys.argv) != 2:
    usage()

  cmd = sys.argv[1]
  if "get" == cmd:
    get()
  elif "test" == cmd:
    test()
  elif "parse" == cmd:
    parse()
  else:
    usage()


if __name__ == '__main__':
  main()
