#!/usr/bin/env python

import os
import subprocess as sp
import sys

# Assumes datasets are all in bz2 format
datasets = {
  "news20":"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2",
#  "covtype":"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2",
#  "webspam":"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2",
#  "rcv1":"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2",
#  "elipson":"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2"
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
  print("usage: {} [get|test]".format(sys.argv[0]))
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

def svmTest(filename):
  out_name = filename + ".out"
  out_file = open(out_name, 'w')
  cmd = ["../build/obamadb_main",
         " -train_file=", filename,
         " -algorithm=", "svm",
         " -num_epochs=", "50",
         " -num_trials=", "10",
         " -threads=", "2"
         " -verbose=", "true"
         ]
  rc = -1
  try:
    rc = sp.call("".join(cmd), stdout=out_file, shell=True)
  except Exception as e:
    print "{} test excepted: {}".format(filename, e)
  out_file.close()
  if rc != 0:
    print "command was: {}".format("".join(cmd))
  return rc

def test():
  for dataset in datasets.keys():
    fname = dataset + ".data"
    if os.path.isfile(fname):
      if svmTest(fname) != 0:
        print "{} test failed".format(dataset)


def main():
  if len(sys.argv) != 2:
    usage()

  cmd = sys.argv[1]
  if "get" == cmd:
    get()
  elif "test" == cmd:
    test()
  else:
    usage()


if __name__ == '__main__':
  main()
