#!/usr/bin/env python
import re

logdir = '/Users/nav/Programming/obamadb_results/ObamaDb-osx/'
logname = 'after_inlining_vtune_expt.out'

result_line_re = re.compile('amplxe: Collection started.*(r\d*ge).*')
output_schema_line_re = re.compile('train_file, rows,.*')
output_line_re = re.compile('/users/navsan/Programming/obamadb/data/_synth_svm.*')

def find_vtune_result_name(f):
  while True:
    line = f.readline()
    if not line:
      return None
    s = result_line_re.search(line)
    if s:
      result = s.groups()[0]
      return result

def scan_till_output_line(f, print_schema_line):
  while True:
    line = f.readline()
    if not line:
      break
    if output_schema_line_re.match(line):
      if print_schema_line:
        print 'result, ' + line,
      return

def print_output_lines(f, result):
  while True:
    line = f.readline()
    if not line:
      break
    if not output_line_re.match(line):
      return
    print result + ',',
    print line.split('/')[-1],

def main():
  output_schema_line_printed = False
  with open(logdir + '/' + logname) as f:
    while True:
      result = find_vtune_result_name(f)
      if not result:
        break
      scan_till_output_line(f, not output_schema_line_printed)
      output_schema_line_printed = True
      print_output_lines(f, result)

main()



