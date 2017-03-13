#!/usr/bin/env python
from importlib import import_module
from os import chdir, listdir
from os.path import isfile
from subprocess import call
import pandas as pd

def get_report_names(result_dir, reports_dir):
  report_base = reports_dir + '/' + result_dir
  return (report_base + '.csv', report_base + '-summary.txt')

def generate_reports_for_result_dir(result_dir, reports_dir):
  vtune = '/Applications/Intel\ VTune\ Amplifier\ XE\ 2017.app/Contents/MacOS/amplxe-cl'
  csv_name, txt_name = get_report_names(result_dir, reports_dir)
  d = {'vtune':vtune, 'result':result_dir, 'csv_name': csv_name, 'txt_name': txt_name}
  hw_events_cmd = '{vtune} -report hw-events -r {result} -format csv -csv-delimiter comma -report-output {csv_name}'.format(**d)
  print hw_events_cmd
  call(hw_events_cmd, shell=True)
  summary_cmd = '{vtune} -report summary -r {result} -report-output {txt_name}'.format(**d)
  print summary_cmd
  call(summary_cmd, shell=True)

def generate_reports_for_result_dir_if_not_exists(result_dir, reports_dir):
  csv_name, txt_name = get_report_names(result_dir, reports_dir)
  if not isfile(csv_name) or not isfile(txt_name):
    generate_reports_for_result_dir(result_dir, reports_dir)

def extract_svm_cmd_params(report):
  txt = report + '-summary.txt'
  with open(txt, 'r') as f:
    for line in f:
      if line.startswith('Application Command Line'):
        parts = line.split(' ')                      #Split into words
        parts = [p.replace('"', '') for p in parts]  #Remove quotes
        assert len(parts) > 10
        for i,_ in enumerate(parts):
          if parts[i] == '-threads':
            threads = int(parts[i+1])
          elif parts[i] == '-train_file':
            data = parts[i+1]
            assert data.endswith('.train.tsv')
            data = data.replace('.train.tsv','')
            data = data.replace('/users/navsan/Programming/obamadb/data/','')
            if '_synth_svm' in data:
              data = data.replace('_synth_svm_','')
        return (report, data, threads)

def extend_with_calc_columns(df):
  df.loc[:,'CPI'] = df['CPU_CLK_UNHALTED.THREAD']/df['INST_RETIRED.ANY']
  df.loc[:,'NumLoads'] = \
      df['MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_HIT_PS']\
    + df['MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_HITM_PS']\
    + df['MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_MISS_PS']\
    + df['MEM_LOAD_UOPS_L3_MISS_RETIRED.LOCAL_DRAM_PS']\
    + df['MEM_LOAD_UOPS_RETIRED.HIT_LFB_PS']\
    + df['MEM_LOAD_UOPS_RETIRED.L1_HIT_PS']\
    + df['MEM_LOAD_UOPS_RETIRED.L2_HIT_PS']\
    + df['MEM_LOAD_UOPS_RETIRED.L3_HIT_PS']
  df.loc[:,'L1 Hit Ratio'] = df['MEM_LOAD_UOPS_RETIRED.L1_HIT_PS']/df['NumLoads']
  df.loc[:,'LFB Hit Ratio'] = df['MEM_LOAD_UOPS_RETIRED.HIT_LFB_PS']/df['NumLoads']
  df.loc[:,'L1 + LFB Hit Ratio'] = df['L1 Hit Ratio'] + df['LFB Hit Ratio']
  df.loc[:,'L2 Hit Ratio'] = df['MEM_LOAD_UOPS_RETIRED.L2_HIT_PS']/df['NumLoads']
  df.loc[:,'L3 Hit Ratio'] = df['MEM_LOAD_UOPS_RETIRED.L3_HIT_PS']/df['NumLoads']
  df.loc[:,'Other Cache (clean) Hit Ratio'] = df['MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_HIT_PS']/df['NumLoads']
  df.loc[:,'Other Cache (dirty) Hit Ratio'] = df['MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_HITM_PS']/df['NumLoads']
  df.loc[:,'DRAM Hit Ratio'] = df['MEM_LOAD_UOPS_L3_MISS_RETIRED.LOCAL_DRAM_PS']/df['NumLoads']

  df.loc[:,'L1 Hit Cost'] = 4 * df['MEM_LOAD_UOPS_RETIRED.L1_HIT_PS']
  #df.loc[:,'LFB Hit Cost'] = df['MEM_LOAD_UOPS_RETIRED.HIT_LFB_PS']
  df.loc[:,'L2 Hit Cost'] = 12 * df['MEM_LOAD_UOPS_RETIRED.L2_HIT_PS']
  df.loc[:,'L3 Hit Cost'] = 26 * df['MEM_LOAD_UOPS_RETIRED.L3_HIT_PS']
  df.loc[:,'Other Cache (clean) Hit Cost'] = 43 * df['MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_HIT_PS']
  df.loc[:,'Other Cache (dirty) Hit Cost'] = 60 * df['MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_HITM_PS']
  df.loc[:,'DRAM Hit Cost'] = 200 * df['MEM_LOAD_UOPS_L3_MISS_RETIRED.LOCAL_DRAM_PS']

  df.loc[:,'Total Load Cost'] = \
      df['L1 Hit Cost']\
    + df['L2 Hit Cost']\
    + df['L3 Hit Cost']\
    + df['Other Cache (clean) Hit Cost']\
    + df['Other Cache (dirty) Hit Cost']\
    + df['DRAM Hit Cost']

  df.loc[:,'Relative L1 Hit Cost'] = df['L1 Hit Cost']/df['Total Load Cost']
  df.loc[:,'Relative L2 Hit Cost'] = df['L2 Hit Cost']/df['Total Load Cost']
  df.loc[:,'Relative L3 Hit Cost'] = df['L3 Hit Cost']/df['Total Load Cost']
  df.loc[:,'Relative Other Cache (clean) Hit Cost'] = df['Other Cache (clean) Hit Cost']/df['Total Load Cost']
  df.loc[:,'Relative Other Cache (dirty) Hit Cost'] = df['Other Cache (dirty) Hit Cost']/df['Total Load Cost']
  df.loc[:,'Relative DRAM Hit Cost'] = df['DRAM Hit Cost']/df['Total Load Cost']
  return df

def drop_unnecessary_columns(df):
  cols = ['Function (Full)', 'Module','Source File', 'Start Address']
  for col in df.columns:
    if col.startswith('OFFCORE_'):
      cols.append(col)
    elif col.startswith('Total_Latency'):
      cols.append(col)
    elif col.startswith('PAGE_WALKER_LOADS'):
      cols.append(col)
  for col in cols:
    df = df.drop(col, 1)
  #print df.columns
  #print cols
  return df

def load_report(reports_dir, report_name):
  report = reports_dir + '/' + report_name
  print "Loading report ", report
  pd.options.mode.chained_assignment = None
  raw_df = pd.read_csv(report + '.csv')
  raw_df.rename(columns=lambda c: c.replace('Hardware Event Count:','',1), inplace=True)
  interesting_fns = ('obamadb::ml::dot','obamadb::ml::scale_and_add', 'obamadb::SVMTask::execute')
  df = raw_df[raw_df.Function.apply(lambda f: f in interesting_fns)]
  df.loc[:,'Function'] = df['Function'].apply(lambda f: f.replace('obamadb::',''))
  df.loc[:,'report'] = report
  (_, data, threads) = extract_svm_cmd_params(report)
  df.loc[:,'data'] = data
  df.loc[:,'threads'] = threads
  df = extend_with_calc_columns(df)
  return df

def load_all_reports(reports_dir):
  reports = listdir(reports_dir)
  reports = [r.replace('.csv','') for r in reports if r.endswith('.csv')]
  dfs = [load_report(reports_dir, r) for r in reports]
  df = pd.concat(dfs)
  df = drop_unnecessary_columns(df)
  return df

def write_csv(df, output_dir):
  df.sort_values(['data','Function','threads'], inplace=True)
  #print_columns_in_df(df) # Use to change schema
  cols = get_output_schema()
  df = df[cols]
  # Write all data into raw CSV file
  raw_output = output_dir + '/' + 'raw_nav_svm_analysis.csv'
  print 'Dumping raw CSV to ', raw_output
  df.to_csv(raw_output)
  # Group data and write to "grouped" CSV file
  grouped_output = output_dir + '/' + 'grouped_nav_svm_analysis.csv'
  print 'Dumping grouped CSV to ', grouped_output
  df = df.groupby(['data','Function','threads']).mean()
  df.to_csv(grouped_output)

def main():
  directory = '/Users/nav/Programming/obamadb/ObamaDb-osx/repeat_n_times'
  #directory = '/Users/nav/Programming/obamadb/ObamaDb-osx/svm_with_scaling'
  reports_dir_name = 'reports'
  reports_dir_path = directory + '/' + reports_dir_name
  chdir(directory)
  for result_dir in listdir(directory):
    if result_dir != 'reports':
      generate_reports_for_result_dir_if_not_exists(result_dir, reports_dir_path)
  df = load_all_reports(reports_dir_path)
  write_csv(df, reports_dir_path)

if __name__ == "__main__":
  execfile('df_schema.py')
  main()



