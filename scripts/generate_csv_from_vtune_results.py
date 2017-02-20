#!/usr/bin/env python
from subprocess import call
import pandas as pd

report_dirs = ['r' + str(x).zfill(3) + 'runsa' for x in range(63, 98, 2)]
report_dirs += ['r' + str(x).zfill(3) + 'nav' for x in (58, 60, 59)]

def generate_reports():
  vtune = '/Applications/Intel\ VTune\ Amplifier\ XE\ 2017.app/Contents/MacOS/amplxe-cl'
  for report in report_dirs:
    d = {'vtune':vtune, 'report':report}
    hw_events_cmd = '{vtune} -report hw-events -r {report} -format csv -csv-delimiter comma -report-output {report}.csv'.format(**d)
    print hw_events_cmd
    call(hw_events_cmd, shell=True)
    summary_cmd = '{vtune} -report summary -r {report} -report-output {report}-summary.txt'.format(**d)
    print summary_cmd
    call(summary_cmd, shell=True)

def extract_cmd_params(report):
  txt = report + '-summary.txt'
  with open(txt, 'r') as f:
    for line in f:
      if line.startswith('Application Command Line'):
        parts = line.split(' ')
        parts = [p.replace('"', '') for p in parts]
        assert len(parts) > 10
        assert parts[5] == '-threads'
        threads = int(parts[6])
        data = parts[10]
        data = data.replace('/users/navsan/Programming/obamadb/data/_synth_svm_','')
        data = data.replace('.train.tsv','')
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

def load_report(report):
  pd.options.mode.chained_assignment = None
  raw_df = pd.read_csv(report + '.csv')
  raw_df.rename(columns=lambda c: c.replace('Hardware Event Count:','',1), inplace=True)
  df = raw_df[raw_df.Function.apply(lambda f: f in ('obamadb::ml::dot','obamadb::ml::scale_and_add'))]
  df.loc[:,'report'] = report
  (_, data, threads) = extract_cmd_params(report)
  df.loc[:,'data'] = data
  df.loc[:,'threads'] = threads
  df = extend_with_calc_columns(df)
  return df

def load_reports(reports):
  dfs = [load_report(r) for r in reports]
  df = pd.concat(dfs)
  df = drop_unnecessary_columns(df)
  return df

#generate_reports()
df = load_reports(report_dirs)
df.to_csv('nav_svm_analysis.csv')



