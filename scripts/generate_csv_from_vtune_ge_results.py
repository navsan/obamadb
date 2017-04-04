#!/usr/bin/env python
from importlib import import_module
from os import chdir, listdir
from os.path import isfile
import pandas as pd
from statsmodels.regression.linear_model import OLS
from subprocess import call

base_dir = '/Users/nav/Programming/obamadb_results/ObamaDb-osx/'

def generate_reports():
  call('./generate_csv_from_vtune_ge_results.sh')

def load_summary_report():
  cfg = pd.read_csv(base_dir + 'synth_cfg_files.csv', index_col=False)
  summary = pd.read_csv(base_dir +  'after_inlining_summary_combined.csv', index_col=False)
  summary = summary[['result','threads','num_epochs','num_trials','train_file']]
  summary['train_file_name'] = summary.apply(lambda r:r['train_file'].split('/')[-1], axis=1)
  summary = summary.merge(cfg, left_on='train_file_name', right_on='filename')
  summary = summary.drop(['train_file', 'filename'],1)
  summary['total_num_epochs'] = summary['num_epochs'] * summary['num_trials']
  summary['model_size_kB'] = summary['num_features'] * 4.0 / 1000
  summary['num_nonzeroes'] = summary['num_features'] * summary['num_rows'] * summary['density']
  return summary

def map_to_schema(df, schema):
  existing_cols = set(df.columns)
  cols = [col for col in schema if col in existing_cols]
  return df[cols]

def load_hotspots_report():
  summary = load_summary_report()
  hotspots = pd.read_csv(base_dir +  'after_inlining_hotspots_combined.csv', index_col=False, delimiter='|')
  hotspots = hotspots.merge(summary, left_on='result', right_on='result')
  hotspots['InstructionsPerEpoch'] = hotspots['Instructions Retired'] / hotspots['total_num_epochs']
  hotspots['TotalClockticksPerEpoch'] = hotspots['Clockticks'] * hotspots['threads'] / hotspots['total_num_epochs']
  hotspots = map_to_schema(hotspots, get_hotspots_schema())
  return hotspots

def hwevents_extend_with_calc_columns(df):
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

#def hwevents_drop_unnecessary_columns(df):
#  cols = ['Function (Full)', 'Module','Source File', 'Start Address']
#  for col in df.columns:
#    if col.startswith('OFFCORE_'):
#      cols.append(col)
#    elif col.startswith('Total_Latency'):
#      cols.append(col)
#    elif col.startswith('PAGE_WALKER_LOADS'):
#      cols.append(col)
#  df = df.drop(cols, 1)
#  return df

def load_hwevents_report():
  interesting_fns = ('obamadb::ml::dot','obamadb::SVMTask::execute')
  dfs = []
  for r in xrange(433, 703):
    result = 'r{r}ge'.format(**{'r':str(r).zfill(3)})
    csv = base_dir + '{result}/{result}-hwevents.csv'.format(**{'result':result})
    if not isfile(csv):
      print 'WARNING: could not find file', csv, '. Skipping!'
      continue
    raw_df = pd.read_csv(csv, delimiter='|', index_col=False)
    raw_df.rename(columns=lambda c: c.replace('Hardware Event Count:','',1), inplace=True)
    raw_df = raw_df[raw_df.Function.apply(lambda f: f in interesting_fns)]
    raw_df['result'] = result
    dfs.append(raw_df)
  hwevents = pd.concat(dfs)
  #for col in hwevents.columns: print '   ', col, ','
  hwevents = hwevents_extend_with_calc_columns(hwevents)
  #hwevents = hwevents_drop_unnecessary_columns(hwevents)
  summary = load_summary_report()
  hwevents = hwevents.merge(summary, left_on='result', right_on='result')
  hwevents = map_to_schema(hwevents, get_hwevents_schema())
  return hwevents

def write_combined_report(combined):
  combined.to_csv(base_dir + 'after_inlining_combined.csv', index=False)

def regression(df, ycol, xcols):
  model = OLS(df[ycol],df[xcols]).fit()
  print model.summary()

def perform_regression_on_instructions(df):
  df = df[['Function','InstructionsPerEpoch','num_rows','num_nonzeroes']]
  df_dot = df[df['Function'] == 'obamadb::ml::dot']
  df_exec = df[df['Function'] == 'obamadb::SVMTask::execute']

  print("Regression Summary for obamadb::ml::dot")
  regression(df_dot, 'InstructionsPerEpoch', ['num_rows','num_nonzeroes'])

  print("Regression Summary for obamadb::SVMTask::execute")
  regression(df_exec, 'InstructionsPerEpoch', ['num_rows','num_nonzeroes'])

def perform_regression_on_clockticks(df):
  df = df[['Function','TotalClockticksPerEpoch','num_rows','num_nonzeroes']]
  df_dot = df[df['Function'] == 'obamadb::ml::dot']
  df_exec = df[df['Function'] == 'obamadb::SVMTask::execute']

  print("Regression Summary for obamadb::ml::dot")
  regression(df_dot, 'TotalClockticksPerEpoch', ['num_rows','num_nonzeroes'])

  print("Regression Summary for obamadb::SVMTask::execute")
  regression(df_exec, 'TotalClockticksPerEpoch', ['num_rows','num_nonzeroes'])

def main():
  generate_reports()
  hotspots = load_hotspots_report()
  #for col in hotspots.columns: print "   '" + col + "',"
  hotspots.to_csv(base_dir + 'after_inlining_combined.csv', index=False)
  hwevents = load_hwevents_report()
  hwevents.to_csv(base_dir + 'after_inlining_combined2.csv', index=False)
  #perform_regression_on_instrctions(hotspots)
  #perform_regression_on_clockicks(hotspots)

execfile('hwevents_df_schema.py')
execfile('hotspots_df_schema.py')
main()

