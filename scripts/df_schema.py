def print_columns_in_df(df):
  cols = df.columns.tolist()
  print "["
  for col in cols:
    print "  '" + col + "',"
  print "]"

def get_output_schema():
  return [
       # "Group by" columns
      'data',
      'threads',
      'Function',

      # Important aggregate counts
      'CPI',
      'INST_RETIRED.ANY',
      'CPU_CLK_UNHALTED.THREAD',
      'NumLoads',
      'CYCLE_ACTIVITY.CYCLES_LDM_PENDING',
      'CYCLE_ACTIVITY.STALLS_LDM_PENDING',

      # Hit Ratios
      'L1 Hit Ratio',
      'LFB Hit Ratio',
      'L1 + LFB Hit Ratio',
      'L2 Hit Ratio',
      'L3 Hit Ratio',
      'Other Cache (clean) Hit Ratio',
      'Other Cache (dirty) Hit Ratio',
      'DRAM Hit Ratio',

      # Hit Costs
      'L1 Hit Cost',
      'L2 Hit Cost',
      'L3 Hit Cost',
      'Other Cache (clean) Hit Cost',
      'Other Cache (dirty) Hit Cost',
      'DRAM Hit Cost',
      'DTLB_LOAD_MISSES.WALK_DURATION',
      'Total Load Cost',

      # Relative Hit Costs
      'Relative L1 Hit Cost',
      'Relative L2 Hit Cost',
      'Relative L3 Hit Cost',
      'Relative Other Cache (clean) Hit Cost',
      'Relative Other Cache (dirty) Hit Cost',
      'Relative DRAM Hit Cost',

      # Other columns
      'CPU_CLK_UNHALTED.REF_TSC',
      'CPU_CLK_UNHALTED.THREAD_P',
      'CYCLE_ACTIVITY.CYCLES_L1D_PENDING',
      'CYCLE_ACTIVITY.CYCLES_L2_PENDING',
      'CYCLE_ACTIVITY.CYCLES_NO_EXECUTE',
      'CYCLE_ACTIVITY.STALLS_L1D_PENDING',
      'CYCLE_ACTIVITY.STALLS_L2_PENDING',
      'DTLB_LOAD_MISSES.MISS_CAUSES_A_WALK',
      'DTLB_LOAD_MISSES.STLB_HIT',
      'DTLB_STORE_MISSES.MISS_CAUSES_A_WALK',
      'DTLB_STORE_MISSES.STLB_HIT',
      'IDQ_UOPS_NOT_DELIVERED.CORE',
      'INT_MISC.RECOVERY_CYCLES',
      'L1D.REPLACEMENT',
      'L1D_PEND_MISS.PENDING',
      'L1D_PEND_MISS.PENDING_CYCLES',
      'L2_LINES_IN.ALL',
      'L2_LINES_IN.E',
      'L2_LINES_IN.I',
      'L2_LINES_IN.S',
      'L2_RQSTS.ALL_DEMAND_DATA_RD',
      'L2_RQSTS.ALL_DEMAND_MISS',
      'L2_RQSTS.ALL_DEMAND_REFERENCES',
      'L2_RQSTS.DEMAND_DATA_RD_HIT',
      'L2_RQSTS.DEMAND_DATA_RD_MISS',
      'L2_RQSTS.L2_PF_HIT',
      'L2_RQSTS.L2_PF_MISS',
      'L2_RQSTS.MISS',
      'L2_RQSTS.REFERENCES',
      'MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_HITM_PS',
      'MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_HIT_PS',
      'MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_MISS_PS',
      'MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_NONE_PS',
      'MEM_LOAD_UOPS_L3_MISS_RETIRED.LOCAL_DRAM_PS',
      'MEM_LOAD_UOPS_RETIRED.HIT_LFB_PS',
      'MEM_LOAD_UOPS_RETIRED.L1_HIT_PS',
      'MEM_LOAD_UOPS_RETIRED.L1_MISS_PS',
      'MEM_LOAD_UOPS_RETIRED.L2_HIT_PS',
      'MEM_LOAD_UOPS_RETIRED.L2_MISS_PS',
      'MEM_LOAD_UOPS_RETIRED.L3_HIT_PS',
      'MEM_LOAD_UOPS_RETIRED.L3_MISS_PS',
      'MEM_TRANS_RETIRED.LOAD_LATENCY_GT_4',
      'UOPS_EXECUTED.CYCLES_GE_2_UOPS_EXEC',
      'UOPS_EXECUTED.CYCLES_GE_3_UOPS_EXEC',
      'UOPS_EXECUTED.CYCLES_GE_4_UOPS_EXEC',
      'UOPS_EXECUTED.STALL_CYCLES',
      'UOPS_ISSUED.ANY',
      'UOPS_RETIRED.RETIRE_SLOTS_PS',
      'report',
  ]
