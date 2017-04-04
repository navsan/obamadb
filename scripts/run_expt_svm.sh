#!/bin/bash
OBAMADB="/users/navsan/Programming/obamadb/build/obamadb_main"

run_cmd() {
  echo `date` | tee -a $2
  echo $1 | tee -a $2
  $1 2>&1 | tee -a $2
}

run_vtune_ge_cmd() {
  echo
  echo "-----------------------------------------------------------------"
  echo
  echo `date` | tee -a $2
  echo $1 | tee -a $2
  echo "Running NAV experiment: $1"
/slowdisk/opt/intel/vtune_amplifier_xe/bin64/amplxe-cl -collect general-exploration \
-app-working-dir /users/navsan/Programming/obamadb/build -- $1 2>&1 | tee -a $2
}

run_vtune_cmd() {
  echo
  echo "-----------------------------------------------------------------"
  echo
  echo `date` | tee -a $2
  echo $1 | tee -a $2
/slowdisk/opt/intel/vtune_amplifier_xe/bin64/amplxe-cl -collect-with runsa \
-knob collect-io-waits=true \
-knob stack-size=1000 \
-knob event-config=CPU_CLK_UNHALTED.REF_TSC:sa=2000003,CPU_CLK_UNHALTED.THREAD:sa=2000003,CPU_CLK_UNHALTED.THREAD_P:sa=2000003,CYCLE_ACTIVITY.CYCLES_L1D_PENDING:sa=2000003,CYCLE_ACTIVITY.CYCLES_L2_PENDING:sa=2000003,CYCLE_ACTIVITY.CYCLES_LDM_PENDING:sa=2000003,CYCLE_ACTIVITY.CYCLES_NO_EXECUTE:sa=2000003,CYCLE_ACTIVITY.STALLS_L1D_PENDING:sa=2000003,CYCLE_ACTIVITY.STALLS_L2_PENDING:sa=2000003,CYCLE_ACTIVITY.STALLS_LDM_PENDING:sa=2000003,DTLB_LOAD_MISSES.MISS_CAUSES_A_WALK:sa=100003,DTLB_LOAD_MISSES.STLB_HIT:sa=2000003,DTLB_LOAD_MISSES.WALK_DURATION:sa=2000003,DTLB_STORE_MISSES.MISS_CAUSES_A_WALK:sa=100003,DTLB_STORE_MISSES.STLB_HIT:sa=100003,IDQ_UOPS_NOT_DELIVERED.CORE:sa=2000003,INST_RETIRED.ANY:sa=2000003,INT_MISC.RECOVERY_CYCLES:sa=2000003,L1D.REPLACEMENT:sa=2000003,L1D_PEND_MISS.PENDING:sa=2000003,L1D_PEND_MISS.PENDING_CYCLES:sa=2000003,L2_LINES_IN.ALL:sa=100003,L2_LINES_IN.E:sa=100003,L2_LINES_IN.I:sa=100003,L2_LINES_IN.S:sa=100003,L2_RQSTS.ALL_DEMAND_DATA_RD:sa=200003,L2_RQSTS.ALL_DEMAND_MISS:sa=200003,L2_RQSTS.ALL_DEMAND_REFERENCES:sa=200003,L2_RQSTS.DEMAND_DATA_RD_HIT:sa=200003,L2_RQSTS.DEMAND_DATA_RD_MISS:sa=200003,L2_RQSTS.L2_PF_HIT:sa=200003,L2_RQSTS.L2_PF_MISS:sa=200003,L2_RQSTS.MISS:sa=200003,L2_RQSTS.REFERENCES:sa=200003,MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_HITM_PS:sa=20011,MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_HIT_PS:sa=20011,MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_MISS_PS:sa=20011,MEM_LOAD_UOPS_L3_HIT_RETIRED.XSNP_NONE_PS:sa=100003,MEM_LOAD_UOPS_L3_MISS_RETIRED.LOCAL_DRAM_PS:sa=100003,MEM_LOAD_UOPS_RETIRED.HIT_LFB_PS:sa=100003,MEM_LOAD_UOPS_RETIRED.L1_HIT_PS:sa=2000003,MEM_LOAD_UOPS_RETIRED.L1_MISS_PS:sa=100003,MEM_LOAD_UOPS_RETIRED.L2_HIT_PS:sa=100003,MEM_LOAD_UOPS_RETIRED.L2_MISS_PS:sa=50021,MEM_LOAD_UOPS_RETIRED.L3_HIT_PS:sa=50021,MEM_LOAD_UOPS_RETIRED.L3_MISS_PS:sa=100003,MEM_TRANS_RETIRED.LOAD_LATENCY_GT_4:sa=100003,UOPS_EXECUTED.CYCLES_GE_2_UOPS_EXEC:sa=2000003,UOPS_EXECUTED.CYCLES_GE_3_UOPS_EXEC:sa=2000003,UOPS_EXECUTED.CYCLES_GE_4_UOPS_EXEC:sa=2000003,UOPS_EXECUTED.STALL_CYCLES:sa=2000003,UOPS_ISSUED.ANY:sa=2000003,UOPS_RETIRED.RETIRE_SLOTS_PS:sa=2000003 \
-knob collectMemBandwidth=false \
-app-working-dir /users/navsan/Programming/obamadb/build -- $1 2>&1 | tee -a $2
}


train_file=""
test_file=""
create_synth_files() { # $1:num_features $2:density
  nnz="100000000.0"
  num_features="$1"
  density=$2
  num_rows=`echo "scale=1; $nnz / $density / $num_features" | bc -l`
  train_file="/users/navsan/Programming/obamadb/data/_synth_svm_m${num_features}_d${density}.train.tsv"
  test_file="/users/navsan/Programming/obamadb/data/_synth_svm_m${num_features}_d${density}.test.tsv"
  echo "$num_rows $num_features $density" > $train_file
  echo "10.0  $num_features $density" > $test_file
}

run_experiment_for_num_features() {
  num_features="$1"
  for density in 0.01 0.02; do
    create_synth_files $num_features $density
    for t in 1 2 4 6 8 10; do
      out_file="/users/navsan/Programming/obamadb/epoch_results/after_inlining/output_${i}_m${num_features}_d${density}_${t}.txt"
      run_vtune_ge_cmd "$OBAMADB -threads $t -num_epochs 5 -num_trials 10 -train_file $train_file -test_file $test_file" "$out_file"
    done
  done
}

for i in `seq 1 5`; do
  for num_features in 1000.0 10000.0 100000.0 1000000.0 10000000.0; do
    run_experiment_for_num_features $num_features
  done
done

