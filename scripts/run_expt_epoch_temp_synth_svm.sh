#!/bin/bash
OBAMADB="/users/navsan/Programming/obamadb/build/obamadb_main"

run_cmd() {
  echo `date` | tee -a $2
  echo $1 | tee -a $2
  $1 2>&1 | tee -a $2
}

train_file=""
test_file=""
create_synth_files() { # $1:num_features
  nnz="50000000.0"
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
  for density in 0.001 0.004 0.006 0.008 0.01 0.02 0.04 0.06 0.08 0.1; do
  #for density in 0.002; do
    create_synth_files $num_features $density
    out_file="/users/navsan/Programming/obamadb/epoch_results/vary_model_size_and_density/output_${i}_m${num_features}_d${density}_${t}.txt"
    for t in 1 2 4 6 8 10; do
      run_cmd "$OBAMADB -threads $t -num_epochs 5 -num_trials 5 -train_file $train_file -test_file $test_file" "$out_file"
    done
  done
}

for i in `seq 1 5`; do
  for num_features in 1000.0 10000.0 100000.0 1000000.0 10000000.0; do
    run_experiment_for_num_features $num_features
  done
  for num_features in 2000.0 20000.0 200000.0 2000000.0 20000000.0; do
    run_experiment_for_num_features $num_features
  done
  for num_features in 4000.0 40000.0 400000.0 4000000.0 40000000.0; do
    run_experiment_for_num_features $num_features
  done
  for num_features in 6000.0 60000.0 600000.0 6000000.0 60000000.0; do
    run_experiment_for_num_features $num_features
  done
  for num_features in 8000.0 80000.0 800000.0 8000000.0 80000000.0; do
    run_experiment_for_num_features $num_features
  done
done

