#!/bin/bash

run_cmd() {
  echo `date` | tee -a $2
  echo $1 | tee -a $2
  $1 2>&1 | tee -a $2
}

for i in `seq 1 5`; do
  OBAMADB=/users/navsan/Programming/obamadb/build/obamadb_main
  for s in 0 1 2 3 4 5 6; do
    for t in 1 2 4 6 8 10; do
      run_cmd "$OBAMADB -threads $t -num_epochs 10 -num_trials 5 -train_file /users/navsan/Programming/obamadb/data/_synth_svm_s${s}.train.tsv -test_file /users/navsan/Programming/obamadb/data/_synth_svm_s${s}.test.tsv" "epoch_results/output.${i}.s${s}.${t}.txt"
    done
  done
  for f in 1 2 3; do
    for t in 1 2 4 6 8 10; do
      run_cmd "$OBAMADB -threads $t -num_epochs 10 -num_trials 5 -train_file /users/navsan/Programming/obamadb/data/_synth_svm_f${f}.train.tsv -test_file /users/navsan/Programming/obamadb/data/_synth_svm_f${f}.test.tsv" "epoch_results/output.${i}.f${f}.${t}.txt"
    done
  done
  for t in 1 2 4 6 8 10; do
    run_cmd "$OBAMADB -threads $t -num_epochs 10 -num_trials 5 -train_file /users/navsan/Programming/obamadb/data/RCV1.train.tsv -test_file /users/navsan/Programming/obamadb/data/RCV1.test.tsv" "epoch_results/output.${i}.RCV1.${t}.txt"
  done
done

