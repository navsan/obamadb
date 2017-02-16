#!/bin/bash

# used to test increasing model size's effect on HW!
set -e
EXE=/fastdisk/obamadb/build/obamadb_main
DAT=/fastdisk/obamadb/data/

test_epoch() {
  rm *.out
  echo "" > results_mc_model_size.csv
  for model_size in 0 1 2
  do
    for thread in 1 2 4 8 10
    do
    	FNAME=s$sparsity_t$thread.out
    	$EXE -train_file $DAT/_synth_mc_d$model_size.train.tsv -test_file $DAT/_synth_mc_d$model_size.probe.tsv -threads $thread -num_epochs 20 -num_trials 5 -core_affinities=0,1,2,3,4,5,6,7,8,9 -verbose true -rank 12 -algorithm mc > $FNAME 
    	echo $model_size,$thread,`tail -n 1 $FNAME` >> results_mc_model_size.csv
	  done
  done
}

test_epoch
