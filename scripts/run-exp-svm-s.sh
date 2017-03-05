#!/bin/bash

# used to test increasing density's effect on the SVM problem

EXE=/fastdisk/obamadb/build/obamadb_main
DAT=/fastdisk/obamadb/data/

test_epoch() {
  rm *.out
  echo "" > results_svm_sparsity.csv
  for sparsity in 0 1 2 3 4 5 6
  do
    for thread in 1 2 4 8 10
    do
    	FNAME=s$sparsity_t$thread.out
    	$EXE -train_file $DAT/_synth_svm_s$sparsity.train.tsv -test_file $DAT/_synth_svm_s$sparsity.test.tsv -threads $thread -num_epochs 20 -num_trials 5 -core_affinities=0,1,2,3,4,5,6,7,8,9 -verbose true > $FNAME 
    	echo $sparsity,$thread,`tail -n 1 $FNAME` >> results_epoch.csv
	  done
  done
}

test_epoch
