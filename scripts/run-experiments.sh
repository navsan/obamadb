test_convergence() {
	rm *.out
	echo "" > results_conv.csv
	for thread in 1 2 4 8 10
	do
		./obamadb_main ../data/RCV1.train.tsv ../data/RCV1.test.tsv 0 $thread 1 > tmp.out
    sed -e '1,/^Convergence.*$/d' tmp.out >> results_conv.csv
		rm tmp.out
	done
}

test_epoch() {
  rm *.out
  echo "" > results_epoch.csv
  for thread in 1 2 4 8 10
  do
    ./obamadb_main ../data/RCV1.train.tsv ../data/RCV1.test.tsv 0 $thread 0 > tmp.out
    sed -e '1,/^>>>.*$/d' tmp.out >> results_epoch.csv
    rm tmp.out
  done
}
test_epoch
test_convergence
