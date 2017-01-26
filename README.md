ObamaDB
----
Async machine learning experiments.

```
./build.sh
cd build
./obamadb_main -help
```

Flags:
```
  Flags from /Users/cramja/workspace/obamadb/main.cpp:
    -algorithm (The machine learning algorithm to use. Select one of [svm,
      mc].) type: string default: "svm"
    -measure_convergence (If true, an observer thread will collect copies of
      the model as the algorithm does its first iteration. Useful for the SVM.)
      type: bool default: false
    -num_epochs (The number of passes over the training data while training the
      model.) type: int64 default: 10
    -test_file (The TSV format file to test the algorithm over.) type: string
      default: ""
    -threads (The number of threads the system will use to run the machine
      learning algorithm) type: int64 default: 1
    -train_file (The TSV format file to train the algorithm over.) type: string
      default: ""
    -verbose (Print out extra diagnostic information.) type: bool
      default: false
```