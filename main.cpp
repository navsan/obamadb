#include "ml/ConvergenceObserver.h"
#include "ml/MCTask.h"
#include "ml/MLTask.h"
#include "ml/SVMTask.h"
#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/IO.h"
#include "storage/Matrix.h"
#include "utils/LogUtils.h"
#include "utils/ValidationUtils.h"

#include <algorithm>
#include <gflags/gflags.h>
#include <string>
#include <unistd.h>
#include <vector>

DEFINE_int64(threads, 1, "The number of threads the system will use to run the machine learning algorithm");
DEFINE_validator(threads, &ValidateThreads);

DEFINE_bool(measure_convergence, false, "If true, an observer thread will collect copies of the model"
  " as the algorithm does its first iteration. Useful for the SVM.");

DEFINE_string(algorithm, "svm", "The machine learning algorithm to use. Select one of [svm, mc].");
DEFINE_validator(algorithm, &ValidateAlgorithm);

DEFINE_string(train_file, "", "The TSV format file to train the algorithm over.");
DEFINE_string(test_file, "", "The TSV format file to test the algorithm over.");

DEFINE_bool(verbose, false, "Print out extra diagnostic information.");

DEFINE_int64(num_epochs, 10, "The number of passes over the training data while training the model.");

DEFINE_int64(num_trials, 1, "The number of trials to perform."
  "This means the number of times we will train a model."
  "This is useful for computing the variance/stddev between trials.");
DEFINE_string(core_affinities, "-1", "A comma separated list of cores to have threads bind to."
  " The program will greedily use core, so over specify if you like. Ex: "
  " -core_affinities 0,1,2,3 -threads 2 is valid");

// DEFINE_bool(apply_hinge_loss, true, "For SVM: whether to selectively apply gradient update only for mispredictions.")
// DEFINE_bool(apply_regularization, true, "For SVM: whether to use a regularization term in the loss function and gradient.")
DEFINE_int64(rank, 10, "The rank of the LR factoring matrices used in Matrix Completion");


namespace obamadb {
void printSVMEpochStats(Matrix const* matTrain, Matrix const* matTest,
                        fvector const& theta, int iteration, float timeTrain) {
  if (!FLAGS_verbose) {
    return;
  }

  if (iteration == -1)
    VPRINT(
        "epoch, train_time, train_fraction_misclassified, train_RMS_loss, "
        "test_fraction_misclassified, test_RMS_loss\n");

  double const trainRmsLoss = SVMTask::rmsErrorLoss(theta, matTrain->blocks_);
  double const testRmsLoss = SVMTask::rmsErrorLoss(theta, matTest->blocks_);
  double const trainFractionMisclassified =
      SVMTask::fractionMisclassified(theta, matTrain->blocks_);
  double const testFractionMisclassified =
      SVMTask::fractionMisclassified(theta, matTest->blocks_);

  printf("%-3d, %.3f, %.4f, %.2f, %.4f, %.2f\n", iteration, timeTrain,
         trainFractionMisclassified, trainRmsLoss, testFractionMisclassified,
         testRmsLoss);
}

/**
 * @return A vector of the epoch times.
 */
std::vector<double> trainSVM(Matrix* mat_train, Matrix* mat_test,
                             const int trial_num = 0) {
  SVMHyperParams* svm_params = DefaultSVMHyperParams<num_t>(mat_train->blocks_);
  DCHECK_EQ(svm_params->degrees.size(), maxColumns(mat_train->blocks_));
  fvector sharedTheta = fvector::GetRandomFVector(mat_train->numColumns_);

  // Arguments to the thread pool.
  std::vector<void*> threadStates;
  std::vector<std::function<void(int, void*)>> threadFns;

  // Create an observer thread, if specified
  std::unique_ptr<ConvergenceObserver> observer;
  if (FLAGS_measure_convergence) {
    observer.reset(new ConvergenceObserver(&sharedTheta));
    auto observer_fn = obamadb::observerThreadFn;
    threadFns.push_back(observer_fn);
    threadStates.push_back(observer.get());
  }

  // Create the tasks for the thread pool.
  // Roughly allocates work.
  std::vector<std::unique_ptr<DataView>> data_views;

  allocateBlocks(FLAGS_threads, mat_train->blocks_, data_views);
  // Create tasks
  auto update_fn = [](int tid, void* state) {
    SVMTask* task = reinterpret_cast<SVMTask*>(state);
    task->execute(tid, nullptr);
  };
  std::vector<std::unique_ptr<SVMTask>> tasks(FLAGS_threads);
  for (int i = 0; i < tasks.size(); i++) {
    tasks[i].reset(
        new SVMTask(data_views[i].release(), &sharedTheta, svm_params));
    threadStates.push_back(tasks[i].get());
    threadFns.push_back(update_fn);
  }

  ThreadPool tp(threadFns, threadStates);

  // If we are observing convergence, the thread pool must be referenced.
  if (observer) {
    observer->threadPool_ = &tp;
  }

  tp.begin();

  printSVMEpochStats(mat_train, mat_test, sharedTheta, -1, -1);
  double totalTrainTime = 0.0;
  std::vector<double> epoch_times;
  for (int cycle = 0; cycle < FLAGS_num_epochs; cycle++) {
    auto time_start = std::chrono::steady_clock::now();
    tp.cycle();
    auto time_end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> time_ms = time_end - time_start;
    double elapsedTimeSec = (time_ms.count()) / 1e3;
    totalTrainTime += elapsedTimeSec;

    printSVMEpochStats(mat_train, mat_test, sharedTheta, cycle, elapsedTimeSec);
    epoch_times.push_back(elapsedTimeSec);
  }
  tp.stop();

  if (!FLAGS_verbose) {
    if (trial_num == 0)
      std::cout << "train_file, rows, features, sparsity, nnz, num_threads, "
                   "trial, epoch, train_time\n";

    for (int i = 0; i < epoch_times.size(); ++i) {
      std::cout << FLAGS_train_file << ", " << mat_train->numRows_ << ", "
                << mat_train->numColumns_ << ", " << mat_train->getSparsity()
                << ", " << mat_train->getNNZ() << ", " << (int)FLAGS_threads
                << ", " << trial_num << ", " << i << ", " << epoch_times[i]
                << "\n";
    }
  }

  if (FLAGS_measure_convergence) {
    printf("Convergence Info (%d measures)\n",
           (int)observer->observedModels_.size());

    for (int obs_idx = 0; obs_idx < observer->observedModels_.size();
         obs_idx++) {
      std::uint64_t timeObs = observer->observedTimes_[obs_idx];
      fvector const& thetaObs = observer->observedModels_[obs_idx];
      printf("%d,%llu,%.4f,%.4f\n", (int)FLAGS_threads, timeObs,
             SVMTask::rmsErrorLoss(thetaObs, mat_test->blocks_),
             SVMTask::fractionMisclassified(thetaObs, mat_test->blocks_));
    }
  }
  return epoch_times;
  }

  void runSvmExperiment() {
    std::unique_ptr<Matrix> mat_train;
    std::unique_ptr<Matrix> mat_test;

    VPRINT("Reading input files...\n");
    VPRINTF("Loading: %s\n", FLAGS_train_file.c_str());
    PRINT_TIMING({mat_train.reset(IO::load(FLAGS_train_file));});
    VSTREAM(*mat_train);

    if (FLAGS_test_file.size() == 0) {
      VPRINT("Test file not specified, using a sample of the train file\n");
      mat_test.reset(mat_train->sample(0.2));
      VSTREAM(*mat_test);
    } else {
      VPRINTF("Loading: %s\n", FLAGS_test_file.c_str());
      PRINT_TIMING({mat_test.reset(IO::load(FLAGS_test_file));});
      VSTREAM(*mat_test);
    }
    CHECK_EQ(mat_test->numColumns_, mat_train->numColumns_)
      << "Train and Test matrices had differing number of features.";

    std::vector<double> all_epoch_times;
    for (int i = 0; i < FLAGS_num_trials; i++) {
      std::vector<double> times = trainSVM(mat_train.get(), mat_test.get(), i);
      all_epoch_times.insert(all_epoch_times.end(), times.begin(), times.end());

      if (FLAGS_num_trials != i -1) {
        usleep(1e7);
      }
    }

    // calculate variance, etc.
    if (FLAGS_verbose) {
      printf("epoch runtimes:\n");
      for (double time : all_epoch_times) {
        printf("%f\n", time);
      }
    }

    // printf("epoch runtime summary:\nmean,variance,stddev,stderr\n%f,%f,%f,%f\n",
    //        stats::mean<double>(all_epoch_times),
    //        stats::variance<double>(all_epoch_times),
    //        stats::stddev<double>(all_epoch_times),
    //        stats::stderr<double>(all_epoch_times));
  }

  void printMCEpochStats(int epoch, double time, MCState const * state, UnorderedMatrix const * probe_mat) {
    if (FLAGS_verbose) {
      double rmse = MCTask::rmse(state, probe_mat);
      printf("%d,%.6f,%.4f\n",epoch, time, rmse);
    }
  }

  std::vector<double> trainMC(const UnorderedMatrix* train_matrix,
                              const UnorderedMatrix* probe_matrix) {
    int const rank = FLAGS_rank;
    MCState* mcstate = new MCState(train_matrix, rank); // TODO: this is a memory leak which
		// fixes a mysterious memory corruption bug on ubuntu systems. It should be a smart pointer
		// and also, there should be no memory corruption
    if (FLAGS_verbose) {
      printf("Model matrix properties (L,R):\n");
      std::cout << *mcstate->mat_l << std::endl;
      std::cout << *mcstate->mat_r << std::endl;
    }

    // Arguments to the thread pool.
    std::vector<std::function<void(int, void*)>> threadFns;
    auto update_fn = [](int tid, void* state) {
      MCTask* task = reinterpret_cast<MCTask*>(state);
      task->execute(tid, nullptr);
    };
    std::vector<std::unique_ptr<MCTask>> tasks(FLAGS_threads);
    std::vector<void*> tp_states;
    for (int i = 0; i < tasks.size(); i++) {
      tasks[i].reset(new MCTask(FLAGS_threads, train_matrix, mcstate));
      tp_states.push_back(tasks[i].get());
      threadFns.push_back(update_fn);
    }

    ThreadPool tp(threadFns, tp_states);
    tp.begin();

    VPRINT("epoch, train_time, probe_RMS_loss\n");
    printMCEpochStats(-1, -1, mcstate, probe_matrix);
    double totalTrainTime = 0.0;
    std::vector<double> epoch_times;
    for (int cycle = 0; cycle < FLAGS_num_epochs; cycle++) {
      auto time_start = std::chrono::steady_clock::now();
      tp.cycle();
      auto time_end = std::chrono::steady_clock::now();
      std::chrono::duration<double, std::milli> time_ms = time_end - time_start;
      double elapsedTimeSec = (time_ms.count())/ 1e3;
      totalTrainTime += elapsedTimeSec;

      printMCEpochStats(cycle, elapsedTimeSec, mcstate, probe_matrix);
      epoch_times.push_back(elapsedTimeSec);
    }
    tp.stop();
    return epoch_times;
  }

  void trainMC() {
    std::unique_ptr<UnorderedMatrix> train_matrix;
    std::unique_ptr<UnorderedMatrix> probe_matrix;

    VPRINT("Reading input files...\n");
    VPRINTF("Loading: %s\n", FLAGS_train_file.c_str());
    PRINT_TIMING({train_matrix.reset(IO::loadUnorderedMatrix(FLAGS_train_file));});
    VSTREAM(*train_matrix);

    VPRINTF("Loading: %s\n", FLAGS_test_file.c_str());
    PRINT_TIMING({probe_matrix.reset(IO::loadUnorderedMatrix(FLAGS_test_file));});
    VSTREAM(*probe_matrix);

    CHECK_LE(probe_matrix->numColumns(), train_matrix->numColumns());
    CHECK_LE(probe_matrix->numRows(), train_matrix->numRows());

    std::vector<double> all_epoch_times;
    for (int i = 0; i < FLAGS_num_trials; i++) {
      std::vector<double> times = trainMC(train_matrix.get(), probe_matrix.get());
      all_epoch_times.insert(all_epoch_times.end(), times.begin(), times.end());

      if (FLAGS_num_trials != i - 1) {
        usleep(1e7);
      }
    }

    // calculate variance, etc.
    if (FLAGS_verbose) {
      printf("epoch runtimes:\n");
      for (double time : all_epoch_times) {
        printf("%f\n", time);
      }
    }

    printf("epoch runtime summary:\nmean,variance,stddev,stderr\n%f,%f,%f,%f\n",
           stats::mean<double>(all_epoch_times),
           stats::variance<double>(all_epoch_times),
           stats::stddev<double>(all_epoch_times),
           stats::stderr<double>(all_epoch_times));
  }

  int main(int argc, char** argv) {
    // Pin main thread to 1st socket, 1st core, for simplicity
    threading::setCoreAffinity(1);

    ::google::InitGoogleLogging(argv[0]);
    ::gflags::SetUsageMessage(std::string(argv[0]) + " -help");
    ::gflags::SetVersionString("0.0");
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::vector<int> affinities = GetIntList(FLAGS_core_affinities);
    if (affinities[0] != -1) {
      threading::setCoreAffinity(affinities[0]);
    } else {
      LOG(INFO) << "Main thread affinitized to core 0";
      threading::setCoreAffinity(0);
    }

    if (FLAGS_algorithm.compare("svm") == 0) {
      runSvmExperiment();
    } else if (FLAGS_algorithm.compare("mc") == 0) {
      LOG_IF(WARNING, FLAGS_measure_convergence)
        << "Measure convergence not implemented for MC";
      trainMC();
    } else {
      LOG(FATAL) << "unknown training algorithm";
    }

    return 0;
  }
} // namespace obamadb

int main(int argc, char **argv) {
  return obamadb::main(argc, argv);
}
