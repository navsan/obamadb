#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/IO.h"
#include "storage/Matrix.h"
#include "storage/MCTask.h"
#include "storage/MLTask.h"
#include "storage/SVMTask.h"

#include <algorithm>
#include <gflags/gflags.h>
#include <string>
#include <unistd.h>
#include <vector>
#include <cmath>

DEFINE_string(train_file, "", "The TSV format file to train the algorithm over.");
DEFINE_string(test_file, "", "The TSV format file to test the algorithm over.");

DEFINE_bool(verbose, false, "Print out extra diagnostic information.");

DEFINE_int64(num_epochs, 10, "The number of passes over the training data while training the model.");

// DEFINE_int64(num_trials, 1, "The number of trials to perform."
//   "This means the number of times we will train a model."
//   "This is useful for computing the variance/stddev between trials.");
// DEFINE_string(core_affinities, "-1", "A comma separated list of cores to have threads bind to."
//   " The program will greedily use core, so over specify if you like. Ex: "
//   " -core_affinities 0,1,2,3 -threads 2 is valid");

DEFINE_int64(num_param_values, 7, "Number of grid points in grid search for mu in SVM.");
DEFINE_double(mu_min, 1e-3, "Minimum value of mu in grid search for SVM.");
DEFINE_double(mu_max, 1e3, "Maximum value of mu in grid search for SVM.");
DEFINE_int64(num_threads_per_grid_point, 1, "Number of parallel SGD worker threads used to evaluate each grid point.");

#define VPRINT(str) { if(FLAGS_verbose) { printf(str); } }
#define VPRINTF(str, ...) { if(FLAGS_verbose) { printf(str, __VA_ARGS__); } }
#define VSTREAM(obj) {if(FLAGS_verbose){ std::cout << obj <<std::endl; }}

namespace obamadb {

  /**
   * Allocates Datablocks to DataViews. Dataviews will then be given to threads in the form of tasks.
   * @param num_threads How many dataviews to allocate and distribute amongst.
   * @param data_blocks The set of training data.
   * @return Vector of Dataviews.
   */
  template<class T>
  void allocateBlocks(const int num_threads,
                      const std::vector<SparseDataBlock<T> *> &data_blocks,
                      std::vector<std::unique_ptr<DataView>>& views) {
    CHECK(views.size() == 0) << "Only accepts empty view vectors";
    CHECK_GE(data_blocks.size(), views.size())
      << "Partitioned data would not distribute to all threads."
      << " Use fewer threads.";

    for (int i = 0; i < data_blocks.size(); i++) {
      if (i < num_threads) {
        views.push_back(std::unique_ptr<DataView>(new DataView()));
      }
      SparseDataBlock<T> const *dbptr = data_blocks[i];
      views[i % num_threads]->appendBlock(dbptr);
    }
  }

  struct SVMTrainingLogRecord;
  struct SVMTrainerParams {
    SVMTrainerParams(Matrix *mat_train_, Matrix *mat_test_, SVMParams *svm_params_, int num_threads_, int starting_worker_id_for_core_assignment_)
    : mat_train(mat_train_), mat_test(mat_test_), svm_params(svm_params_), 
      num_threads(num_threads_), 
      starting_worker_id_for_core_assignment(starting_worker_id_for_core_assignment_),
      out_log_record(nullptr) {};
    
    Matrix *mat_train;
    Matrix *mat_test;
    SVMParams *svm_params;
    int num_threads;
    int starting_worker_id_for_core_assignment;
    SVMTrainingLogRecord* out_log_record;
  };

  struct SVMTrainingLogRecord {
    double train_rms_loss;
    double test_rms_loss;
    double train_fraction_misclassified;
    double test_fraction_misclassified;
    double total_train_time; 
    double mean_epoch_time;
    double stddev_epoch_time;

    static SVMTrainingLogRecord *Create(Matrix const * matTrain,
                                        Matrix const * matTest,
                                        fvector const & theta,
                                        std::vector<float> all_epoch_times) {
      SVMTrainingLogRecord *record = new SVMTrainingLogRecord;
      record->train_rms_loss = SVMTask::rmsErrorLoss(theta, matTrain->blocks_);
      record->test_rms_loss = SVMTask::rmsErrorLoss(theta, matTest->blocks_);
      record->train_fraction_misclassified = SVMTask::fractionMisclassified(theta,matTrain->blocks_);
      record->test_fraction_misclassified = SVMTask::fractionMisclassified(theta,matTest->blocks_);

      record->total_train_time = stats::mean(all_epoch_times) * all_epoch_times.size();
      record->mean_epoch_time = stats::mean(all_epoch_times);
      record->stddev_epoch_time = stats::stddev(all_epoch_times);

      return record;
    }
  };

  void trainSVM(void *svm_trainer_params_void_ptr) {
    SVMTrainerParams *svm_trainer_params = reinterpret_cast<SVMTrainerParams*>(svm_trainer_params_void_ptr);
    Matrix *mat_train = svm_trainer_params->mat_train;
    Matrix *mat_test = svm_trainer_params->mat_test;
    SVMParams* svm_params = svm_trainer_params->svm_params;
    int num_threads = svm_trainer_params->num_threads;
    int starting_worker_id_for_core_assignment = svm_trainer_params->starting_worker_id_for_core_assignment;
    fvector sharedTheta = fvector::GetRandomFVector(mat_train->numColumns_);
    
    std::cout << "Running SVM training with mu = " << svm_params->mu << std::endl;

    // Arguments to the thread pool.
    std::vector<void*> threadStates;
    std::vector<std::function<void(int, void*)>> threadFns;

    // Create the tasks for the thread pool.
    // Roughly allocates work.
    std::vector<std::unique_ptr<DataView>> data_views;

    allocateBlocks(num_threads, mat_train->blocks_, data_views);
    // Create tasks
    auto update_fn = [](int tid, void* state) {
      SVMTask* task = reinterpret_cast<SVMTask*>(state);
      task->execute(tid, nullptr);
    };
    std::vector<std::unique_ptr<SVMTask>> tasks(num_threads);
    for (int i = 0; i < tasks.size(); i++) {
      tasks[i].reset(new SVMTask(data_views[i].release(), &sharedTheta, svm_params));
      threadStates.push_back(tasks[i].get());
      threadFns.push_back(update_fn);
    }

    ThreadPool tp(threadFns, threadStates, starting_worker_id_for_core_assignment);

    tp.begin();
    std::vector<float> all_epoch_times;
    for (int cycle = 0; cycle < FLAGS_num_epochs; cycle++) {
      auto time_start = std::chrono::steady_clock::now();
      tp.cycle();
      auto time_end = std::chrono::steady_clock::now();
      std::chrono::duration<double, std::milli> time_ms = time_end - time_start;
      double elapsedTimeSec = (time_ms.count())/ 1e3;
      all_epoch_times.push_back(elapsedTimeSec);
    }
    tp.stop();

    svm_trainer_params->out_log_record = SVMTrainingLogRecord::Create(
      mat_train, mat_test, sharedTheta, all_epoch_times);
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

    // Grid search: Set up num_param_values SVM parameters in logspace(mu_min, mu_max)
    auto grid_search_start_time = std::chrono::steady_clock::now();
    uint64_t num_param_values = FLAGS_num_param_values;
    std::vector<SVMParams*> params_list;
    params_list.reserve(num_param_values);
    const double mu_min = FLAGS_mu_min;
    const double mu_max = FLAGS_mu_max;
    const double mu_factor = std::pow(mu_max / mu_min, 1.0/(num_param_values-1));
    for (int p = 0; p < num_param_values; ++p) {
      SVMParams* params = DefaultSVMParams<num_t>(mat_train->blocks_);
      DCHECK_EQ(params->degrees.size(), maxColumns(mat_train->blocks_));
      params->mu = mu_min * std::pow(mu_factor, p);
      params_list.push_back(params);
    }

    int starting_worker_id_for_core_assignment = 0;
    std::vector<std::thread*> svm_trainer_threads;
    svm_trainer_threads.reserve(num_param_values);
    std::vector<SVMTrainerParams*> svm_trainer_params_list;
    svm_trainer_params_list.reserve(num_param_values);
    int p = 0;
    while (p < num_param_values) {
      if (starting_worker_id_for_core_assignment + FLAGS_num_threads_per_grid_point > threading::numCores()) {
        std::cout << "Before starting grid point " << p << " waiting for current runs to finish." << std::endl;
        for (auto* thd: svm_trainer_threads) {
          thd->join();
          delete thd;
        }
        svm_trainer_threads.clear();
        starting_worker_id_for_core_assignment = 0;
      }
      SVMParams* svm_params = params_list[p];
      svm_trainer_params_list.push_back(new SVMTrainerParams(mat_train.get(), 
        mat_test.get(), svm_params, FLAGS_num_threads_per_grid_point, 
        starting_worker_id_for_core_assignment));
      
      svm_trainer_threads.push_back(new std::thread(trainSVM, 
        static_cast<void*>(svm_trainer_params_list[p])));
      ++p;
      starting_worker_id_for_core_assignment += FLAGS_num_threads_per_grid_point;
    }
    
    std::cout << "Waiting for all threads to finish" << std::endl; 
    for (auto* thd: svm_trainer_threads) {
      thd->join();
      delete thd;
    }

    auto grid_search_end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> grid_search_time_ms = 
        grid_search_end_time - grid_search_start_time;
    double grid_search_time_sec = (grid_search_time_ms.count())/ 1e3;
    
    svm_trainer_threads.clear();
    std::cout << "Done executing all grid points!" << std::endl;

    std::cout << "\n\n--------------------------------------------------------\n";
    std::cout << "> mu, num_threads_per_grid_point, starting_worker_id, num_epochs, "
              << "train_rms_loss, test_rms_loss, train_fraction_misclassified, "
              << "test_fraction_misclassified, total_train_time, "
              << "mean_epoch_time, stddev_epoch_time\n";
    double total_grid_search_worker_time = 0.0;
    for (const auto* P : svm_trainer_params_list) {
      std::cout << "> " << P->svm_params->mu << ","
                << P->num_threads << ","
                << P->starting_worker_id_for_core_assignment << ","
                << FLAGS_num_epochs << ","
                << P->out_log_record->train_rms_loss << ","
                << P->out_log_record->test_rms_loss << ","
                << P->out_log_record->train_fraction_misclassified << ","
                << P->out_log_record->test_fraction_misclassified << ","
                << P->out_log_record->total_train_time << ","
                << P->out_log_record->mean_epoch_time << ","
                << P->out_log_record->stddev_epoch_time << "\n";
      total_grid_search_worker_time += P->out_log_record->total_train_time;
    }
    std::cout << "--------------------------------------------------------\n";

    std::cout << "\n\n--------------------------------------------------------\n";
    std::cout << "@ num_threads_per_grid_point, max_parallel_grid_points, total_grid_search_worker_time, total_grid_search_time\n";
    std::cout << "@ " << FLAGS_num_threads_per_grid_point << ","
              << threading::numCores() / FLAGS_num_threads_per_grid_point << ","
              << total_grid_search_worker_time << ","
              << grid_search_time_sec << "\n";
    std::cout << "--------------------------------------------------------\n";
  }

  int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    ::gflags::SetUsageMessage(std::string(argv[0]) + " -help");
    ::gflags::SetVersionString("0.0");
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);

    // std::vector<int> affinities = GetIntList(FLAGS_core_affinities);
    // if (affinities[0] != -1) {
    //   threading::setCoreAffinity(affinities[0]);
    // } else {
    //   LOG(INFO) << "Main thread affinitized to core 0";
    //   threading::setCoreAffinity(0);
    // }
    
    runSvmExperiment();

    return 0;
  }
} // namespace obamadb

int main(int argc, char **argv) {
  return obamadb::main(argc, argv);
}
