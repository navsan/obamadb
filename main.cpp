#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/IO.h"
#include "storage/Matrix.h"
#include "storage/MLTask.h"
#include "storage/tests/StorageTestHelpers.h"

#include <unistd.h>

namespace obamadb {

  /**
   * Parses and holds user's command line arguments.
   */
  struct TestParams {
    TestParams(const std::string &train_fp,
               const std::string &test_fp,
               int compressionConst,
               int numThreads,
               int measureConvergence)
      : train_fp(train_fp),
        test_fp(test_fp),
        compressionConst(compressionConst),
        numThreads(numThreads),
        measureConvergence(measureConvergence){}

    TestParams(TestParams const & other) {
      train_fp = other.train_fp;
      test_fp = other.test_fp;
      compressionConst = other.compressionConst;
      numThreads = other.numThreads;
      measureConvergence = other.measureConvergence;
    }

    static TestParams ReadParams(int argc, char** argv);

    std::string train_fp;
    std::string test_fp;
    int compressionConst;
    int numThreads;
    int measureConvergence;
  };

  TestParams TestParams::ReadParams(int argc, char** argv) {
    CHECK_LE(5, argc) << "usage: " << argv[0]
                      << " [training data][testing data][compression constant][num threads][(optional)measureConvergence]";

    TestParams params(argv[1],argv[2],std::stoi(argv[3]),std::stoi(argv[4]),0);
    if (argc == 6) {
      params.measureConvergence = std::stoi(argv[5]);
    }
    CHECK_LE(0, params.compressionConst);
    return params;
  }

  /**
   * Used to snoop on the inter-epoch values of the model.
   */
  class ConvergenceObserver {
  public:
    ConvergenceObserver(fvector const * const sharedTheta) :
      model_ref_(sharedTheta),
      observedTimes_(),
      observedModels_(),
      threadPool_(nullptr) {}

    void record() {
      auto current_time = std::chrono::steady_clock::now();
      observedTimes_.push_back(current_time.time_since_epoch().count());
      observedModels_.push_back(*model_ref_);
    }

    int size() {
      CHECK_EQ(observedModels_.size(), observedTimes_.size());
      return observedModels_.size();
    }

    fvector const * const model_ref_;
    std::vector<std::uint64_t> observedTimes_;
    std::vector<fvector> observedModels_;
    ThreadPool * threadPool_;
  };

  /**
   * The function which a thread will call once per epoch to record the
   * model's state change over time.
   * @param tid Thread id. Not used.
   * @param observerState Pair<remaining epochs to observe, observer>.
   */
  void observerThreadFn(int tid, void* observerState) {
    std::pair<int*, ConvergenceObserver*> *statePair =
      reinterpret_cast<std::pair<int*, ConvergenceObserver*>*>(observerState);
    int *remaining_measurement_epochs = statePair->first;
    if (*remaining_measurement_epochs <= 0) {
      return;
    }
    // loop until all of the other threads have completed. We don't know this exactly, so
    // we'll instead loop for some arbitrary number of times.
    // TODO: get access to barrier information to tell when all threads have completed their cycle.
    ConvergenceObserver* observer = statePair->second;
    int const total_workers = observer->threadPool_->getNumWorkers();
    while (total_workers > observer->threadPool_->getWaiterCount()) {
      observer->record();
      usleep(1000);
    }
    (*remaining_measurement_epochs)--;
  }

  /**
   * Allocates Datablocks to DataViews. Dataviews will then be given to threads in the form of tasks.
   * @param num_threads How many dataviews to allocate and distrubute amongst.
   * @param data_blocks The set of training data.
   * @return Vector of Dataviews.
   */
  template<class T>
  void allocateBlocks(
    const int num_threads,
    const std::vector<SparseDataBlock<T> *> &data_blocks,
    std::vector<std::unique_ptr<DataView>>& views) {
    CHECK(views.size() == 0) << "Only accepts empty view vectors";

    for (int i = 0; i < data_blocks.size(); i ++) {
      if (i < num_threads) {
        views.push_back(std::unique_ptr<DataView>(new DataView()));
      }
      SparseDataBlock<T> const *dbptr = data_blocks[i];
      views[i % num_threads]->appendBlock(dbptr);
    }
  }

  void printSVMItrStats(Matrix const * matTrain,
                        Matrix const * matTest,
                        fvector const & theta,
                        fvector const & oldTheta,
                        int itr,
                        float timeTrain) {
    float_t trainRmsLoss = ml::rmsErrorLoss(theta, matTrain->blocks_);
    float_t testRmsLoss = ml::rmsErrorLoss(theta, matTest->blocks_);
    float_t trainFractionMisclassified = ml::fractionMisclassified(theta,matTrain->blocks_);
    float_t testFractionMisclassified = ml::fractionMisclassified(theta,matTest->blocks_);
    float_t dTheta = 0;
    for (int i = 0; i < oldTheta.dimension_; i++) {
      dTheta += std::abs(oldTheta.values_[i] - theta.values_[i]);
    }
    printf("%-3d: %.3f, %.4f, %.2f, %.4f, %.2f, %.4f\n", itr, timeTrain, trainFractionMisclassified,
           trainRmsLoss, testFractionMisclassified, testRmsLoss, dTheta);
  }

  void trainSVM(Matrix *mat_train, Matrix *mat_test, TestParams const & params) {
    SVMParams* svm_params = DefaultSVMParams<float_t>(mat_train->blocks_);
    DCHECK_EQ(svm_params->degrees.size(), maxColumns(mat_train->blocks_));
    fvector sharedTheta = fvector::GetRandomFVector(mat_train->numColumns_);

    // arguments to the threadpool.
    std::vector<void*> threadStates;
    std::vector<std::function<void(int, void*)>> threadFns;

    // create an observer thread, if the params specify it
    int observationEpoch = params.measureConvergence;
    ConvergenceObserver observer(&sharedTheta);
    std::unique_ptr<std::pair<int*, ConvergenceObserver*>> observerInfo(
      new std::pair<int*, ConvergenceObserver*>(&observationEpoch, &observer));
    if (params.measureConvergence > 0) {
      auto observer_fn = observerThreadFn;
      threadFns.push_back(observer_fn);
      threadStates.push_back(observerInfo.get());
    }

    // Create the tasks for the Threadpool.
    // Roughly allocates work.
    std::vector<std::unique_ptr<DataView>> data_views;

    allocateBlocks(params.numThreads, mat_train->blocks_, data_views);
    // Create tasks
    auto update_fn = [](int tid, void* state) {
      SVMTask* task = reinterpret_cast<SVMTask*>(state);
      task->execute(tid, nullptr);
    };
    std::vector<std::unique_ptr<SVMTask>> tasks(params.numThreads);
    for (int i = 0; i < tasks.size(); i++) {
      tasks[i].reset(new SVMTask(data_views[i].release(), &sharedTheta, svm_params));
      threadStates.push_back(tasks[i].get());
      threadFns.push_back(update_fn);
    }

    // Create ThreadPool + Workers
    const int totalCycles = 2;
    ThreadPool tp(threadFns, threadStates);
    observer.threadPool_ = &tp;
    tp.begin();
    printf("i : train_time, train_fraction_misclassified, train_RMS_loss, test_fraction_misclassified, test_RMS_loss, dtheta\n");
    printSVMItrStats(mat_train, mat_test, sharedTheta, sharedTheta, -1, 0);
    float totalTrainTime = 0.0;
    int observationCycles = params.measureConvergence;
    for (int cycle = 0; cycle < totalCycles; cycle++) {
      fvector last_theta(sharedTheta);

      auto time_start = std::chrono::steady_clock::now();
      tp.cycle();
      auto time_end = std::chrono::steady_clock::now();
      std::chrono::duration<double, std::milli> time_ms = time_end - time_start;
      double elapsedTimeSec = (time_ms.count())/ 1e3;
      totalTrainTime += elapsedTimeSec;
      printSVMItrStats(mat_train, mat_test, sharedTheta, last_theta, cycle, elapsedTimeSec);
    }
    tp.stop();

    float avgTrainTime = totalTrainTime / totalCycles;
    float finalFractionMispredicted = ml::fractionMisclassified(sharedTheta, mat_test->blocks_);
    printf("num_cols, num_threads, avg_train_time, frac_mispredicted, nnz_train\n");
    printf(">%d,%d,%f,%f,%d\n",mat_test->numColumns_, params.numThreads, avgTrainTime, finalFractionMispredicted, mat_train->getNNZ());

    // Store the observer information:
    if (observationCycles > 0) {
      printf("Convergence Info: (%d measures)\n", observer.size());

      for (int obs_idx = 0; obs_idx < observer.size(); obs_idx++) {
        std::uint64_t timeObs = observer.observedTimes_[obs_idx];
        fvector const & thetaObs = observer.observedModels_[obs_idx];
        float_t testLoss = ml::rmsErrorLoss(thetaObs, mat_test->blocks_);
        float_t testFractionMisclassified = ml::fractionMisclassified(thetaObs, mat_test->blocks_);
        printf("%d,%llu,%.4f,%.4f\n", params.numThreads, timeObs, testLoss, testFractionMisclassified);
      }
      observationCycles--;
    }
  }

  void doCompression(Matrix const * train,
                     Matrix const * test,
                     std::unique_ptr<Matrix>& compressedTrain,
                     std::unique_ptr<Matrix>& compressedTest,
                     int compressionConst) {
    std::pair<Matrix*, SparseDataBlock<signed char>*> compressResult;
    PRINT_TIMING_MSG("Compress Training Mat", { compressResult = train->randomProjectionsCompress(compressionConst);} );
    compressedTrain.reset(compressResult.first);
    std::cout << *compressedTrain << std::endl;
    //PRINT_TIMING_MSG("Save Compress Training Mat", {IO::save("/tmp/matB_train.dat", *compressedTrain);});
    std::vector<SparseDataBlock<signed char>*> blocksR = { compressResult.second };
    //PRINT_TIMING_MSG("Save R Mat", {IO::save<signed char>("/tmp/matR.dat", blocksR, 1);});
    std::unique_ptr<SparseDataBlock<signed char>> blockR(compressResult.second);
    PRINT_TIMING_MSG("Compress Test Mat", { compressedTest.reset(test->randomProjectionsCompress(blockR.get(), compressionConst)); });
    //PRINT_TIMING_MSG("Save Test Mat", {IO::save("/tmp/matB_test.dat", *compressedTest);});
  }

  int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    TestParams params = TestParams::ReadParams(argc, argv);

    printf("NumThreads: %d, CompressionConst: %d\n", params.numThreads, params.compressionConst);

    std::unique_ptr<Matrix> mat_train;
    std::unique_ptr<Matrix> mat_test;

    printf("Reading input files...\n");
    printf("Loading: %s\n", params.train_fp.c_str());
    PRINT_TIMING({mat_train.reset(IO::load(params.train_fp));});
    std::cout << *mat_train << std::endl;

    printf("Loading: %s\n", params.test_fp.c_str());
    PRINT_TIMING({mat_test.reset(IO::load(params.test_fp));});
    std::cout << *mat_test.get() << std::endl;

    CHECK_EQ(mat_test->numColumns_, mat_train->numColumns_)
      << "Train and Test matrices had differing number of features.";

    if (params.compressionConst == 0) {
      printf("No compression will be applied\n");
      trainSVM(mat_train.get(), mat_test.get(), params);
    } else {
      printf("Compression will be applied\n");
      std::unique_ptr<Matrix> ctrain;
      std::unique_ptr<Matrix> ctest;
      doCompression(mat_train.get(), mat_test.get(), ctrain, ctest, params.compressionConst);
      std::cout << *ctrain.get() << std::endl;
      std::cout << *ctest.get() << std::endl;

      trainSVM(ctrain.get(), ctest.get(), params);
    }

    return 0;
  }
} // namespace obamadb

int main(int argc, char **argv) {
  return obamadb::main(argc, argv);
}