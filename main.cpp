#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/IO.h"
#include <storage/Matrix.h>
#include "storage/MLTask.h"
#include "storage/SparseDataBlock.h"
#include "storage/StorageConstants.h"
#include "storage/tests/StorageTestHelpers.h"
#include "storage/ThreadPool.h"

#include <iostream>
#include <string>
#include <memory>
#include <vector>

#include <glog/logging.h>


namespace obamadb {

  fvector getTheta(int dim) {
    fvector shared_theta(dim);
    // initialize to values [-1,1]
    for (unsigned i = 0; i < dim; ++i) {
        shared_theta[i] = static_cast<float_t>((1.0 - fmod((double)rand()/100.0, 2)) / 10.0);
    }
    return shared_theta;
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

  void printMatrixStats(Matrix const * mat) {
    printf("Matrix: %lu training blocks for a total size of %ldmb with %d examples with %f sparsity\n",
           mat->blocks_.size(),
           (long) (mat->sizeBytes() / 1e6),
           mat->numRows_,
           mat->getSparsity());
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
    printf("%-3d: %.3f, %.4f, %.2f, %.4f, %.2f, %.4f\n", itr, timeTrain, trainFractionMisclassified, trainRmsLoss, testFractionMisclassified, testRmsLoss, dTheta);
  }

  void trainSVM(Matrix *mat_train, Matrix *mat_test, int num_threads) {
    SVMParams* svm_params = DefaultSVMParams<float_t>(mat_train->blocks_);
    DCHECK_EQ(svm_params->degrees.size(), maxColumns(mat_train->blocks_));
    fvector sharedTheta = getTheta(mat_train->numColumns_);

    // Create the tasks for the Threadpool.
    // Roughly allocates work.
    std::vector<std::unique_ptr<DataView>> data_views;

    allocateBlocks(num_threads, mat_train->blocks_, data_views);
    // Create tasks
    std::vector<std::unique_ptr<SVMTask>> tasks(num_threads);
    std::vector<void*> threadStates;
    for (int i = 0; i < tasks.size(); i++) {
      tasks[i].reset(new SVMTask(data_views[i].release(), &sharedTheta, svm_params));
      threadStates.push_back(tasks[i].get());
    }

    auto update_fn = [](int tid, void* state) {
      SVMTask* task = reinterpret_cast<SVMTask*>(state);
      task->execute(tid, nullptr);
    };

    // Create ThreadPool + Workers
    const int totalCycles = 20;
    ThreadPool tp(update_fn, threadStates);
    tp.begin();
    printf("i : train_time, train_fraction_misclassified, train_RMS_loss, test_fraction_misclassified, test_RMS_loss, dtheta\n");
    printSVMItrStats(mat_train, mat_test, sharedTheta, sharedTheta, -1, 0);
    float totalTrainTime = 0.0;
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
    printf(">%d,%d,%f,%f\n",mat_test->numColumns_, num_threads, avgTrainTime, finalFractionMispredicted);
  }

  void doCompression(Matrix const * train,
                     Matrix const * test,
                     std::unique_ptr<Matrix>& compressedTrain,
                     std::unique_ptr<Matrix>& compressedTest,
                     int compressionConst) {

    std::pair<Matrix*, SparseDataBlock<signed char>*> compressResult;
    PRINT_TIMING_MSG("Compress Training Mat", { compressResult = train->randomProjectionsCompress(compressionConst);} );
    compressedTrain.reset(compressResult.first);
    printMatrixStats(compressedTrain.get());
    //PRINT_TIMING_MSG("Save Compress Training Mat", {IO::save("/tmp/matB_train.dat", *compressedTrain);});
    std::vector<SparseDataBlock<signed char>*> blocksR = { compressResult.second };
    //PRINT_TIMING_MSG("Save R Mat", {IO::save<signed char>("/tmp/matR.dat", blocksR, 1);});
    std::unique_ptr<SparseDataBlock<signed char>> blockR(compressResult.second);
    PRINT_TIMING_MSG("Compress Test Mat", { compressedTest.reset(test->randomProjectionsCompress(blockR.get(), compressionConst)); });
    //PRINT_TIMING_MSG("Save Test Mat", {IO::save("/tmp/matB_test.dat", *compressedTest);});
  }

  int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    CHECK_EQ(5, argc) << "usage: " << argv[0] << " [training data] [testing data] [compression constant] [num threads]";

    std::string train_fp(argv[1]);
    std::string test_fp(argv[2]);
    const int kCompressionConst = std::stoi(argv[3]);
    CHECK_LE(0, kCompressionConst);
    const int numThreads = std::stoi(argv[4]);

    printf("Nthreads: %d, compressionConst: %d\n", numThreads, kCompressionConst);

    std::unique_ptr<Matrix> mat_train;
    std::unique_ptr<Matrix> mat_test;

    printf("Reading input files...\n");
    printf("Loading: %s\n", train_fp.c_str());
    PRINT_TIMING({mat_train.reset(IO::load(train_fp));});
    printMatrixStats(mat_train.get());

    printf("Loading: %s\n", test_fp.c_str());
    PRINT_TIMING({mat_test.reset(IO::load(test_fp));});
    printMatrixStats(mat_test.get());

    DCHECK_EQ(mat_test->numColumns_, mat_train->numColumns_);

    if (kCompressionConst == 0) {
      printf("No compression will be applied\n");
      trainSVM(mat_train.get(), mat_test.get(), numThreads);
    } else {
      printf("Compression will be applied\n");
      std::unique_ptr<Matrix> ctrain;
      std::unique_ptr<Matrix> ctest;
      doCompression(mat_train.get(), mat_test.get(), ctrain, ctest, kCompressionConst);
      printMatrixStats(ctrain.get());
      printMatrixStats(ctest.get());

      trainSVM(ctrain.get(), ctest.get(), numThreads);
    }

    return 0;
  }
} // namespace obamadb

int main(int argc, char **argv) {
  return obamadb::main(argc, argv);
}