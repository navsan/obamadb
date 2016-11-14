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

  f_vector getTheta(int dim) {
    f_vector shared_theta(dim);
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

  void train_svm(Matrix* mat_train, Matrix* mat_test, int num_threads) {
    SVMParams* svm_params = DefaultSVMParams<float_t>(mat_train->blocks_);
    DCHECK_EQ(svm_params->degrees.size(), maxColumns(mat_train->blocks_));
    f_vector shared_theta = getTheta(mat_train->numColumns_);

    // Create the tasks for the Threadpool.
    // Roughly allocates work.
    std::vector<std::unique_ptr<DataView>> data_views;

    allocateBlocks(num_threads, mat_train->blocks_, data_views);
    // Create tasks
    std::vector<std::unique_ptr<SVMTask>> tasks(num_threads);
    std::vector<void*> threadStates;
    for (int i = 0; i < tasks.size(); i++) {
      tasks[i].reset(new SVMTask(data_views[i].release(), &shared_theta, svm_params));
      threadStates.push_back(tasks[i].get());
    }

    auto update_fn = [](int tid, void* state) {
      SVMTask * task = reinterpret_cast<SVMTask*>(state);
      task->execute(tid, nullptr);
    };

    // Create ThreadPool + Workers
    const int tcycles = 20;
    float_t train_rmse = ml::rmsError(shared_theta, mat_train->blocks_);
    float_t test_rmse = ml::rmsError(shared_theta, mat_test->blocks_);
    printf("itr: train {fraction misclassified, RMSE, time to train}, test {fraction misclassified, RMSE}, Dtheta \n");
    printf("%-3d: train {%f, %f}, test {%f, %f], Dtheta: %d\n", -1, train_rmse, std::sqrt(train_rmse), test_rmse, std::sqrt(test_rmse), 0);
    ThreadPool tp(update_fn, threadStates);
    tp.begin();
    for (int cycle = 0;
         cycle < tcycles;
         cycle++) {
      f_vector last_theta(shared_theta);

      auto time_start = std::chrono::steady_clock::now();
      tp.cycle();
      auto time_end = std::chrono::steady_clock::now();
      std::chrono::duration<double, std::milli> time_ms = time_end - time_start;

      double elapsed_time_s = (time_ms.count())/ 1e3;

      double train_fraction_error = 0; // Task::fraction_error(shared_theta, mat_train->blocks_);
      train_rmse = 0; // Task::rms_error_loss(shared_theta, mat_train->blocks_);

      double test_fraction_error = ml::fractionMisclassified(shared_theta,  mat_test->blocks_);
      test_rmse = ml::rmsErrorLoss(shared_theta,  mat_test->blocks_);

      float_t diff = 0;
      for (int i = 0; i < last_theta.dimension_; i++) {
        diff += std::abs(last_theta.values_[i] - shared_theta.values_[i]);
      }

      printf("%-3d: train {%f, %f, %f}, test {%f, %f}, dtheta: %f\n", cycle, train_fraction_error, train_rmse, elapsed_time_s, test_fraction_error, test_rmse, diff);
    }
    tp.stop();
  }

  void doCompression(Matrix const * train,
                     Matrix const * test,
                     std::unique_ptr<Matrix>& compressedTrain,
                     std::unique_ptr<Matrix>& compressedTest) {
    const int compressionConst = 500;
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
    CHECK_EQ(3, argc) << "usage: " << argv[0] << " [training data] [testing data]";

    std::string train_fp(argv[1]);
    std::string test_fp(argv[2]);

    const int num_threads = 40;
    const float_t error_bound = 0.008; // stop when error falls below.
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

    //train_svm(mat_train.get(), mat_test.get(), num_threads);

    std::unique_ptr<Matrix> ctrain;
    std::unique_ptr<Matrix> ctest;
    doCompression(mat_train.get(), mat_test.get(), ctrain, ctest);

    train_svm(ctrain.get(), ctest.get(), num_threads);

    return 0;
//    mat_test.reset(compress_res.first);
//    mat_test.reset(mat_test->randomProjectionsCompress(compress_res.second, compressionConst));
//    delete compress_res.second;
//
//    printMatrixStats(mat_train.get());
//    printMatrixStats(mat_test.get());

//    std::ofstream outfile;
//    outfile.open("/tmp/theta.dat", std::ios::binary | std::ios::out);
//    for(int i = 0; i < shared_theta.dimension_; i++) {
//       outfile << shared_theta.values_[i] << " ";
//    }
//    printf("Saved model to /tmp/theta.dat\n");
  }
} // namespace obamadb

int main(int argc, char **argv) {
  return obamadb::main(argc, argv);
}