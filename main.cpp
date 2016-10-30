#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/IO.h"
#include "storage/SparseDataBlock.h"
#include "storage/Task.h"
#include "storage/ThreadPool.h"
#include "storage/StorageConstants.h"

#include "storage/tests/StorageTestHelpers.h"

#include <ctime>

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

  int main() {
    const int num_threads = 4;
    const float_t error_bound = 0.008; // stop when error falls below.

    printf("Reading input files...\n");
    std::vector<SparseDataBlock<float_t>*> blocks_train = IO::load<float_t>("/home/marc/workspace/obamaDB/data/RCV1.train.tsv");
 //   std::vector<SparseDataBlock<float_t>*> blocks_train = IO::load<float_t>("/home/marc/workspace/obamaDB/data/RCV1.test.tsv");

    printf("Read in %lu training blocks for a total size of %ldmb\n",blocks_train.size(), ((blocks_train.size() * kStorageBlockSize) / 1000000));

//    IO::save("/tmp/rcv_block.csv", blocks_train, 1);
//    return 0;

    std::vector<SparseDataBlock<float_t>*> blocks_test = IO::load<float_t>("/home/marc/workspace/obamaDB/data/RCV1.test.tsv");
    printf("Read in %lu test blocks for a total size of %ldmb\n",blocks_test.size(), ((blocks_test.size() * kStorageBlockSize) / 1000000));

    SVMParams svm_params = DefaultSVMParams<float_t>(blocks_train);
    DCHECK_EQ(svm_params.degrees.size(), maxColumns(blocks_train));
    f_vector shared_theta = getTheta(maxColumns(blocks_train));

    // Create the tasks for the Threadpool.
    // Roughly allocates work.
    std::vector<std::unique_ptr<DataView>> data_views;

    allocateBlocks(num_threads, blocks_train, data_views);
    // Create tasks
    std::vector<SVMTask*> tasks;
    for (int i = 0; i < num_threads; i++) {
      tasks.push_back(new SVMTask(data_views[i].get(), &shared_theta, svm_params));
    }

    // Create ThreadPool + Workers
    const int tcycles = 20;
    float_t last_error_train = Task::error(shared_theta, blocks_train);
    float_t last_error_test = Task::error(shared_theta, blocks_test);
    printf("itr: train {fraction misclassified, RMSE, time to train}, test {fraction misclassified, RMSE}, Dtheta \n");
    printf("%-3d: train {%f, %f}, test {%f, %f], Dtheta: %d\n", -1, last_error_train, std::sqrt(last_error_train), last_error_test, std::sqrt(last_error_test), 0);
    ThreadPool tp(tasks);
    tp.begin();
    for (int cycle = 0;
         cycle < tcycles && last_error_test > error_bound;
         cycle++) {
      f_vector last_theta(shared_theta);

      clock_t begin = clock();
      tp.cycle();
      clock_t end = clock();
      float_t elapsed_secs = float_t(end - begin) / CLOCKS_PER_SEC;

      last_error_train = Task::error(shared_theta, blocks_train);
      last_error_test = Task::error(shared_theta, blocks_test);

      float_t diff = 0;
      for (int i = 0; i < last_theta.dimension_; i++) {
        diff += std::abs(last_theta.values_[i] - shared_theta.values_[i]);
      }

      printf("%-3d: train {%f, %f, %f}, test {%f, %f], Dtheta: %f\n", cycle, last_error_train, std::sqrt(last_error_train), elapsed_secs, last_error_test, std::sqrt(last_error_test), diff);
    }
    tp.stop();


//    const int blocks_to_save = 2;
//    IO::save("/tmp/RCV1.dense.csv", blocks_train, blocks_to_save);
//    printf("Saved %d blocks to /tmp\n", blocks_to_save);

    std::ofstream outfile;
    outfile.open("/tmp/theta.dat", std::ios::binary | std::ios::out);
    for(int i = 0; i < shared_theta.dimension_; i++) {
       outfile << shared_theta.values_[i] << " ";
    }
    printf("Saved model to /tmp/theta.dat\n");


  }


} // namespace obamadb

int main() {
  obamadb::main();
}