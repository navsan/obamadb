#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/IO.h"
#include "storage/SparseDataBlock.h"
#include "storage/Task.h"
#include "storage/ThreadPool.h"

#include "storage/tests/StorageTestHelpers.h"

#include <iostream>
#include <string>
#include <memory>
#include <vector>

#include <glog/logging.h>

namespace obamadb {

  DoubleVector getTheta(int dim) {
    DoubleVector shared_theta(dim);
    // initialize to values [-1,1]
    for (unsigned i = 0; i < dim; ++i) {
        shared_theta[i] = (1.0 - fmod((double)rand()/100.0, 2)) / 10.0;
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
    const double error_bound = 0.08; // stop when error falls below.

    std::vector<SparseDataBlock<double>*> blocks = IO::load<double>("/home/marc/workspace/obamaDB/data/RCV1.train.tsv");
    printf("Read in %lu blocks for a total size of %ldmb\n",blocks.size(), ((blocks.size() * kStorageBlockSize) / 1000000));
//    IO::save("/tmp/RCV1.dense.csv", blocks, 10);
//    printf("Saved %d blocks to /tmp\n", 10);

    SVMParams svm_params = DefaultSVMParams<double>(blocks);
    DCHECK_EQ(svm_params.degrees.size(), maxColumns(blocks));
    DoubleVector shared_theta = getTheta(maxColumns(blocks));

    // Create the tasks for the Threadpool.
    // Roughly allocates work.
    std::vector<std::unique_ptr<DataView>> data_views;

    allocateBlocks(num_threads, blocks, data_views);
    // Create tasks
    std::vector<SVMTask*> tasks;
    for (int i = 0; i < num_threads; i++) {
      tasks.push_back(new SVMTask(data_views[i].get(), &shared_theta, svm_params));
    }

    // Create ThreadPool + Workers
    const int tcycles = 1000;
    double last_error = error_bound + 1;
    ThreadPool tp(tasks);
    tp.begin();
    for (int cycle = 0; cycle < tcycles && last_error > error_bound; cycle++) {
      tp.cycle();
      last_error = Task::error(shared_theta, *blocks[0]);
      printf("%-3d: %f\n", cycle, last_error);
    }
    tp.stop();
//
//    const DataBlock& block = *blocks[0];
//    Loader::save("/tmp/sparse.dat", &block);
//
    std::ofstream outfile;
    outfile.open("/tmp/theta.dat", std::ios::binary | std::ios::out);
    for(int i = 0; i < shared_theta.dimension_; i++) {
       outfile << shared_theta.values_[i] << " ";
    }
//
  }


} // namespace obamadb

int main() {
  obamadb::main();
}