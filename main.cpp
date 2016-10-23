#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/LinearMath.h"
#include "storage/Loader.h"
#include "storage/Task.h"
#include "storage/ThreadPool.h"

#include "storage/tests/StorageTestHelpers.h"

#include <iostream>
#include <string>
#include <memory>

#include <glog/logging.h>

namespace obamadb {


  DoubleVector makeTheta(int dim) {
    DoubleVector shared_theta(dim);
    // initialize to values [-1,1]
    for (unsigned i = 0; i < dim; ++i) {
        shared_theta[i] = (1.0 - fmod((double)rand()/100.0, 2)) / 10.0;
    }
    return shared_theta;
  }

  /**
   * Allocates Datablocks to DataViews. Dataviews will then be given to threads in the form of tasks.
   * @param num_threads
   * @param data_blocks
   * @return Vector of Dataviews.
   */
  void allocateBlocks(
    const int num_threads,
    const std::vector<DenseDataBlock *> &data_blocks,
    std::vector<std::unique_ptr<DataView>>& views) {
    CHECK(views.size() == 0) << "Only accepts empty view vectors";

    for (int i = 0; i < data_blocks.size(); i ++) {
      if (i < num_threads) {
        views.push_back(std::unique_ptr<DataView>(new DataView()));
      }
      DenseDataBlock const *dbptr = data_blocks[i];
      views[i % num_threads]->appendBlock(dbptr);
    }
  }

  int main() {
    const int num_threads = 4;
    const double error_bound = 0.08; // stop when error falls below.

    SynthDataParams dataset_params = DefaultSynthDataParams();
    SynthData dataset(dataset_params);
    std::vector<DenseDataBlock *> data_blocks = dataset.getDataSet();
    printf("Dataset size: %ldmb\n",((data_blocks.size() * kStorageBlockSize) / 1000000));

    SVMParams svm_params = DefaultSVMParams(data_blocks);
    DoubleVector shared_theta = makeTheta(dataset_params.dim);

    // Create the tasks for the Threadpool.
    // Roughly allocates work.
    std::vector<std::unique_ptr<DataView>> data_views;
    allocateBlocks(num_threads, data_blocks, data_views);

    // Create tasks
    std::vector<Task*> tasks;
    for (int i = 0; i < num_threads; i++) {
      tasks.push_back(new LinearRegressionTask(data_views[i].get(), &shared_theta, dataset_params.dim));
      //tasks.push_back(new SVMTask(data_views[i].get(), &shared_theta, svm_params));
    }

    // Create ThreadPool + Workers
    const int tcycles = 20;
    double last_error = error_bound + 1;
    ThreadPool tp(tasks);
    tp.begin();
    for (int cycle = 0; cycle < tcycles && last_error > error_bound; cycle++) {
      tp.cycle();
      last_error = Task::error(shared_theta, *data_blocks[rand() % data_blocks.size()]);
      printf("%-3d: %f\n", cycle, last_error);
    }
    tp.stop();

    const DenseDataBlock& block = *data_blocks[rand() % data_blocks.size()];
    Loader::save("/tmp/obamadb.out", block);
    printf("Theta:\n[");
    for(int i = 0; i < shared_theta.dimension_; i++) {
      printf("%f ", shared_theta[i]);
    }
    printf("]\n");
    printf("Final error: %f\n", Task::error(shared_theta, block));
    return 0;
  }

} // namespace obamadb

int main() {
  obamadb::main();
}