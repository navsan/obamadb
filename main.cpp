#include "storage/DataBlock.h"
#include "storage/LinearMath.h"
#include "storage/Loader.h"
#include "storage/ThreadPool.h"

#include "storage/tests/StorageTestHelpers.h"

#include <iostream>
#include <string>
#include <memory>

#include <glog/logging.h>

namespace obamadb {

  int main() {
    const unsigned learning_itrs = 1000;

    const unsigned dimension = 10;
    const unsigned num_examples = 1000000;
    const double elipson = 0.001; // accounts for rounding errors in some checks
    const double radius = 15;
    DoubleVector center_pt1_(dimension);
    DoubleVector center_pt2_(dimension);
    for(unsigned i = 0; i < dimension; ++i) {
      center_pt1_[i] = 5;
      center_pt2_[i] = -5;
    }

    SyntheticDataSet dataset(dimension, num_examples,center_pt1_.values_,center_pt2_.values_, radius, radius);
    std::vector<DataBlock*> data_blocks = dataset.getDataSet();

    DoubleVector shared_theta(dimension);
    for(unsigned i = 0; i < dimension; ++i) {
      shared_theta[i] = -5;
    }
    unsigned itrs_used = 0;

    // Create ThreadPool + Workers.
    {
      ThreadPool threadpool(4);

      for (; itrs_used < learning_itrs; ++itrs_used) {
        for (unsigned block_i = 0; block_i < dimension; ++block_i) {
          DataBlock *block = data_blocks[block_i];
          threadpool.enqueue(
            [&shared_theta, block] {
              std::unique_ptr<DataBlock> a_vals(block->slice(0, block->getWidth() - 1));
              std::unique_ptr<DataBlock> y_vals(block->slice(block->getWidth() - 1, block->getWidth()));
              for (unsigned row = 0; row < block->getNumRows(); ++row) {
                gradientItr(a_vals.get(), y_vals.get(), shared_theta.values_);
              }
            });
        }
        DataBlock *block = data_blocks[rand() % data_blocks.size()];
        std::future<double> recent_error =
          threadpool.enqueue(
            [&shared_theta, block] {
              std::unique_ptr<DataBlock> a_vals(block->slice(0, block->getWidth() - 1));
              std::unique_ptr<DataBlock> y_vals(block->slice(block->getWidth() - 1, block->getWidth()));
              return error(a_vals.get(), y_vals.get(), shared_theta.values_);
            });
        recent_error.wait();
        if (recent_error.get() < 0.05) {
          break;
        }
      }
    }

    for (unsigned i = 0; i < dimension; i++) {
      std::cout << shared_theta[i] << ",";
    }
    std::cout << std::endl;

    DataBlock *block = data_blocks[rand() % data_blocks.size()];
    std::unique_ptr<DataBlock> a_vals(block->slice(0, block->getWidth() - 1));
    std::unique_ptr<DataBlock> y_vals(block->slice(block->getWidth() - 1, block->getWidth()));
    std::cout << error(a_vals.get(), y_vals.get(), shared_theta.values_);

    return 0;
  }
} // namespace obamadb

int main() {
  obamadb::main();
}