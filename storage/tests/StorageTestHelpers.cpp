#include "storage/tests/StorageTestHelpers.h"

#include "storage/DataBlock.h"

namespace obamadb {

  void SyntheticDataSet::initialize_data() {
    DataBlock *current_block = new DataBlock(dim_ + 1);

    for (unsigned itr = 0; itr < training_examples_; itr++) {
      double *current_row = current_block->getRow(current_block->getNumRows());
      if (rand() % 2) {
        generatePoint(pt_1_, current_row, rad_1_);
        current_row[dim_] = 1; // y value.
      } else {
        generatePoint(pt_2_, current_row, rad_2_);
        current_row[dim_] = 0; // y value.
      }
      current_block->setSize(current_block->getSize() + current_block->getWidth());
      if (dim_ + 1 > current_block->getRemainingElements()) {
        blocks_.push_back(current_block);
        current_block = new DataBlock(dim_ + 1);
      }
    }

    if (current_block->getSize() != 0) {
      blocks_.push_back(current_block);
    }
  }


  void SyntheticDataSet::generatePoint(double const *src, double *dst, unsigned radius) {
    // Create a random vector with a magnitude less than radius.
    double sq_sum = 0;
    const unsigned MAX_RAND = 100;
    for (unsigned i = 0; i < dim_; ++i) {
      dst[i] = (rand() % MAX_RAND) - (MAX_RAND / 2);
      sq_sum += (dst[i] * dst[i]);
    }
    double new_magn = std::rand() % radius;
    if (0 == new_magn) {
      new_magn = 1;
    }
    double magn_factor = new_magn / std::sqrt(sq_sum);
    for (unsigned i = 0; i < dim_; ++i) {
      dst[i] *= magn_factor;
      dst[i] += src[i];
    }
  }

  SyntheticDataSet::~SyntheticDataSet() {
    for (auto block_itr : blocks_) {
      delete block_itr;
    }
  }
} // namespace obamadb