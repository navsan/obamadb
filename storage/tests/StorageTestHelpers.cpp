#include "storage/tests/StorageTestHelpers.h"

#include "storage/DataBlock.h"

namespace obamadb {

//  void SynthData::initialize_data() {
//    DenseDataBlock *current_block = new DenseDataBlock(dim_ + 1);
//
//    for (unsigned itr = 0; itr < training_examples_; itr++) {
//      double *current_row = current_block->getRow(current_block->getNumRows());
//      if (rand() % 2) {
//        generatePoint(p1_.values_, current_row, rad1_);
//        current_row[dim_] = 1; // y value.
//      } else {
//        generatePoint(p2_.values_, current_row, rad2_);
//        current_row[dim_] = -1; // y value.
//      }
//      current_block->setSize(current_block->getSize() + current_block->getNumColumns());
//      if (dim_ + 1 > current_block->getRemainingElements()) {
//        blocks_.push_back(current_block);
//        current_block = new DenseDataBlock(dim_ + 1);
//      }
//    }
//
//    if (current_block->getSize() != 0) {
//      blocks_.push_back(current_block);
//    }
//  }
//
//
//  void SynthData::generatePoint(double const *src, double *dst, unsigned radius) {
//    // Create a random vector with a magnitude less than radius.
//    //
//    // src is the vector for which we will write a new vector in dst
//    // whose location is somewhere in radius distance from src
//    double sq_sum = 0;
//    const unsigned MAX_RAND = 100;
//    for (unsigned i = 0; i < dim_; ++i) {
//      dst[i] = rand() % MAX_RAND;
//      sq_sum += (dst[i] * dst[i]);
//    }
//    double new_magn = fmod((double)std::rand() / 1000, radius);
//    double magn_factor = new_magn / std::sqrt(sq_sum);
//    for (unsigned i = 0; i < dim_; ++i) {
//      dst[i] *= magn_factor;
//      dst[i] += src[i];
//    }
//  }
//
//  SynthData::~SynthData() {
//    for ( int i = 0 ; i < blocks_.size(); i ++) {
//      delete blocks_[i];
//    }
//  }
//
//  SynthDataParams DefaultSynthDataParams() {
//    int const dims = 10;
//    SynthDataParams params(dims, 100000, 10, 10);
//    for(int i = 0; i < dims; i++) {
//      params.p1[i] = i % 2 == 0 ? -5 : 5;
//      params.p2[i] = i % 2 == 0 ? 5 : -5;
//    }
//    return params;
//  }
} // namespace obamadb