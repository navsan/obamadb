#include "gtest/gtest.h"

#include "storage/DataBlock.h"
#include "storage/exvector.h"
#include "storage/IO.h"
#include "storage/SparseDataBlock.h"
#include "storage/Utils.h"

#include <cstdlib>
#include <memory>

namespace obamadb {

  TEST(SparseDataBlockTest, TestLoadSparse) {
    std::vector<SparseDataBlock<num_t>*> blocks = IO::loadBlocks<num_t>("sparse.dat");
    ASSERT_EQ(1, blocks.size());
    std::unique_ptr<SparseDataBlock<num_t>> block(dynamic_cast<SparseDataBlock<num_t>*>(blocks[0]));

    svector<num_t> r1;
    block->getRowVector(0, &r1);
    EXPECT_EQ(-1, *r1.getClassification());
    EXPECT_EQ(3, r1.numElements());
    EXPECT_EQ(10, r1.index_[0]);
  }

  TEST(SparseDataBlockTest, TestRandomSparseDataBlock) {
    int ncolumns = 1000;
    int blockSizeMb = 10;
    double sparsity = 0.999;
    std::unique_ptr<SparseDataBlock<num_t>> sparseBlock(GetRandomSparseDataBlock(blockSizeMb * 1e6, ncolumns, sparsity));
    EXPECT_LT(ncolumns * 0.9, sparseBlock->num_columns_);
    EXPECT_EQ(blockSizeMb * 1e6, sparseBlock->block_size_bytes_);
    // the low bound is calculated by (totalSizeBytes / (floats per column + size of header + size of classification) * 0.9
    double blockRowsLowBound = (((double)blockSizeMb * 1e6) / ((1.0 - sparsity) * ncolumns * sizeof(num_t) + (sizeof(num_t)*4) )) * 0.9;
    EXPECT_LT(blockRowsLowBound, sparseBlock->num_rows_);
    std::vector<int> columnCounts(sparseBlock->num_columns_);
    svector<num_t> rowView(0, nullptr);
    int numPositive = 0;
    for (int row = 0; row < sparseBlock->num_rows_; row++) {
      sparseBlock->getRowVectorFast(row, &rowView);
      ASSERT_GT((1.0 - sparsity) * ncolumns + 1, rowView.num_elements_);
      for (int i = 0; i < rowView.num_elements_; i++) {
        columnCounts[rowView.index_[i]] += 1;
        if (*rowView.class_ == -1) {
          if (rowView.index_[i] % 2 == 1) {
            EXPECT_LE(0, rowView.values_[i]);
          } else {
            EXPECT_GE(0, rowView.values_[i]);
          }
        } else {
          numPositive++;
          ASSERT_EQ(1, *rowView.class_);
          if (rowView.index_[i] % 2 == 0) {
            EXPECT_LE(0, rowView.values_[i]);
          } else {
            EXPECT_GE(0, rowView.values_[i]);
          }
        }
      }
    }

    // TODO: check that the distribution of chosen indices weren't overly skewed.
    // The verification I did was to put this in octave and make sure stdev was low, no outliers
    // for (int i = 0; i < columnCounts.size(); i++) {
    //   printf("%d\n", columnCounts[i]);
    // }

    EXPECT_GE(sparseBlock->num_rows_ * 0.1, std::abs( (int)(sparseBlock->num_rows_ / 2) - numPositive));
  }
}