#include "gtest/gtest.h"
#include "storage/DataBlock.h"
#include "storage/exvector.h"
#include "storage/IO.h"
#include "storage/SparseDataBlock.h"
#include "storage/Utils.h"

#include "storage/tests/StorageTestHelpers.h"

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

  TEST(SparseDataBlockTest, TestCreateSparseProjection) {
    double m = 47000, n = 20; // m corresponds to the dimension of the original matrix.
    std::unique_ptr<SparseDataBlock<signed char>> pdb(GetRandomProjectionMatrix(m,n));
    ASSERT_EQ(n, pdb->getNumRows());
    ASSERT_EQ(m, pdb->getNumColumns());
    int counts[3] = {0,0,0};
    svector<signed char> row;
    row.setMemory(0, nullptr); // cheap way to clear ownership
    for (int i = 0; i < pdb->getNumRows(); i++) {
      pdb->getRowVectorFast(i, &row);
      for (int j = 0; j < row.num_elements_; j++) {
        counts[1 + row.values_[j]]++;
      }
    }
    counts[1] = (m * n) - (counts[0] + counts[2]);
    const double tolerance = 0.1; // falls within 10% of the expected value.
    const double freq_ones = (1.0/(2.0*sqrt(m))) * m * n;
    const double freq_zero = m * n - 2.0 * freq_ones;
    EXPECT_TRUE(counts[0] > freq_ones * (1.0 - tolerance) && counts[0] < freq_ones * (1.0 + tolerance));
    EXPECT_TRUE(counts[2] > freq_ones * (1.0 - tolerance) && counts[2] < freq_ones * (1.0 + tolerance));
    EXPECT_TRUE(counts[1] > freq_zero * (1.0 - tolerance) && counts[1] < freq_zero * (1.0 + tolerance));
    // TODO could also test for variance in the distribution.
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