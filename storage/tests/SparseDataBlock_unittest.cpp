#include "gtest/gtest.h"
#include "storage/DataBlock.h"
#include "storage/IO.h"
#include "storage/SparseDataBlock.h"
#include "storage/Utils.h"

#include "storage/tests/StorageTestHelpers.h"

#include <cstdlib>
#include <memory>


namespace obamadb {

  TEST(IOTest, TestLoadSparse) {
    std::vector<SparseDataBlock<float_t>*> blocks = IO::load<float_t>("sparse.dat");
    ASSERT_EQ(1, blocks.size());
    std::unique_ptr<SparseDataBlock<float_t>> block(dynamic_cast<SparseDataBlock<float_t>*>(blocks[0]));

    se_vector<float_t> r1;
    block->getRowVector(0, &r1);
    EXPECT_EQ(-1, *r1.getClassification());
    EXPECT_EQ(3, r1.numElements());
    EXPECT_EQ(10, r1.index_[0]);
  }

  TEST(IOTest, TestCreateSparseProjection) {
    double m = 47000, n = 20; // m corresponds to the dimension of the original matrix.
    std::unique_ptr<SparseDataBlock<signed char>> pdb(GetRandomProjectionMatrix(m,n));
    ASSERT_EQ(m, pdb->getNumRows());
    ASSERT_EQ(n, pdb->getNumColumns());
    int counts[3] = {0,0,0};
    se_vector<signed char> row;
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

}