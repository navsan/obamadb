#include "gtest/gtest.h"
#include "storage/IO.h"
#include "storage/DataBlock.h"
#include "storage/SparseDataBlock.h"

#include "storage/tests/StorageTestHelpers.h"

namespace obamadb {

  TEST(IOTest, TestLoadSparse) {

    std::vector<SparseDataBlock<double>*> blocks = IO::load<double>("sparse.dat");
    ASSERT_EQ(1, blocks.size());
    std::unique_ptr<SparseDataBlock<double>> block(dynamic_cast<SparseDataBlock<double>*>(blocks[0]));

    EXPECT_EQ(2, block->getNumRows());
    EXPECT_EQ(24, block->getNumColumns());

    se_vector<double> row;
    block->getRowVector(0, &row);

    EXPECT_EQ(12.111, row.values_[row.numElements() - 1]);
    EXPECT_EQ(-1, *row.getClassification());
  }
}