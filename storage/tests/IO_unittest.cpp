#include "gtest/gtest.h"
#include "storage/IO.h"
#include "storage/exvector.h"
#include "storage/DataBlock.h"
#include "storage/SparseDataBlock.h"

#include <memory>

#include "storage/tests/StorageTestHelpers.h"

namespace obamadb {

  TEST(IOTest, TestLoadSparse) {

    std::vector<SparseDataBlock<num_t>*> blocks = IO::loadBlocks<num_t>("sparse.dat");
    ASSERT_EQ(1, blocks.size());
    std::unique_ptr<SparseDataBlock<num_t>> block(dynamic_cast<SparseDataBlock<num_t>*>(blocks[0]));

    EXPECT_EQ(2, block->getNumRows());
    EXPECT_EQ(24, block->getNumColumns());

    svector<num_t> row;
    block->getRowVector(0, &row);

    EXPECT_EQ(12.111, row.values_[row.numElements() - 1]);
    EXPECT_EQ(-1, *row.getClassification());
  }
}