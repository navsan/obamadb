#include "gtest/gtest.h"
#include "storage/DataBlock.h"
#include "storage/IO.h"
#include "storage/SparseDataBlock.h"
#include "storage/Utils.h"

#include "storage/tests/StorageTestHelpers.h"

#include <cstdlib>

namespace obamadb {

  TEST(IOTest, TestLoadSparse) {
    std::vector<SparseDataBlock<double>*> blocks = IO::load<double>("sparse.dat");
    ASSERT_EQ(1, blocks.size());
    std::unique_ptr<SparseDataBlock<double>> block(dynamic_cast<SparseDataBlock<double>*>(blocks[0]));

    se_vector<double> r1;
    block->getRowVector(0, &r1);
    EXPECT_EQ(-1, *r1.getClassification());
    EXPECT_EQ(3, r1.numElements());
    EXPECT_EQ(10, r1.index_[0]);
  }

}