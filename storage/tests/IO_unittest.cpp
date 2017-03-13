#include "gtest/gtest.h"

#include "storage/IO.h"
#include "storage/exvector.h"
#include "storage/DataBlock.h"
#include "storage/SparseDataBlock.h"
#include "storage/tests/StorageTestHelpers.h"

#include "gflags/gflags.h"

#include <memory>

DEFINE_bool(liblinear, false, "");

namespace obamadb {

  TEST(IOTest, TestLoadSparse) {

    std::vector<SparseDataBlock<num_t>*> blocks = IO::loadBlocks<num_t>("sparse.dat");
    ASSERT_EQ(1, blocks.size());
    std::unique_ptr<SparseDataBlock<num_t>> block(dynamic_cast<SparseDataBlock<num_t>*>(blocks[0]));

    EXPECT_EQ(2, block->getNumRows());
    EXPECT_EQ(24, block->getNumColumns());

    svector<num_t> row;
    block->getRowVector(0, &row);

    EXPECT_NEAR(12.111, row.values_[row.numElements() - 1], 0.001);
    EXPECT_EQ(-1, *row.getClassification());
  }

  TEST(IOTest, TestLoadLibLinearFile) {
    bool const prev_flag = FLAGS_liblinear;
    FLAGS_liblinear = true;

    std::vector<SparseDataBlock<num_t>*> blocks = IO::loadBlocks<num_t>("heart_scale.dat");
    ASSERT_EQ(1, blocks.size());
    ASSERT_EQ(269, blocks[0]->num_rows_);
    ASSERT_EQ(14, blocks[0]->num_columns_);

    FLAGS_liblinear = prev_flag;
  }
}

int main(int argc, char **argv) {
  ::google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}