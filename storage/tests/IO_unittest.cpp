#include "gtest/gtest.h"

#include "storage/DataBlock.h"
#include "storage/exvector.h"
#include "storage/IO.h"
#include "storage/SparseDataBlock.h"

#include "gflags/gflags.h"

#include <memory>

namespace obamadb {

  TEST(IOTest, TestLoadLibLinearFile) {
    std::vector<num_t> first_row = {0, 0.708333, 1, 1,-0.320755,-0.105023,-1,1,-0.419847,-1,-0.225806,0,1,-1};
    std::vector<num_t> last_row = {0,0.583333,1,1,0.245283,-0.269406,-1,1,-0.435115,1,-0.516129,0,1,-1};

    std::vector<SparseDataBlock<num_t>*> blocks = IO::loadBlocks<num_t>("heart_scale.dat");
    ASSERT_EQ(1, blocks.size());
    ASSERT_EQ(270, blocks[0]->num_rows_);
    ASSERT_EQ(14, blocks[0]->num_columns_);

    svector<num_t> row(0, nullptr);
    blocks[0]->getRowVector(0, &row);
    for(int i = 0; i < first_row.size(); i++) {
      num_t expected = first_row[i];
      num_t *actual = row.get(i);
      if (expected == 0.0) {
        EXPECT_EQ(nullptr, actual);
      } else {
        ASSERT_NE(nullptr, actual);
        EXPECT_NEAR(*actual, expected, 0.001);
      }

    }
    blocks[0]->getRowVector(blocks[0]->num_rows_-1, &row);
    for(int i = 0; i < last_row.size(); i++) {
      num_t expected = last_row[i];
      num_t *actual = row.get(i);
      if (expected == 0.0) {
        EXPECT_EQ(nullptr, actual);
      } else {
        ASSERT_NE(nullptr, actual);
        EXPECT_NEAR(*actual, expected, 0.001);
      }
    }
  }

  TEST(IOTest, TestScanDoubles) {
    Scanner scanner("doubles.dat");
    int rows = 0;
    while (true) {
      std::vector<double> row = scanner.scanLine();
      if (row.size() == 0)
        break;
      rows++;
      ASSERT_EQ(2, row.size());
      EXPECT_NEAR(row[0], row[1], 1e-6);
    }
    ASSERT_EQ(4, rows);
  }
}

int main(int argc, char **argv) {
  ::google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}