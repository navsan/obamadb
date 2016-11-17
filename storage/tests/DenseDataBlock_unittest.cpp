#include "gtest/gtest.h"
#include "storage/DataBlock.h"
#include "storage/DenseDataBlock.h"
#include "storage/exvector.h"
#include "storage/IO.h"
#include "storage/Utils.h"

#include "storage/tests/StorageTestHelpers.h"

#include <memory>

namespace obamadb {

  TEST(DenseDataBlockTest, TestLoadDense) {
    int m = 1000;
    int n = 10000;
    std::unique_ptr<DenseDataBlock<float_t>> block(new DenseDataBlock<float_t>(m,n));
    dvector<float_t> row(n);
    row.num_elements_ = n;
    for(int i = 0; i < m; i++) {
      for(int j = 0; j < n; j++) {
        row.values_[j] = j + (i * n);
      }
      row.class_[0] = (i % 2) == 0;
      ASSERT_TRUE(block->appendRow(row));
    }
    if (kStorageBlockSize > (m * (n+1) * sizeof(float_t))) {
      EXPECT_TRUE(block->appendRow(row));
    } else {
      EXPECT_FALSE(block->appendRow(row));
    }
    block->finalize();

    dvector<float_t> read_vec(0, nullptr);
    for(int i = 0; i < m - 1; i++) {
      block->getRowVectorFast(i, &read_vec);
      ASSERT_EQ(read_vec.size(), n);
      ASSERT_EQ(*read_vec.class_, (i % 2) == 0);
      for (int j = 0; j < n; j++) {
        ASSERT_EQ(read_vec.values_[j], j + (i * n));
      }
    }
  }
}