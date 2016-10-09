#include "gtest/gtest.h"
#include "storage/Loader.h"
#include "storage/DataBlock.h"

#include "storage/tests/StorageTestHelpers.h"

namespace obamadb {

  TEST(LoaderTest, TestLoad) {
    const unsigned size_dataset = sizeof(iris_data)/sizeof(double);
    ASSERT_EQ(150 * 5, size_dataset);

    Loader loader;
    std::unique_ptr<DataBlock> block(loader.load(test_file));

    EXPECT_EQ(size_dataset, block->getSize());
    double *store = block->getStore();
    for (unsigned i = 0; i < size_dataset; i++) {
      EXPECT_EQ(iris_data[i], store[i]);
    }
  }
}