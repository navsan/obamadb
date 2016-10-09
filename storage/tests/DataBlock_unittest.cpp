#include "gtest/gtest.h"
#include "storage/DataBlock.h"
#include "storage/Loader.h"

#include "storage/tests/StorageTestHelpers.h"

namespace obamadb {

  TEST(DataBlockTest, TestSlice) {
    std::unique_ptr<DataBlock> iris_datablock(Loader::load("iris.dat"));
    iris_datablock->setWidth(5);
    std::unique_ptr<DataBlock> yvalues(iris_datablock->slice(0,1));
    ASSERT_EQ(150, yvalues->getSize());
    ASSERT_EQ(1, yvalues->getWidth());
    for (unsigned i = 0; i < yvalues->getSize(); ++i) {
      EXPECT_EQ(iris_data[i * 5], iris_datablock->get(i, 0));
    }

    std::unique_ptr<DataBlock> data_values(iris_datablock->slice(1,5));
    ASSERT_EQ(150 * 4, data_values->getSize());
    ASSERT_EQ(4, data_values->getWidth());
    unsigned iris_cursor = 1;
    for (unsigned i = 0; i < yvalues->getSize(); ++i) {
      EXPECT_EQ(iris_data[iris_cursor], data_values->get(i/4, i%4));
      iris_cursor++;
      if(iris_cursor % 5 == 0) {
        iris_cursor++;
      }
    }

  }
}