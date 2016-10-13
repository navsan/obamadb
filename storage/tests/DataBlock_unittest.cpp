#include "gtest/gtest.h"
#include "storage/DataBlock.h"
#include "storage/LinearMath.h"
#include "storage/Loader.h"

#include "storage/tests/StorageTestHelpers.h"

namespace obamadb {

  class SyntheticDataSet {
  public:
    SyntheticDataSet(
      unsigned dimension,
      unsigned training_examples,
      double *p1,
      double *p2,
      double r1,
      double r2)
      : dimension_(dimension),
        training_examples_(training_examples),
        p1_(p1),
        p2_(p2),
        r1_(r1),
        r2_(r2),
        d_(distance(p1, p2, dimension)) {

    }

    unsigned dimension_;
    unsigned training_examples_;
    double *p1_;
    double *p2_;
    const double r1_;
    const double r2_;
    const double d_;

    std::vector<DataBlock*> blocks_;
  };

  DataBlock* getIrisData() {
    auto blocks = Loader::load("iris.dat");
    LOG_IF(FATAL, 0 == blocks.size()) << "Could not open file: iris.dat. Are you running this in the test directory?";
    DataBlock *data = *blocks.begin();
    data->setWidth(5);
    CHECK_EQ(750, data->getSize());
    return data;
  }

  TEST(DataBlockTest, TestSlice) {
    std::unique_ptr<DataBlock> iris_datablock(getIrisData());

    // These are the category values.
    std::unique_ptr<DataBlock> yvalues(iris_datablock->slice(0,1));
    ASSERT_EQ(150, yvalues->getSize());
    ASSERT_EQ(1, yvalues->getWidth());
    for (unsigned i = 0; i < yvalues->getSize(); ++i) {
      EXPECT_EQ(iris_data[i * 5], iris_datablock->get(i, 0));
    }

    // These are the size data values.
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

  TEST(DataBlockTest, TestMatchRows) {
    std::unique_ptr<DataBlock> iris_datablock(getIrisData());
    std::vector<double> check_for_values = {0.0, 1.0, 2.0};
    for(auto double_itr : check_for_values) {
      std::vector<unsigned> row_matches;
      std::function<bool(double)> filter_fn = [double_itr] (double input) {
        return input == double_itr;
      };
      iris_datablock->matchRows(filter_fn, 0, row_matches);
      EXPECT_EQ(50, row_matches.size());
      for(unsigned row = 0; row < (sizeof(iris_data)/sizeof(double))/5; row++) {
        bool row_present = std::find(row_matches.begin(), row_matches.end(), row) != row_matches.end();
        bool should_be_present = iris_data[row * 5] == double_itr;
        EXPECT_EQ(should_be_present, row_present);
      }
    }
  }

  TEST(DataBlockTest, TestSliceRows) {
    std::unique_ptr<DataBlock> iris_datablock(getIrisData());\
    std::vector<unsigned> row_matches;
    std::function<bool(double)> filter_fn = [] (double input) {
      return input == 0.0;
    };
    iris_datablock->matchRows(filter_fn, 0, row_matches);
    ASSERT_EQ(50, row_matches.size());
    std::unique_ptr<DataBlock> iris_slice(iris_datablock->sliceRows(row_matches));
    EXPECT_EQ(50, iris_slice->getNumRows());
    unsigned dst_row = 0;
    for(unsigned src_row = 0; src_row < iris_datablock->getNumRows(); src_row++) {
      if (iris_datablock->get(src_row, 0) == 0.0) {
        for (unsigned attr = 1; attr < 5; ++attr) {
          EXPECT_EQ(iris_datablock->get(src_row, attr), iris_slice->get(dst_row, attr));
        }
        dst_row++;
      }
    }
  }

  TEST(DataBlockTest, TestSyntheticDataSet) {
    ASSERT_EQ(1, 10);
  }

}