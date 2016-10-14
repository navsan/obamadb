#include "gtest/gtest.h"
#include "storage/DataBlock.h"
#include "storage/LinearMath.h"
#include "storage/Loader.h"

#include "storage/tests/StorageTestHelpers.h"

#include <cstdlib>

namespace obamadb {

  DataBlock *getIrisData() {
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
    std::unique_ptr<DataBlock> yvalues(iris_datablock->slice(0, 1));
    ASSERT_EQ(150, yvalues->getSize());
    ASSERT_EQ(1, yvalues->getWidth());
    for (unsigned i = 0; i < yvalues->getSize(); ++i) {
      EXPECT_EQ(iris_data[i * 5], iris_datablock->get(i, 0));
    }

    // These are the size data values.
    std::unique_ptr<DataBlock> data_values(iris_datablock->slice(1, 5));
    ASSERT_EQ(150 * 4, data_values->getSize());
    ASSERT_EQ(4, data_values->getWidth());
    unsigned iris_cursor = 1;
    for (unsigned i = 0; i < yvalues->getSize(); ++i) {
      EXPECT_EQ(iris_data[iris_cursor], data_values->get(i / 4, i % 4));
      iris_cursor++;
      if (iris_cursor % 5 == 0) {
        iris_cursor++;
      }
    }
  }

  TEST(DataBlockTest, TestMatchRows) {
    std::unique_ptr<DataBlock> iris_datablock(getIrisData());
    std::vector<double> check_for_values = {0.0, 1.0, 2.0};
    for (auto double_itr : check_for_values) {
      std::vector<unsigned> row_matches;
      std::function<bool(double)> filter_fn = [double_itr](double input) {
        return input == double_itr;
      };
      iris_datablock->matchRows(filter_fn, 0, row_matches);
      EXPECT_EQ(50, row_matches.size());
      for (unsigned row = 0; row < (sizeof(iris_data) / sizeof(double)) / 5; row++) {
        bool row_present = std::find(row_matches.begin(), row_matches.end(), row) != row_matches.end();
        bool should_be_present = iris_data[row * 5] == double_itr;
        EXPECT_EQ(should_be_present, row_present);
      }
    }
  }

  TEST(DataBlockTest, TestSliceRows) {
    std::unique_ptr<DataBlock> iris_datablock(getIrisData());\
    std::vector<unsigned> row_matches;
    std::function<bool(double)> filter_fn = [](double input) {
      return input == 0.0;
    };
    iris_datablock->matchRows(filter_fn, 0, row_matches);
    ASSERT_EQ(50, row_matches.size());
    std::unique_ptr<DataBlock> iris_slice(iris_datablock->sliceRows(row_matches));
    EXPECT_EQ(50, iris_slice->getNumRows());
    unsigned dst_row = 0;
    for (unsigned src_row = 0; src_row < iris_datablock->getNumRows(); src_row++) {
      if (iris_datablock->get(src_row, 0) == 0.0) {
        for (unsigned attr = 1; attr < 5; ++attr) {
          EXPECT_EQ(iris_datablock->get(src_row, attr), iris_slice->get(dst_row, attr));
        }
        dst_row++;
      }
    }
  }

  TEST(DataBlockTest, TestSyntheticDataSet) {
    // This test asserts that the Synthetic dataset class produces 2 clusters of data which are
    // linearly seperable.
    const unsigned dimension = 10;
    const unsigned num_examples = 100000;
    const double elipson = 0.001; // accounts for rounding errors in some checks
    const double radius = 2.5;
    DoubleVector v1(dimension);
    DoubleVector v2(dimension);

    for (unsigned i = 0; i < dimension; ++i) {
      v1.values_[i] = 25;
      v2.values_[i] = -25;
    }

    SyntheticDataSet data_set(dimension, num_examples, v1.values_, v2.values_, radius, radius);
    std::vector<DataBlock *> blocks = data_set.getDataSet();
    ASSERT_LT(0, blocks.size());
    unsigned num_positive_examples = 0;
    for (DataBlock *block : blocks) {
      EXPECT_LT(0, block->getNumRows());
      std::unique_ptr<DataBlock> training_data(block->slice(0, dimension));
      std::unique_ptr<DataBlock> y_vals(block->slice(dimension, dimension + 1));

      for (unsigned row = 0; row < block->getNumRows(); ++row) {
        double y_val = y_vals->get(row, 0);
        double *train_example = training_data->getRow(row);

        if (0 == y_val) {
          EXPECT_GE(radius + elipson, distance(train_example, v2.values_, dimension));
        } else if (1 == y_val) {
          EXPECT_GE(radius + elipson, distance(train_example, v1.values_, dimension));
          num_positive_examples++;
        } else {
          EXPECT_TRUE(false) << "Y vals should be either 1 or 0";
        }
      }
    }
    EXPECT_NEAR(num_positive_examples, num_examples - num_positive_examples, num_examples * 0.05);
  }

  TEST(DataBlockTest, TestSGD) {
    const unsigned learning_itrs = 1000;

    const unsigned dimension = 10;
    const unsigned num_examples = 10000;
    const double elipson = 0.001; // accounts for rounding errors in some checks
    const double radius = 20;
    DoubleVector v1(dimension);
    DoubleVector v2(dimension);

    for (unsigned i = 0; i < dimension; ++i) {
      v1.values_[i] = 25;
      v2.values_[i] = -25;
    }

    SyntheticDataSet data_set(dimension, num_examples, v1.values_, v2.values_, radius, radius);
    std::vector<DataBlock *> blocks = data_set.getDataSet();

    // Initialize theta
    DoubleVector theta(dimension);

    for (unsigned i = 0; i < dimension; i++) {
      theta[i] = 0.2;
    }

    for (unsigned i = 0; i < learning_itrs; ++i) {
      for (DataBlock *block : blocks) {
        std::unique_ptr<DataBlock> a_vals(block->slice(0, dimension));
        std::unique_ptr<DataBlock> y_vals(block->slice(dimension, dimension + 1));
        gradientItr(a_vals.get(), y_vals.get(), theta.values_);
      }
      // Calculate the error using the back block.

      std::unique_ptr<DataBlock> a_vals(blocks.back()->slice(0, dimension));
      std::unique_ptr<DataBlock> y_vals(blocks.back()->slice(dimension, dimension + 1));
      double err = error(a_vals.get(), y_vals.get(), theta.values_);
      if (err == 0)
        break;
    }
    std::unique_ptr<DataBlock> a_vals(blocks.back()->slice(0, dimension));
    std::unique_ptr<DataBlock> y_vals(blocks.back()->slice(dimension, dimension + 1));
    double err = error(a_vals.get(), y_vals.get(), theta.values_);
    EXPECT_NEAR(0, err, 0.03); // really should be zero
  }

}