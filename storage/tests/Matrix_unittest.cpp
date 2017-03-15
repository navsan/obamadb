#include "gtest/gtest.h"

#include "storage/DataBlock.h"
#include "storage/exvector.h"
#include "storage/IO.h"
#include "storage/Matrix.h"
#include "storage/SparseDataBlock.h"
#include "storage/Utils.h"

#include <cstdlib>
#include <memory>

namespace obamadb {

  /**
   * Creates a random matrix with a desired sparsity.
   * 0.7 sparsity says 70% elements are zero
   */
  Matrix* getRandomSparseMatrix(int m, int n, double sparsity) {
    Matrix *mat = new Matrix();
    int sparsity_lim = INT_MAX;
    sparsity_lim *= sparsity;

    svector<num_t> row;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (std::abs(randomInt()) > sparsity_lim) {
          row.push_back(j, randomFloat());
        }
      }
      mat->addRow(row);
      row.clear();
    }
    return mat;
  }

  TEST(TestMatrix, TestLoad) {
    std::vector<SparseDataBlock<num_t>*> blocks = IO::loadBlocks<num_t>("sparse.dat");
    ASSERT_EQ(1, blocks.size());
    Matrix mat(blocks);
    EXPECT_EQ(blocks[0]->getNumColumns(), mat.numColumns_);
    EXPECT_EQ(blocks[0]->getNumRows(), mat.numRows_);

    // try another method of loading the matrix
    std::unique_ptr<SparseDataBlock<num_t>> block(IO::loadBlocks<num_t>("sparse.dat").front());
    Matrix mat2;
    svector<num_t> row;
    row.setMemory(0, nullptr);
    for(int i = 0; i < block->getNumRows(); i++) {
      block->getRowVectorFast(i, &row);
      mat2.addRow(row);
    }
    EXPECT_EQ(block->getNumColumns(), mat2.numColumns_);
    EXPECT_EQ(block->getNumRows(), mat2.numRows_);
  }

  /**
   * Tests the sparsity method.
   */
  TEST(TestMatrix, TestGetSparsity) {
    int m = 1000, n = 100;
    double sparsity = 0.9, tolerance = 0.03;
    std::unique_ptr<Matrix> mat(getRandomSparseMatrix(m,n, sparsity));
    double actual_sparsity = mat->getSparsity();
    DCHECK(actual_sparsity > sparsity - tolerance && actual_sparsity < sparsity + tolerance);
  }

  TEST(TestMatrix, TestRandomSparseMatrix) {
    int ncolumns = 10000;
    int matrixSizeBytes = 16e6;
    double sparsity = 0.99;
    double const tolerance = 0.03;
    std::unique_ptr<Matrix> mat(Matrix::GetRandomMatrix(matrixSizeBytes, ncolumns, sparsity));
    double actual_sparsity = mat->getSparsity();
    ASSERT_TRUE(actual_sparsity > sparsity - tolerance && actual_sparsity < sparsity + tolerance);
    EXPECT_LE(matrixSizeBytes * tolerance, mat->sizeBytes());
    EXPECT_GE(matrixSizeBytes * (1 + tolerance), mat->sizeBytes());

    int numPositive = 0;
    svector<float_t> rowView(0,nullptr);
    for(int i = 0; i < mat->blocks_.size(); i++) {
      SparseDataBlock<float_t> const *block = mat->blocks_[i];
      for (int j = 0; j < block->num_rows_; j++) {
        block->getRowVectorFast(j, &rowView);
        if (*rowView.class_ == 1) {
          numPositive++;
        } else {
          ASSERT_EQ(-1, *rowView.class_);
        }
      }
    }
    EXPECT_GE(((int)mat->numRows_/2) * tolerance, (int)(mat->numRows_/2) - numPositive);
  }
}