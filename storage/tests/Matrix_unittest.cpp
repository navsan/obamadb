#include "gtest/gtest.h"
#include "storage/DataBlock.h"
#include "storage/IO.h"
#include "storage/Matrix.h"
#include "storage/SparseDataBlock.h"
#include "storage/Utils.h"

#include "storage/tests/StorageTestHelpers.h"

#include <cstdlib>
#include <memory>


namespace obamadb {

  // 0.7 sparsity says 70% elements are zero
  Matrix* getRandomSparseMatrix(int m, int n, double sparsity) {
    Matrix *mat = new Matrix();
    QuickRandom qr;
    std::uint32_t sparsity_lim = 0;
    sparsity_lim -= 1;
    sparsity_lim *= sparsity;

    se_vector<float_t> row;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (qr.nextInt32() > sparsity_lim) {
          row.push_back(j, qr.nextInt32()/10000.0);
        }
      }
      mat->addRow(row);
      row.clear();
    }
    return mat;
  }

  TEST(TestMatrix, TestLoad) {
    std::vector<SparseDataBlock<float_t>*> blocks = IO::load_blocks<float_t>("sparse.dat");
    ASSERT_EQ(1, blocks.size());
    Matrix mat(blocks);
    EXPECT_EQ(blocks[0]->getNumColumns(), mat.numColumns_);
    EXPECT_EQ(blocks[0]->getNumRows(), mat.numRows_);

    // try another method of loading the matrix
    std::unique_ptr<SparseDataBlock<float_t>> block(IO::load_blocks<float_t>("sparse.dat").front());
    Matrix mat2;
    se_vector<float_t> row;
    row.setMemory(0, nullptr);
    for(int i = 0; i < block->getNumRows(); i++) {
      block->getRowVectorFast(i, &row);
      mat2.addRow(row);
    }
    EXPECT_EQ(block->getNumColumns(), mat2.numColumns_);
    EXPECT_EQ(block->getNumRows(), mat2.numRows_);
  }

  TEST(TestMatrix, TestProjection) {
    std::unique_ptr<Matrix> mat(getRandomSparseMatrix(1000,1000, 0.9));
    std::pair<Matrix*, SparseDataBlock<signed char>*> res_pair = mat->randomProjectionsCompress(mat->numColumns_ * 0.5);
    std::unique_ptr<Matrix> compressed_mat(res_pair.first);
    const int kCompressionConstant = 0.5 * mat->numColumns_;
    std::vector<SparseDataBlock<signed char>*> proj_vec = { res_pair.second };

    IO::save<signed char>("/tmp/matB.csv", proj_vec, 1);
    IO::save("/tmp/matA.csv", *mat);
    IO::save("/tmp/matR.csv", *compressed_mat);
  }

  TEST(TestMatrix, TestRandomMatrix) {
    int m = 1000, n = 100;
    double sparsity = 0.9, tolerance = 0.03;
    std::unique_ptr<Matrix> mat(getRandomSparseMatrix(m,n, sparsity));
    double actual_sparsity = mat->getSparsity();
    DCHECK(actual_sparsity > sparsity - tolerance && actual_sparsity < sparsity + tolerance);
  }
}