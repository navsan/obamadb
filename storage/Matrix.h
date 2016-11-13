#ifndef OBAMADB_MATRIX_H
#define OBAMADB_MATRIX_H

#include "storage/SparseDataBlock.h"
#include "storage/StorageConstants.h"

#include <vector>

namespace obamadb {

  namespace {

    inline float_t sparseDot(const se_vector<float_t> & a, const se_vector<signed char> & b) {
      int ai = 0, bi = 0;
      float_t sum_prod = 0;
      while(ai < a.num_elements_ && bi < b.num_elements_) {
        if(a.index_[ai] == b.index_[bi]) {
          sum_prod += a.values_[ai] * b.values_[bi];
          ai++; bi++;
        } else if (a.index_[ai] < b.index_[bi]) {
          ai++;
        } else {
          bi++;
        }
      }
      return sum_prod;
    }
  }

  class Matrix {
  public:
    /**
     * Takes ownership of the passed DataBlocks.
     */
    Matrix(const std::vector<SparseDataBlock<float_t> *> &blocks)
      : numColumns_(0),
        numRows_(0),
        blocks_() {
      for (int i = 0; i < blocks.size(); i++) {
        addBlock(blocks[i]);
      }
    }

    /**
     * Takes ownership of the passed DataBlocks.
     */
    Matrix()
      : numColumns_(0),
        numRows_(0),
        blocks_() {}

    Matrix(const Matrix& other) = delete;

    Matrix& operator=(const Matrix& other) = delete;

    ~Matrix() {
      for(auto block : blocks_)
        delete block;
    }

    /**
     * Takes ownership of an entire block and adds it to the matrix and increases the size if necessary.
     * @param block The sparse datablock to add.
     */
    void addBlock(SparseDataBlock<float_t> *block) {
      if (block->getNumColumns() > numColumns_) {
        numColumns_ = block->getNumColumns();
        // each block should be the same dimension as the matrix.
        for (int i = 0; i < blocks_.size(); i++) {
          blocks_[i]->num_columns_ = numColumns_;
        }
      }
      numRows_ += block->getNumRows();
      blocks_.push_back(block);
    }

    /**
     * Appends the row to the last block in the matrix's list of datablocks. If it does not fit, a new data
     * block will be created.
     * @param row Row to append
     */
    void addRow(const se_vector<float_t> &row) {
      if(blocks_.size() == 0 || !blocks_.back()->appendRow(row)) {
        blocks_.push_back(new SparseDataBlock<float_t>());
        bool appended = blocks_.back()->appendRow(row);
        DCHECK(appended);
      }

      if (row.size() > numColumns_) {
        numColumns_ = row.size();
        for (int i = 0; i < blocks_.size(); i++) {
          blocks_[i]->num_columns_ = numColumns_;
        }
      }

      numRows_++;
    }

    /**
     * Do a row-by-row multiplication (normally we do a row-column multiplication, but here we
     * are much better optimized for row wise multiplications and so we do this method.
     *
     * The operation A * B is equivilent to A rowwise* B' where B' is the transpose of B.
     * @param mat
     * @param kNormalizingConstant An optional constant to mutliply each memeber by (chose 1 if not desired)
     * @return Caller-owned matrix result of the multiplication.
     */
    Matrix* matrixMultiplyRowWise(const SparseDataBlock<signed char>* mat, float_t kNormalizingConstant) const {
      Matrix *result = new Matrix();
      se_vector<float_t> row_a(0, nullptr);
      se_vector<signed char> row_b(0, nullptr);

      int current_block = 0;
      for (int i = 0; i < blocks_.size(); i++) {
        const SparseDataBlock<float_t> *block = blocks_[i];
        for (int j = 0; j < block->getNumRows(); j++) {
          se_vector<float_t> row_c;
          block->getRowVectorFast(j, &row_a);
          for (int k = 0; k < mat->getNumRows(); k++) {
            mat->getRowVectorFast(k, &row_b);
            float_t f = sparseDot(row_a, row_b);
            if (f != 0) {
              row_c.push_back(k, f * kNormalizingConstant);
            }
          }
          result->addRow(row_c);
        }
      }
      return result;
    }

    /**
     * Performs a random projection multiplication on the matrix and returns a new compressed
     * version of the matrix.
     *
     * This corresponds to the creating the matrix b in b = (1/sqrt(k))A*R where R is the random
     * projections matrix with i.i.d entries, zero mean, and constant variance.
     *
     * @return the compressed matrix (b), and the projection matrix (r) used to generate it.
     */
    std::pair<Matrix*, SparseDataBlock<signed char>*> randomProjectionsCompress(int compressionConstant) const {
      // TODO: How do we choose compressionConstant? Corresponds to k in the Very Sparse Random Projections
      const float_t kNormalizingConstant = 1.0/sqrt(compressionConstant);
      std::unique_ptr<SparseDataBlock<signed char>> projection(
        GetRandomProjectionMatrix(numColumns_, compressionConstant));
      return
        {
          this->matrixMultiplyRowWise(projection.get(), compressionConstant),
          projection.release()
        };
    }

    /**
     * Uses a given projection matrix to perform compression
     *
     * @param projection_mat
     * @param compressionConstant
     * @return
     */
    Matrix* randomProjectionsCompress(SparseDataBlock<signed char>* projection_mat, int compressionConstant) const {
      const float_t kNormalizingConstant = 1.0/sqrt(compressionConstant);
      return matrixMultiplyRowWise(projection_mat, compressionConstant);
    }

    /**
     * @return Fraction of elements which are zero.
     */
    double getSparsity() const {
      std::uint64_t nnz = 0;
      std::uint64_t numElements = static_cast<std::uint64_t >(numColumns_) * static_cast<std::uint64_t >(numRows_);
      for (auto block : blocks_) {
        nnz += block->numNonZeroElements();
      }
      return (double ) (numElements - nnz) / (double) numElements;
    }

    /**
     * @return The total size of the owned data.
     */
    std::uint64_t sizeBytes() const {
      std::uint64_t size;
      for(auto block : blocks_) {
        size += block->block_size_bytes_;
      }
      return size;
    }

    int numColumns_;
    int numRows_;
    std::vector<SparseDataBlock<float_t>*> blocks_;
  };

}

#endif //OBAMADB_MATRIX_H
