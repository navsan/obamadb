#ifndef OBAMADB_MATRIX_H
#define OBAMADB_MATRIX_H

#include "storage/exvector.h"
#include "storage/SparseDataBlock.h"
#include "storage/StorageConstants.h"
#include "storage/ThreadPool.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

namespace obamadb {

  class Matrix {
  public:
    /**
     * Takes ownership of the passed DataBlocks.
     */
    Matrix(const std::vector<SparseDataBlock<num_t> *> &blocks)
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

    ~Matrix() {
      for(auto block : blocks_) {
        delete block;
      }
    }

    /**
     * Takes ownership of an entire block and adds it to the matrix and increases the size if necessary.
     * @param block The sparse datablock to add.
     */
    void addBlock(SparseDataBlock<num_t> *block) {
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
     * Samples a percentage of the table and returns a new matrix which is some size of the original.
     * Samples with replacement.
     * @param percent
     * @return A user owned matrix of sampled entries
     */
    Matrix* sample(float percent) const {
      std::vector<SparseDataBlock<num_t> *> blocks;
      int rows_per_block = percent * (static_cast<float>(numRows_)/this->blocks_.size());
      DCHECK_GT(rows_per_block, 0);
      auto insertInto = [&](svector<num_t> const & src,
                            SparseDataBlock<num_t> * &dst) {
        if (!dst->appendRow(src)) {
          blocks.push_back(dst);
          dst = new SparseDataBlock<num_t>();
          CHECK(dst->appendRow(src));
        }
      };
      svector<num_t> rand_row;
      SparseDataBlock<num_t> * curr_block = new SparseDataBlock<num_t>();
      for(auto & block : this->blocks_) {
        for (int row = 0; row < rows_per_block; row++) {
          int rrow = rand() % block->num_rows_;
          block->getRowVector(rrow, &rand_row);
          insertInto(rand_row, curr_block);
        }
      }
      blocks.push_back(curr_block);
      return new Matrix(blocks);
    }

    /**
     * Appends the row to the last block in the matrix's list of datablocks. If it does not fit, a new data
     * block will be created.
     * @param row Row to append
     */
    void addRow(const svector<num_t> &row) {
      if(blocks_.size() == 0 || !blocks_.back()->appendRow(row)) {
        blocks_.push_back(new SparseDataBlock<num_t>());
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

    struct PMultiState {
      PMultiState(const Matrix * matA,
                  const SparseDataBlock<signed char> *matB,
                  num_t kNormalizingConstant,
                  Matrix *result,
                  int total_threads)
        : matA_(matA),
          matB_(matB),
          kNormalizingConstant_(kNormalizingConstant),
          result_(result),
          total_threads_(total_threads),
          result_lock_() {}

      const Matrix * matA_;
      const SparseDataBlock<signed char>* matB_;
      num_t kNormalizingConstant_;
      Matrix *result_;
      int total_threads_;
      std::mutex result_lock_;
    };

    /**
     * Creates a random matrix which should be linearly seperable.
     *
     * @param matrixSizeBytes The approximate size of the resulting matrix.
     * @param numColumns The number of columns for the matrix to have.
     * @param sparsity The sparsity of the matrix.
     * @return A caller-owned sparse matrix.
     */
    static Matrix* GetRandomMatrix(int matrixSizeBytes, int numColumns, double sparsity) {
      Matrix* matrix = new Matrix();
      int totalDataSizeBytes = 0;
      while(totalDataSizeBytes < matrixSizeBytes) {
        int sizeNextBlockBytes = std::min(matrixSizeBytes - totalDataSizeBytes, (int)kStorageBlockSize);
        matrix->addBlock(
          GetRandomSparseDataBlock(sizeNextBlockBytes, numColumns, sparsity));
        totalDataSizeBytes += sizeNextBlockBytes;
      }
      return matrix;
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
     * Number of non-zero elements
     * @return
     */
    int getNNZ() const {
      int nnz = 0;
      for (auto block : blocks_) {
        nnz += block->numNonZeroElements();
      }
      return nnz;
    }

    /**
     * @return The total size of the owned data.
     */
    std::uint64_t sizeBytes() const {
      std::uint64_t size = 0;
      for(auto block : blocks_) {
        size += block->block_size_bytes_;
      }
      return size;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

    int numColumns_;
    int numRows_;
    std::vector<SparseDataBlock<num_t>*> blocks_;

    DISABLE_COPY_AND_ASSIGN(Matrix);
  };

}

#endif //OBAMADB_MATRIX_H
