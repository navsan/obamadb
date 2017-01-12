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

  namespace {

    inline int_t sparseDot(const svector<int_t> & a, const svector<signed char> & b) {
      int ai = 0, bi = 0;
      int_t sum_prod = 0;
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
    Matrix(const std::vector<SparseDataBlock<int_t> *> &blocks)
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
    void addBlock(SparseDataBlock<int_t> *block) {
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
    void addRow(const svector<int_t> &row) {
      if(blocks_.size() == 0 || !blocks_.back()->appendRow(row)) {
        blocks_.push_back(new SparseDataBlock<int_t>());
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
                  int_t kNormalizingConstant,
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
      int_t kNormalizingConstant_;
      Matrix *result_;
      int total_threads_;
      std::mutex result_lock_;
    };

    static void parallelMultiplyHelper(int thread_id, void* state) {
      PMultiState *pstate = reinterpret_cast<PMultiState *>(state);
      const int numBlocks = pstate->matA_->blocks_.size();
      const int blocks_per_thread = numBlocks / pstate->total_threads_;
      int block_lower_lim = blocks_per_thread * thread_id;
      int block_upper_lim = std::min(blocks_per_thread * (thread_id + 1), numBlocks);

      const std::vector<SparseDataBlock<int_t> *> &blocks_ = pstate->matA_->blocks_;
      svector<int_t> row_a(0, nullptr);
      svector<signed char> row_b(0, nullptr);

      SparseDataBlock<int_t> *result_block = new SparseDataBlock<int_t>();
      for (int i = block_lower_lim; i < block_upper_lim; i++) {
        const SparseDataBlock<int_t> *block = blocks_[i];
        for (int j = 0; j < block->getNumRows(); j++) {
          svector<int_t> row_c;
          block->getRowVectorFast(j, &row_a);
          for (int k = 0; k < pstate->matB_->getNumRows(); k++) {
            pstate->matB_->getRowVectorFast(k, &row_b);
            int_t f = sparseDot(row_a, row_b);
            if (f != 0) {
              row_c.push_back(k, f * pstate->kNormalizingConstant_);
            }
          }
          *row_c.class_ = *row_a.class_;
          if (!result_block->appendRow(row_c)) {
            pstate->result_lock_.lock();
            pstate->result_->addBlock(result_block);
            pstate->result_lock_.unlock();
            result_block = new SparseDataBlock<int_t>();
          }
        }
      }

      if (result_block->getNumRows() != 0) {
        pstate->result_lock_.lock();
        pstate->result_->addBlock(result_block);
        pstate->result_lock_.unlock();
      }
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
    Matrix* matrixMultiplyRowWise(const SparseDataBlock<signed char>* mat,
                                  int_t kNormalizingConstant) const {
      Matrix *result = new Matrix();
      // Hack to make this parallel
      int numThreads = std::min((size_t)threading::numCores(), blocks_.size());
      DLOG(INFO) << "Parallelizing matrix multiplication with " << numThreads << " threads";

      PMultiState * shared_state = new PMultiState(this, mat, kNormalizingConstant, result, numThreads);
      ThreadPool tp(Matrix::parallelMultiplyHelper, shared_state, numThreads);
      tp.begin();
      tp.cycle();
      tp.stop();

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
      const int_t kNormalizingConstant = 1.0 / sqrt(compressionConstant);
      return matrixMultiplyRowWise(projection_mat, compressionConstant);
    }

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
    std::vector<SparseDataBlock<int_t>*> blocks_;

    DISABLE_COPY_AND_ASSIGN(Matrix);
  };

}

#endif //OBAMADB_MATRIX_H
