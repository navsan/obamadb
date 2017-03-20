#ifndef OBAMADB_SPARSEDATABLOCK_H
#define OBAMADB_SPARSEDATABLOCK_H

#include "storage/DataBlock.h"
#include "storage/exvector.h"
#include "storage/StorageConstants.h"
#include "storage/Utils.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>

#include <glog/logging.h>

namespace obamadb {

  template<class T> class SparseDataBlock;

  template <typename T>
  std::ostream &operator<<(std::ostream &os, const SparseDataBlock<T> &block);

  std::ostream &operator<<(std::ostream &os, const SparseDataBlock<num_t> &block);

  /**
   * Optimized for storing rows of data where the majority of elements are null.
   */
  template<class T>
  class SparseDataBlock : public DataBlock<T> {
  public:
    struct SDBEntry {
      std::uint32_t offset_;
      std::uint32_t size_;
    };

    /**
     * Creates a datablock with the specified size.
     *
     * @param size_mb Size in bytes.
     */
    SparseDataBlock(unsigned numRows, unsigned numColumns)
      : DataBlock<T>(numRows, numColumns),
        entries_(reinterpret_cast<SDBEntry *>(this->store_)),
        heap_offset_(0),
        end_of_block_(reinterpret_cast<char *>(this->store_) + this->block_size_bytes_) {}

    /**
     * Creates a datablock with the specified size.
     *
     * @param size_mb Size in bytes.
     */
    SparseDataBlock(int size_bytes)
      : DataBlock<T>(size_bytes),
        entries_(reinterpret_cast<SDBEntry *>(this->store_)),
        heap_offset_(0),
        end_of_block_(reinterpret_cast<char *>(this->store_) + size_bytes) {}

    /**
     * Creates a sparse data block with linearly seperable rows
     * @param size_bytes
     * @param numColumns
     * @param sparsity
     */
    SparseDataBlock(int size_bytes, int numColumns, double sparsity)
      : DataBlock<T>(size_bytes),
        entries_(reinterpret_cast<SDBEntry *>(this->store_)),
        heap_offset_(0),
        end_of_block_(reinterpret_cast<char *>(this->store_) + size_bytes) {
      num_t positive = 1.0;
      num_t negative = -1.0;
      unsigned int seed = static_cast<unsigned int>(time(NULL));
      std::default_random_engine rng(seed);
      std::binomial_distribution<int> num_elems_in_row_dist(numColumns,1.0 - sparsity);
      std::bernoulli_distribution class_dist(0.5);
      std::uniform_int_distribution<int> index_dist(0,numColumns-1);
      std::uniform_real_distribution<num_t> feature_dist(0.0,1.0);
      svector<num_t> row_vector;
      int num_elems;
      do {
        row_vector.clear();
        bool isPositive = class_dist(rng);
        row_vector.setClassification(isPositive ? &positive : &negative);

        // Ensure that we have at least one feature in each row
        do {
          num_elems = num_elems_in_row_dist(rng);
        } while(num_elems == 0);

        for (int i = 0; i < num_elems; ++i) {
          int index = index_dist(rng);
          num_t randf = feature_dist(rng);
          row_vector.push_back(index, randf);
        }
        row_vector.sortIndexes();
        /* row_vector.print(); */
        for (int i = 0; i < num_elems; ++i) {
          // The data should end up being perfectly seperable.
          if ((isPositive && row_vector.index_[i] % 2 == 1)
              || (!isPositive && row_vector.index_[i] % 2 == 0)) {
            row_vector.values_[i] *= -1;
          }
        }
      } while (this->appendRow(row_vector));
      this->finalize();
    }

    SparseDataBlock() : SparseDataBlock(kStorageBlockSize) {}

    /**
     * Use this function while initializing to pack the block.
     * @param row Row to append.
     * @return True if the append succeeded, false if the block is full.
     */
    bool appendRow(const svector<T> &row);

    /**
     * Blocks which have been finalized no longer can have rows appended to them.
     */
    void finalize() {
      this->initializing_ = false;
    }

    DataBlockType getDataBlockType() const override {
      return DataBlockType::kSparse;
    }

    void getRowVector(int row, exvector<T> *vec) const override;

    void trimRows(int numRows);

    inline void getRowVectorFast(const int row, svector<T> *vec) const {
      DCHECK_LT(row, this->num_rows_) << "Row index out of range.";
      DCHECK_EQ(false, vec->owns_memory());

      SDBEntry const &entry = entries_[row];

      vec->num_elements_ = entry.size_;
      vec->index_ = reinterpret_cast<int*>(end_of_block_ - entry.offset_);
      vec->values_ = reinterpret_cast<T*>(vec->index_ + entry.size_);
      vec->class_ = vec->values_ + entry.size_;
    }

    /**
     * @return  The value stored at a particular index.
     */
    T* get(unsigned row, unsigned col) const override;

    T* operator()(unsigned row, unsigned col) override;

    int numNonZeroElements() const;

  private:
    /**
     * @return The number of bytes remaining in the heap.
     */
    inline unsigned remainingSpaceBytes() const;

    SDBEntry *entries_;
    unsigned heap_offset_; // the heap grows backwards from the end of the block.
    // The end of last entry offset_ bytes from the end of the structure.
    char *end_of_block_;

    template<class A>
    friend std::ostream &operator<<(std::ostream &os, const SparseDataBlock<A> &block);

    friend std::ostream &operator<<(std::ostream &os, const SparseDataBlock<num_t> &block);
  };

  template<class T>
  std::ostream &operator<<(std::ostream &os, const SparseDataBlock<T> &block) {
    os << "SparseDataBlock[" << block.getNumRows() << ", " << block.getNumColumns() << "]" << std::endl;
  }

  template<class T>
  T* SparseDataBlock<T>::operator()(unsigned row, unsigned col) {
    return get(row, col);
  }

  template<class T>
  bool SparseDataBlock<T>::appendRow(const svector<T> &row) {
    DCHECK(this->initializing_);

    if (remainingSpaceBytes() < (sizeof(SDBEntry) + row.sizeBytes())) {
      return false;
    }

    heap_offset_ += row.sizeBytes();
    SDBEntry *entry = entries_ + this->num_rows_;
    entry->offset_ = heap_offset_;
    entry->size_ = row.numElements();
    row.copyTo(end_of_block_ - heap_offset_);

    this->num_rows_++;
    this->num_columns_ = this->num_columns_ >= row.size() ? this->num_columns_ : row.size();
    return true;
  }

  template<class T>
  void SparseDataBlock<T>::getRowVector(const int row, exvector<T> *vec) const {
    DCHECK_LT(row, this->num_rows_) << "Row index out of range.";
  //  DCHECK(dynamic_cast<se_vector<float_t> *>(vec) != nullptr);

    SDBEntry const &entry = entries_[row];
    vec->setMemory(entry.size_, end_of_block_ - entry.offset_);
  }

  template<class T>
  T* SparseDataBlock<T>::get(unsigned row, unsigned col) const {
    DCHECK_LT(row, this->num_rows_) << "Row index out of range.";
    DCHECK_LT(col, this->num_columns_) << "Column index out of range.";

    SDBEntry const &entry = entries_[row];
    svector<T> vec(entry.size_, end_of_block_ - entry.offset_);
    T* value = vec.get(col);
    if (value == nullptr) {
      return 0;
    } else {
      return value;
    }
  }

  template<class T>
  void SparseDataBlock<T>::trimRows(int rows) {
    DCHECK_LT(rows, this->num_rows_);
    for (int row = this->num_rows_ - rows; row < this->num_rows_; row++) {
      SDBEntry *entry = entries_ + this->num_rows_;
      heap_offset_ -= entry->size_;
    }
    this->num_rows_ -= rows;
  }

  template<class T>
  unsigned SparseDataBlock<T>::remainingSpaceBytes() const {
    return this->block_size_bytes_ - (heap_offset_ + sizeof(SDBEntry) * this->num_rows_);
  }

  /**
   * @return Number of non zero elements. Does not include classification column.
   */
  template<class T>
  int SparseDataBlock<T>::numNonZeroElements() const {
    int nnz = 0;
    for (int i = 0; i < this->num_rows_; i++) {
      const SDBEntry & sdbe = entries_[i];
      nnz += sdbe.size_;
    }
    return nnz;
  }

  /**
   * Get the maximum number of columns in a set of blocks.
   * @param blocks
   * @return
   */
  template<class T>
  int maxColumns(std::vector<SparseDataBlock<T>*> blocks) {
    int max_columns = 0;
    for (int i = 0; i < blocks.size(); ++i) {
      if (max_columns < blocks[i]->getNumColumns()) {
        max_columns = blocks[i]->getNumColumns();
      }
    }
    return max_columns;
  }

  /**
   * Generates a sparse matrix full of random data with an even distribution of sparsity.
   *
   * Sparsity is the number of non-zero elements in a row.
   *
   * @return a caller-owned sparse datablock.
   */
  static SparseDataBlock<num_t>* GetRandomSparseDataBlock(int blockSizeBytes, int numColumns, double sparsity) {
    return new SparseDataBlock<num_t>(blockSizeBytes, numColumns, sparsity);
  }
}

#endif //OBAMADB_SPARSEDATABLOCK_H
