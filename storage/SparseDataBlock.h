#ifndef OBAMADB_SPARSEDATABLOCK_H
#define OBAMADB_SPARSEDATABLOCK_H

#include "storage/DataBlock.h"
#include "storage/exvector.h"
#include "storage/StorageConstants.h"
#include "storage/Utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <functional>

#include <glog/logging.h>

namespace obamadb {

  template<class T> class SparseDataBlock;

  template <typename T>
  std::ostream &operator<<(std::ostream &os, const SparseDataBlock<T> &block);

  std::ostream &operator<<(std::ostream &os, const SparseDataBlock<int_t> &block);


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

    T getClassification(int row) const;

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

    friend std::ostream &operator<<(std::ostream &os, const SparseDataBlock<int_t> &block);
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

  template<class T>
  T SparseDataBlock<T>::getClassification(int row) const {
    svector<T> t_row(0, nullptr);
    this->getRowVector(row, &t_row);
    return *t_row.class_;
  }

  /**
   * Generates a random projection matrix with dimensions k x dimension. Entries of this matrix
   * are in {-1, 0, 1} and appear with probablities { 1/2sqrt(dimension), 1 - 1/sqrt(dimension),
   * 1/sqrt(dimension) }.
   *
   * Ordinarily, we'd return the transpose of this matrix so that the dimensions line up
   * with A*R, but instead we return the transpose so that we can compute the multiplication
   * faster (row,rowwise) instead of (row,columnwise).
   *
   * @param dimension The dimension (n columns) of the data set which we are compressing.
   * @param k Determines the factor of compression for this data set (k << dimension).
   * @return A caller-owned DataBlock.
   */
  static SparseDataBlock<signed char>* GetRandomProjectionMatrix(int dimension, int k) {
    // Estimate how large of a block we will need so that we can fit the entire thing in one go.
    int est_size = ((dimension * k) * (1.0/sqrt(dimension)) * (sizeof(char) + sizeof(int)))          // size of total number of svectors
                   + (dimension * (sizeof(SparseDataBlock<signed char>::SDBEntry) + sizeof(char)));  // size of total number of entries and additional classification in the svector
    SparseDataBlock<signed char> *pdb = new SparseDataBlock<signed char>(est_size);
    pdb->num_columns_ = dimension;
    QuickRandom qr;
    std::uint32_t max_uint32 = 0;
    max_uint32 -= 1;
    std::uint32_t bound_neq1 = (1.0/(2.0*sqrt(dimension))) * max_uint32;
    std::uint32_t bound_pos1 = bound_neq1 * 2;
    for(int i = 0; i < k; i++) {
      svector<signed char> row;
      for (int j = 0; j < dimension; j++) {
        std::uint32_t rand = qr.nextInt32();
        if(rand < bound_neq1) {
          row.push_back(j, -1);
        } else if (rand < bound_pos1) {
          row.push_back(j, 1);
        }
      }
      DCHECK_GE(dimension, row.size());
     // DCHECK_NE(0, row.num_elements_);
      CHECK(pdb->appendRow(row)); // If false, we ran out of room.
    }
    DCHECK_EQ(k, pdb->num_rows_);
    DCHECK_EQ(dimension, pdb->num_columns_);
    return pdb;
  }

    /**
     * Generates a sparse matrix full of random data with an even distribution of sparsity.
     *
     * Sparsity is the number of non-zero elements in a row.
     *
     * @return a caller-owned sparse datablock.
     */
    static SparseDataBlock<int_t>* GetRandomSparseDataBlock(int blockSizeBytes, int numColumns, double sparsity) {
      SparseDataBlock<int_t>* dataBlock = new SparseDataBlock<int_t>(blockSizeBytes);
      svector<int_t> row_vector;
      QuickRandom qr;
      double avgElementsPerRow = (1.0 - sparsity) * numColumns;
      int elementWindowSize = ((double)numColumns) / avgElementsPerRow;
      int elementWindows = std::ceil(avgElementsPerRow);
      int_t positive = 1.0;
      int_t negative = -1.0;
      do {
        row_vector.clear();
        bool isPositive = qr.nextFloat() > 0;
        row_vector.setClassification(isPositive ? &positive : &negative);
        for (int i = 0; i < elementWindows; i++) {
          int randi = qr.nextInt32() % elementWindowSize;
          int index = (i * elementWindowSize) + randi;
          if (index < numColumns) {
            int_t randf = std::abs(qr.nextFloat());
            // The data should end up being perfectly seperable.
            if ((isPositive && index % 2 == 1 )
                || (!isPositive && index % 2 == 0)) {
              randf *= -1;
            }
            row_vector.push_back(index, randf);
          }
        }
      } while(dataBlock->appendRow(row_vector));
      return dataBlock;
    }
}

#endif //OBAMADB_SPARSEDATABLOCK_H
