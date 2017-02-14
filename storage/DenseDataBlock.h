#ifndef OBAMADB_DENSEDATABLOCK_H
#define OBAMADB_DENSEDATABLOCK_H

#include "storage/exvector.h"
#include "storage/DataBlock.h"
#include "storage/StorageConstants.h"
#include "storage/Utils.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <functional>

#include <glog/logging.h>

namespace obamadb {

  template<class T>
  class DenseDataBlock : public DataBlock<T> {
  public:
    DenseDataBlock(unsigned numRows,
                   unsigned numColumns) :
      DataBlock<T>(sizeof(T) * numRows * (numColumns + 1)),
      maxElements(numRows * (numColumns + 1)),
      maxRows(numRows){
      this->num_rows_ = 0;
      this->num_columns_ = numColumns;
    }

    DenseDataBlock(unsigned size_bytes) :
      DataBlock<T>(size_bytes),
      maxElements(size_bytes / (sizeof(T))) {}

    DenseDataBlock() : DenseDataBlock<T>(kStorageBlockSize) {}

    /**
     * Use this function while initializing to pack the block.
     * @param row Row to append.
     * @return True if the append succeeded, false if the block is full.
     */
    bool appendRow(const dvector<T> &row) {
      DCHECK_EQ(row.size(), this->num_columns_);
      if (0 <= (this->maxElements - numElements()) - this->sizeRow()) {
        memcpy(this->store_ + (this->sizeRow() * this->num_rows_), row.values_, sizeof(T) * this->sizeRow());
        this->num_rows_++;
        return true;
      }
      return false;
    }

    /**
     * Blocks which have been finalized can no longer have rows appended to them.
     */
    void finalize() {
      this->initializing_ = false;
    }

    /**
    * Sets the memory of a given example vector to the memory of the block associated with the requested row.
    * @param row
    * @param src
    */
    void getRowVector(int row, exvector<T>* src) const override {
      DCHECK(src->getType() == exvectorType::kDense);
      src->setMemory(this->num_columns_, this->store_ + (this->sizeRow() * row));
    }

    inline void getRowVectorFast(int row, dvector<T>* src) const {
      DCHECK(!src->ownsMemory());
      src->values_ = this->store_ + (this->sizeRow() * row);
      src->class_ = src->values_ + this->num_columns_;
      src->num_elements_ = this->num_columns_;
    }

    DataBlockType getDataBlockType() const override {
      return DataBlockType::kDense;
    }

    /**
     * @param row
     * @param col
     * @return  The value stored at a particular index.
     */
    T* get(unsigned row, unsigned col) const override {
      DCHECK_GT(this->num_rows_, row);
      DCHECK_GT(this->num_columns_, col);
      return this->store_ + (row * this->sizeRow() + col);
    }

    T* operator()(unsigned row, unsigned col) override {
      return get(row, col);
    }

    inline int numElements() const {
      return this->num_rows_ * this->sizeRow();
    }

    /**
     * Rows are the number of attribute columns plus a classification.
     */
    inline int sizeRow() const {
      return this->num_columns_ + 1;
    }

    void randomize() {
      dvector<T> row_vector(0, nullptr);
      this->num_rows_ = this->maxRows;
      for (int row = 0; row < this->num_rows_; row++) {
        this->getRowVectorFast(row, &row_vector);
        for (int column = 0; column < this->num_columns_; column++) {
          row_vector.values_[column] = (T) rand() / (T)INT_MAX; // TODO: this will not work with non-float types
        }
      }
    }

    template<class TT>
    friend std::ostream &operator<<(std::ostream &os, const DenseDataBlock<TT> &block);

    int maxElements; // includes classifications
    int maxRows;
  };


template<class T>
std::ostream &operator<<(std::ostream &os, const DenseDataBlock<T> &block) {
  os << "DenseDataBlock[" << block.getNumRows() << ", " << block.getNumColumns()
     << "] size mb: " << (block.block_size_bytes_ / 1e6);
  return os;
}

}

#endif //OBAMADB_DENSEDATABLOCK_H
