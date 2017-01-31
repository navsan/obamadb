#ifndef OBAMADB_UNORDEREDMATRIX_H
#define OBAMADB_UNORDEREDMATRIX_H

#include "storage/StorageConstants.h"
#include "glog/logging.h"

#include <iostream>

#define USE_SPOOKY_MATRIX 1

namespace obamadb {

  struct MatrixEntry {
    MatrixEntry()
      : row(0),
        column(0),
        value(0) { }

    MatrixEntry(int row, int col, num_t val)
      : row(row),
        column(col),
        value(val) { }

    MatrixEntry(MatrixEntry const & other)
      : row(other.row),
        column(other.column),
        value(other.value) { }

    int row;
    int column;
    num_t value;
  };
#ifdef USE_SPOOKY_MATRIX
  /**
   * A list of row/column/value triples. They are in no particular order.
   * This version manages its own buffer. It works 'spookily' well.
   */
  class UnorderedMatrix {
  public:
    UnorderedMatrix() : rows_(0), columns_(0), size_(0), maxSize_(1<<10), entries_(nullptr) {
      entries_ = new MatrixEntry[maxSize_];
    }

    ~UnorderedMatrix() {
      delete[] entries_;
    }

    void append(int row, int col, num_t val) {
      if (size_ == maxSize_) {
        doubleSize();
      }

      entries_[size_] = MatrixEntry(row,col,val);
      size_++;
      if (row > rows_) {
        rows_ = row;
      }
      if (col > columns_) {
        columns_ = col;
      }
    }

    MatrixEntry const & get(int index) const {
      DCHECK_LT(index, size_);
      return entries_[index];
    }

    MatrixEntry const * get(int row, int column) const {
      for (int i = 0; i < size_; i++) {
        if (entries_[i].row == row && entries_[i].column == column) {
          return &entries_[i];
        }
      }
      return nullptr;
    }

    int numElements() const {
      return size_;
    }

    int numRows() const {
      return rows_;
    }

    int numColumns() const {
      return columns_;
    }

    friend std::ostream& operator<<(std::ostream& os, const UnorderedMatrix& matrix);

  private:
    void doubleSize() {
      MatrixEntry * newEntries = new MatrixEntry[maxSize_ * 2];
      memcpy(newEntries, entries_, sizeof(MatrixEntry) * maxSize_);
      delete [] entries_;
      entries_ = newEntries;
      maxSize_ *= 2;
    }

    int rows_;
    int columns_;
    std::size_t size_;
    std::size_t maxSize_;
    MatrixEntry* entries_;

  };
#else
/**
 * A list of row/column/value triples. They are in no particular order.
 */
  class UnorderedMatrix {
  public:

    void append(int row, int col, num_t val) {
      entries_.push_back(MatrixEntry(row,col,val));
      if (row > rows_) {
        rows_ = row;
      }
      if (col > columns_) {
        columns_ = col;
      }
    }

    MatrixEntry const & get(int index) const {
      DCHECK_LT(index, entries_.size());
      return entries_[index];
    }

    MatrixEntry const * get(int row, int column) const {
      for (int i = 0; i < entries_.size(); i++) {
        if (entries_[i].row == row && entries_[i].column == column) {
          return &entries_[i];
        }
      }
      return nullptr;
    }

    int numElements() const {
      return entries_.size();
    }

    int numRows() const {
      return rows_;
    }

    int numColumns() const {
      return columns_;
    }

    friend std::ostream& operator<<(std::ostream& os, const UnorderedMatrix& matrix);

  private:
    int rows_;
    int columns_;
    std::vector<MatrixEntry> entries_;

  };
#endif
}

#endif //OBAMADB_UNORDEREDMATRIX_H
