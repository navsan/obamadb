#ifndef OBAMADB_UNORDEREDMATRIX_H
#define OBAMADB_UNORDEREDMATRIX_H

#include "storage/StorageConstants.h"
#include "glog/logging.h"

#include <iostream>

namespace obamadb {

  struct MatrixEntry {
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
}

#endif //OBAMADB_UNORDEREDMATRIX_H
