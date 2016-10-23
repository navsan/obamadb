#include "DataBlock.h"

#include <cstring>
#include <iostream>
#include <memory>

namespace obamadb {
  DenseDataBlock* DenseDataBlock::slice(std::int32_t start_index, std::int32_t end_index) const {
    if (end_index < 0) {
      end_index = num_columns_ - end_index;
    }
    if (end_index <= start_index || start_index < 0 || end_index > num_columns_) {
      return nullptr;
    }

    DenseDataBlock *new_block = new DenseDataBlock();
    double *src_cursor = store_ + start_index;
    unsigned copy_width = end_index - start_index;

    new_block->setWidth(end_index - start_index);
    double *dst_cursor = new_block->store_;
    double *src_end = store_ + elements_;
    while(src_cursor < src_end) {
      memcpy(dst_cursor, src_cursor, copy_width * sizeof(double));
      dst_cursor = dst_cursor + copy_width;
      src_cursor = src_cursor + num_columns_;
    }

    new_block->elements_ = (elements_/num_columns_)  * (end_index - start_index);

    return new_block;
  }

  void DenseDataBlock::matchRows(std::function<bool(double)> &filter, unsigned col, std::vector<unsigned> &matches) const {
    unsigned total_rows = elements_/num_columns_;
    double *cursor = store_ + col;
    unsigned row = 0;
    while(total_rows > row) {
      if (filter(*cursor)) {
        matches.push_back(row);
      }
      row++;
      cursor += num_columns_;
    }
  }

  DenseDataBlock* DenseDataBlock::sliceRows(const std::vector<unsigned>& rows) const {
    std::unique_ptr<DenseDataBlock> data_block(new DenseDataBlock());
    data_block->setWidth(num_columns_);
    double *dst_cursor = data_block->getStore();
    for (unsigned row : rows) {
      DCHECK_LT(row, elements_/num_columns_);
      double *row_ptr = store_ + num_columns_ * row;
      std::memcpy(dst_cursor, row_ptr, num_columns_ * sizeof(double));
      dst_cursor += num_columns_;
    }
    data_block->elements_ = rows.size() * num_columns_;
    return data_block.release();
  }

  DenseDataBlock* DenseDataBlock::filter(std::function<bool(double)> &filter, unsigned col) const {
    std::vector<unsigned> row_matches;
    matchRows(filter, col, row_matches);
    return sliceRows(row_matches);
  }

  std::ostream& operator<<(std::ostream& os, const DenseDataBlock& block) {
    os << "[" << block.getNumRows() << ", " << block.num_columns_ << "]\n";
    os << std::setprecision(2);
    const unsigned col_max = 10;
    const unsigned row_max = 10;
    for (unsigned row = 0; row < block.getNumRows(); ++row) {
      for(unsigned col = 0; col < block.num_columns_; ++col) {
        os << block.get(row, col) << " ";
        if (col == col_max/2 && block.num_columns_ > col_max) {
          col = block.num_columns_ - col_max/2;
          os << "... ";
        }
      }
      os << "\n";
      if (row == row_max/2 && block.getNumRows() > row_max) {
        row = block.getNumRows() - row_max/2;
        os << "...\n";
      }
    }
    return os;
  }

  double *DenseDataBlock::getRow(unsigned row) const {
    // TODO: make a DataBlock builder class which takes control of all unsafe editting of this class.
    // DCHECK_LT(row, getNumRows());

    return &store_[row * num_columns_];
  }

  std::ostream& operator<<(std::ostream &os, const DataBlock &block) {
    os << "DataBlock[" << std::to_string(block.num_rows_) << " " << std::to_string(block.num_columns_) << "]";
    return os;
  }
}
