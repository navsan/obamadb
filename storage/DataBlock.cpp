#include "DataBlock.h"

#include <cstring>
#include <iostream>
#include <memory>

namespace obamadb {
  DataBlock* DataBlock::slice(std::int32_t start_index, std::int32_t end_index) const {
    if (end_index < 0) {
      end_index = width_ - end_index;
    }
    if (end_index <= start_index || start_index < 0 || end_index > width_) {
      return nullptr;
    }

    DataBlock *new_block = new DataBlock();
    double *src_cursor = store_ + start_index;
    unsigned copy_width = end_index - start_index;

    new_block->setWidth(end_index - start_index);
    double *dst_cursor = new_block->store_;
    double *src_end = store_ + elements_;
    while(src_cursor < src_end) {
      memcpy(dst_cursor, src_cursor, copy_width * sizeof(double));
      dst_cursor = dst_cursor + copy_width;
      src_cursor = src_cursor + width_;
    }

    new_block->elements_ = (elements_/width_)  * (end_index - start_index);

    return new_block;
  }

  void DataBlock::matchRows(std::function<bool(double)> &filter, unsigned col, std::vector<unsigned> &matches) const {
    unsigned total_rows = elements_/width_;
    double *cursor = store_ + col;
    unsigned row = 0;
    while(total_rows > row) {
      if (filter(*cursor)) {
        matches.push_back(row);
      }
      row++;
      cursor += width_;
    }
  }

  DataBlock* DataBlock::sliceRows(const std::vector<unsigned>& rows) const {
    std::unique_ptr<DataBlock> data_block(new DataBlock());
    data_block->setWidth(width_);
    double *dst_cursor = data_block->getStore();
    for (unsigned row : rows) {
      DCHECK_LT(row, elements_/width_);
      double *row_ptr = store_ + width_ * row;
      std::memcpy(dst_cursor, row_ptr, width_ * sizeof(double));
      dst_cursor += width_;
    }
    data_block->elements_ = rows.size() * width_;
    return data_block.release();
  }

  DataBlock* DataBlock::filter(std::function<bool(double)> &filter, unsigned col) const {
    std::vector<unsigned> row_matches;
    matchRows(filter, col, row_matches);
    return sliceRows(row_matches);
  }

  std::ostream& operator<<(std::ostream& os, const DataBlock& block) {
    os << "[" << block.getNumRows() << ", " << block.width_ << "]\n";
    os << std::setprecision(2);
    const unsigned col_max = 10;
    const unsigned row_max = 10;
    for (unsigned row = 0; row < block.getNumRows(); ++row) {
      for(unsigned col = 0; col < block.width_; ++col) {
        os << block.get(row, col) << " ";
        if (col == col_max/2 && block.width_ > col_max) {
          col = block.width_ - col_max/2;
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

  double *DataBlock::getRow(unsigned row) const {
    DCHECK_LT(row, getNumRows());
    return &store_[row * width_];
  }
}
