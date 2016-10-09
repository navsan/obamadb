//
// Created by marc on 10/8/16.
//

#include <cstring>
#include "DataBlock.h"

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
}
