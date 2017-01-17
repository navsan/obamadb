#ifndef OBAMADB_DATAVIEW_H_
#define OBAMADB_DATAVIEW_H_

#include "storage/DataBlock.h"
#include "storage/exvector.h"
#include "storage/SparseDataBlock.h"
#include "storage/StorageConstants.h"

#include <algorithm>
#include <vector>

namespace obamadb {

  class DataView {
  public:
    DataView(std::vector<SparseDataBlock<num_t> const *> blocks)
      : blocks_(blocks), current_block_(0),current_idx_(0) {}

    DataView() : blocks_(), current_block_(0), current_idx_(0) {}

    inline bool getNext(svector<num_t> * row) {
      if (current_idx_ < blocks_[current_block_]->num_rows_) {
        blocks_[current_block_]->getRowVectorFast(current_idx_++, row);
        return true;
      } else if (current_block_ < blocks_.size() - 1) {
        current_block_++;
        current_idx_ = 0;
        return getNext(row);
      }

      return false;
    }

    void appendBlock(SparseDataBlock<num_t> const * block) {
      blocks_.push_back(block);
    }

    void clear() {
      blocks_.clear();
    }

    inline void reset() {
      current_block_ = 0;
      current_idx_ = 0;
    }

  protected:

    std::vector<SparseDataBlock<num_t> const *> blocks_;
    int current_block_;
    int current_idx_;
  };
}

#endif //OBAMADB_DATAVIEW_H_
