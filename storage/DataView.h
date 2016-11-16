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
    DataView(std::vector<SparseDataBlock<float_t> const *> blocks)
      : blocks_(blocks), current_block_(0),current_idx_(0) {}

    DataView() : blocks_(), current_block_(0), current_idx_(0) {}

    inline bool getNext(svector<float_t> * row) {
//      if (blocks_.size() == 0) {
//        return false;
//      }

      if (current_idx_ < blocks_[current_block_]->num_rows_) {
        //blocks_[current_block_]->getRowVector(current_idx_++, row);
        blocks_[current_block_]->getRowVectorFast(current_idx_++, row);
        return true;
      } else if (current_block_ < blocks_.size() - 1) {
        current_block_++;
        current_idx_ = 0;
        return getNext(row);
      }

      return false;
    }

    void appendBlock(SparseDataBlock<float_t> const * block) {
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

    std::vector<SparseDataBlock<float_t> const *> blocks_;
    int current_block_;
    int current_idx_;
  };

//  class ShuffledDataView : public DataView {
//  public:
//    ShuffledDataView() : DataView(), index_shuffle_() {
//    }
//
//    bool getNext(e_vector<float_t> * row) override {
//      if (blocks_.size() == 0) {
//        return false;
//      }
//
//      if (current_block_ == 0 && current_idx_ == 0 && blocks_.size() > 0) {
//        reshuffle(blocks_[0]->getNumRows());
//      }
//
//      if (current_idx_ < blocks_[current_block_]->getNumRows()) {
//        blocks_[current_block_]->getRowVector(index_shuffle_[current_idx_++], row);
//        return true;
//      } else if (current_block_ < blocks_.size() - 1) {
//        current_block_++;
//        current_idx_ = 0;
//
//        reshuffle(blocks_[current_block_]->getNumRows());
//        return getNext(row);
//      }
//      return false;
//    }
//
//    void reset() override {
//      current_block_ = 0;
//      current_idx_ = 0;
//      reshuffle(0);
//    }
//
//  protected:
//    void reshuffle(int indices) {
//      // between rounds of getNext, we do not reshuffle.
//      if(indices == index_shuffle_.size()) {
//        return;
//      }
//
//      if(index_shuffle_.size() > indices) {
//        index_shuffle_.clear();
//      }
//
//      if(index_shuffle_.size() < indices) {
//        int next = index_shuffle_.size();
//        for (int i = next; i < indices; i++){
//          index_shuffle_.push_back(i);
//        }
//      }
//
//      if (index_shuffle_.size() > 0) {
//        std::random_shuffle(index_shuffle_.begin(), index_shuffle_.end());
//      }
//    }
//
//  private:
//    std::vector<int> index_shuffle_;
//  };
}

#endif //OBAMADB_DATAVIEW_H_
