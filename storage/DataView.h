#ifndef OBAMADB_DATAVIEW_H_
#define OBAMADB_DATAVIEW_H_

#include <algorithm>
#include <vector>

#include "storage/DataBlock.h"

namespace obamadb {

  class DataView {
  public:
    DataView(std::vector<DenseDataBlock const *> blocks)
      : blocks_(blocks), current_block_(0),current_idx_(0) {}

    DataView() : blocks_(), current_block_(0), current_idx_(0) {}


    virtual double * getNext() {
      if (blocks_.size() == 0) {
        return nullptr;
      }

      if (current_idx_ < blocks_[current_block_]->getNumRows()) {
        return blocks_[current_block_]->getRow(current_idx_++);
      } else if (current_block_ < blocks_.size() - 1) {
        current_block_++;
        current_idx_ = 0;
        return getNext();
      } else {
        return nullptr;
      }
    }

    void appendBlock(DenseDataBlock const * block) {
      blocks_.push_back(block);
    }


    virtual void reset() {
      current_block_ = 0;
      current_idx_ = 0;
    }

  protected:

    std::vector<DenseDataBlock const *> blocks_;
    int current_block_;
    int current_idx_;
  };

  class ShuffledDataView : public DataView {
  public:
    ShuffledDataView() : DataView(), index_shuffle_() {
    }

    double * getNext() override {
      if (blocks_.size() == 0) {
        return nullptr;
      }

      if (current_block_ == 0 && current_idx_ == 0 && blocks_.size() > 0) {
        reshuffle(blocks_[0]->getNumRows());
      }

      if (current_idx_ < blocks_[current_block_]->getNumRows()) {
        return blocks_[current_block_]->getRow(index_shuffle_[current_idx_++]);
      } else if (current_block_ < blocks_.size() - 1) {
        current_block_++;
        current_idx_ = 0;

        reshuffle(blocks_[current_block_]->getNumRows());
        return getNext();
      } else {
        return nullptr;
      }
    }

    void reset() override {
      current_block_ = 0;
      current_idx_ = 0;
      reshuffle(0);
    }

  protected:
    void reshuffle(int indices) {
      // between rounds of getNext, we do not reshuffle.
      if(indices == index_shuffle_.size()) {
        return;
      }

      if(index_shuffle_.size() > indices) {
        index_shuffle_.clear();
      }

      if(index_shuffle_.size() < indices) {
        int next = index_shuffle_.size();
        for (int i = next; i < indices; i++){
          index_shuffle_.push_back(i);
        }
      }

      if (index_shuffle_.size() > 0) {
        std::random_shuffle(index_shuffle_.begin(), index_shuffle_.end());
      }
    }

  private:
    std::vector<int> index_shuffle_;
  };
}

#endif //OBAMADB_DATAVIEW_H_
