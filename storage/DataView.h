#ifndef OBAMADB_DATAVIEW_H_
#define OBAMADB_DATAVIEW_H_

#include "storage/DataBlock.h"
#include "storage/exvector.h"
#include "storage/SparseDataBlock.h"
#include "storage/StorageConstants.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace obamadb {
class DataView {
 public:
  DataView(std::vector<SparseDataBlock<num_t> const *> blocks)
      : blocks_(blocks), current_block_(0), current_idx_(0) {}

  DataView() : blocks_(), current_block_(0), current_idx_(0) {}

  inline bool getNext(svector<num_t> *row) {
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

  void appendBlock(SparseDataBlock<num_t> const *block) {
    blocks_.push_back(block);
  }

  void clear() { blocks_.clear(); }

  inline void reset() {
    current_block_ = 0;
    current_idx_ = 0;
  }

 protected:
  std::vector<SparseDataBlock<num_t> const *> blocks_;
  int current_block_;
  int current_idx_;
};

/**
 * Allocates Datablocks to DataViews. Dataviews will then be given to threads in
 * the form of tasks.
 * @param num_threads How many dataviews to allocate and distribute amongst.
 * @param data_blocks The set of training data.
 * @param views Output vector of Dataviews.
 */
template <class T>
void allocateBlocks(const int num_threads,
                    const std::vector<SparseDataBlock<T> *> &data_blocks,
                    std::vector<std::unique_ptr<DataView>> &views) {
  CHECK(views.size() == 0) << "Only accepts empty view vectors";
  CHECK_GE(data_blocks.size(), num_threads)
      << "Partitioned data would not distribute to all threads. "
      << "Use at most " << data_blocks.size() << " threads.";

  for (int i = 0; i < data_blocks.size(); i++) {
    if (i < num_threads) {
      views.push_back(std::unique_ptr<DataView>(new DataView()));
    }
    SparseDataBlock<T> const *dbptr = data_blocks[i];
    views[i % num_threads]->appendBlock(dbptr);
  }
}
} // namespace obamadb

#endif //OBAMADB_DATAVIEW_H_
