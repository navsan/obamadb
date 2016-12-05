#ifndef OBAMADB_DATABLOCK_H_
#define OBAMADB_DATABLOCK_H_

#include "storage/exvector.h"
#include "storage/StorageConstants.h"
#include "storage/Utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <functional>

#include <glog/logging.h>

namespace obamadb {

  enum class DataBlockType {
    kDense,
    kSparse
  };

  template<class T>
  class DataBlock {
  public:
    DataBlock(unsigned numRows, unsigned numColumns) :
      num_columns_(numColumns),
      num_rows_(0),
      block_size_bytes_(kStorageBlockSize),
      store_(nullptr),
      initializing_(true) {
        std::uint64_t requested_size = ((numColumns + 1) * numRows) * sizeof(T);
        if(requested_size > block_size_bytes_) {
            block_size_bytes_ = requested_size;
        }
        reinterpret_cast<T*>(new char[block_size_bytes_]);
    }

    DataBlock(unsigned size_bytes) :
      num_columns_(0),
      num_rows_(0),
      block_size_bytes_(size_bytes),
      store_(reinterpret_cast<T*>(new char[size_bytes])),
      initializing_(true) {}

    DataBlock() : DataBlock(kStorageBlockSize) {}

    ~DataBlock() {
      delete store_;
    }

    /**
     * @return The total number of elements stored. (Rows * columns).
     */
    virtual std::uint32_t getSize() const {
      return num_columns_ * num_rows_;
    }

    virtual std::uint32_t getNumColumns() const {
      return num_columns_;
    }

    virtual std::uint32_t getNumRows() const {
      return num_rows_;
    }

    /**
     * Sets the memory of a given example vector to the memory of the block associated with the requested row.
     * @param row
     * @param src
     */
    virtual void getRowVector(int row, exvector<T>* src) const = 0;

    virtual DataBlockType getDataBlockType() const = 0;

    /**
     * @param row
     * @param col
     * @return  The value stored at a particular index.
     */
    virtual T* get(unsigned row, unsigned col) const = 0;

    virtual T* operator()(unsigned row, unsigned col) = 0;

    template<class A>
    friend std::ostream& operator<<(std::ostream& os, const DataBlock<A>& block);

    std::uint32_t num_columns_;
    std::uint32_t num_rows_;
    std::uint32_t block_size_bytes_;
    T* store_;
    bool initializing_;

    DISABLE_COPY_AND_ASSIGN(DataBlock);
  };

}  // namespace obamadb

#endif //OBAMADB_DATABLOCK_H_
