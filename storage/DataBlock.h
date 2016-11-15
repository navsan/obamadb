#ifndef OBAMADB_DATABLOCK_H_
#define OBAMADB_DATABLOCK_H_

#include "storage/StorageConstants.h"
#include "storage/Utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>

#include <glog/logging.h>
#include <iomanip>

namespace obamadb {

  enum class DataBlockType {
    kSparse
  };

  template<class T>
  class DataBlock {
  public:
    DataBlock(unsigned numRows, unsigned numColumns) :
      num_columns_(numRows),
      num_rows_(numColumns),
      block_size_bytes_(kStorageBlockSize),
      store_(reinterpret_cast<T*>(new char[kStorageBlockSize])) {}

    DataBlock(unsigned size_bytes) :
      num_columns_(0),
      num_rows_(0),
      block_size_bytes_(size_bytes),
      store_(reinterpret_cast<T*>(new char[size_bytes])) {}

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
    virtual void getRowVector(int row, e_vector<T>* src) const = 0;

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

//  protected:
    std::uint32_t num_columns_;
    std::uint32_t num_rows_;
    std::uint32_t block_size_bytes_;
    T* store_;

    DISABLE_COPY_AND_ASSIGN(DataBlock);
  };

}  // namespace obamadb

#endif //OBAMADB_DATABLOCK_H_
