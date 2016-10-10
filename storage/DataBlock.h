#ifndef OBAMADB_DATABLOCK_H_
#define OBAMADB_DATABLOCK_H_

#include "storage/StorageConstants.h"

#include <cstdint>
#include <cstdlib>

#include <glog/logging.h>

namespace obamadb {

  class DataBlock {
  public:
    DataBlock() :
      width_(0),
      height_(0),
      elements_(0),
      max_elements_(kStorageBlockSize/sizeof(double)),
      store_(nullptr) {
      store_ = (double*) new char[kStorageBlockSize];
    }

    ~DataBlock() {
      delete store_;
    }

    void append(double *element) {
      DCHECK_GT(max_elements_, elements_);
      store_[elements_++] = *element;
    }

    std::uint32_t getSize() const {
      return elements_;
    }

    /**
     *
     * @param start_index inclusive
     * @param end_index exclusive
     * @return A new datablock containing the specified columns and all of the rows.
     */
    DataBlock* slice(std::int32_t start_index, std::int32_t end_index) const;

    double get(unsigned row, unsigned col) const {
      return store_[(row * width_) + col];
    }

    double* getStore() const {
      return store_;
    }

    void setWidth(std::uint32_t width) {
      width_ = width;
    }

    std::uint32_t getWidth() const {
      return width_;
    }


  private:

    std::uint32_t width_;   // attributes in row
    std::uint32_t height_;  // number of rows
    std::uint32_t elements_;
    std::uint32_t max_elements_;
    double *store_;
  };

}



#endif //OBAMADB_DATABLOCK_H_
