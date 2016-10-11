#ifndef OBAMADB_DATABLOCK_H_
#define OBAMADB_DATABLOCK_H_

#include "storage/StorageConstants.h"

#include <cstdint>
#include <cstdlib>
#include <functional>

#include <glog/logging.h>
#include <iomanip>

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
     * Make a vertical slice of the data given the column indices to select.
     *
     * @param start_index inclusive column index.
     * @param end_index exclusive column index. Negative indices are treated as
     *              distances from the end index.
     * @return A new datablock containing the specified columns and all of the rows.
     */
    DataBlock* slice(std::int32_t start_index, std::int32_t end_index) const;

    /**
     * Returns a new datablock containing only rows whose columns pass the filter function.
     * @param filter A function which takes a double attribute value and returns whether or not
     *              this attribute is selected.
     * @param col
     * @return
     */
    DataBlock* filter(std::function<bool(double)> &filter, unsigned col) const;

    void matchRows(std::function<bool(double)> &filter, unsigned col, std::vector<unsigned> &matches) const;

    DataBlock* sliceRows(const std::vector<unsigned>& rows) const;

    double get(unsigned row, unsigned col) const {
      return store_[(row * width_) + col];
    }

    inline double* getStore() const {
      return store_;
    }

    void setWidth(std::uint32_t width) {
      width_ = width;
    }

    std::uint32_t getWidth() const {
      return width_;
    }

    std::uint32_t getNumRows() const {
      return elements_/width_;
    }

  private:

    std::uint32_t width_;   // attributes in row
    std::uint32_t height_;  // number of rows
    std::uint32_t elements_;
    std::uint32_t max_elements_;
    double *store_;

    friend std::ostream& operator<<(std::ostream& os, const DataBlock& block);
  };

  std::ostream& operator<<(std::ostream& os, const DataBlock& block);

}  // namespace obamadb

#endif //OBAMADB_DATABLOCK_H_
