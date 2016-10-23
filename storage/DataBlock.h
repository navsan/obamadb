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
    DataBlock(unsigned num_columns) :
      num_columns_(num_columns),
      num_rows_(0),
      store_((double*) new char[kStorageBlockSize]) {}

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

    virtual double *getRow(unsigned row) const {
      DCHECK_LT(row, num_rows_);

      return store_ + row * num_columns_;
    }

    /**
     * @param row
     * @param col
     * @return  The value stored at a particular index.
     */
    virtual double get(unsigned row, unsigned col) const {
      DCHECK_LT(row, getNumRows()) << "Row index out of range.";
      DCHECK_LT(col, getNumColumns()) << "Column index out of range.";

      return store_[(row * num_columns_) + col];
    }

    virtual double operator()(unsigned row, unsigned col) {
      return get(row,col);
    }

    /**
     * @return The pointer to the raw underlying data store.
     */
    virtual double* getStore() const {
      return store_;
    }

  protected:
    std::uint32_t num_columns_;
    std::uint32_t num_rows_;
    double *store_;

    friend std::ostream& operator<<(std::ostream& os, const DataBlock& block);
  };

  class DenseDataBlock : public DataBlock {
  public:

    DenseDataBlock(unsigned width) :
      DataBlock(width),
      elements_(0),
      max_elements_(kStorageBlockSize/sizeof(double)) {}

    DenseDataBlock() : DenseDataBlock(1) { }

    void append(double *element) {
      DCHECK_GT(max_elements_, elements_);
      store_[elements_++] = *element;
    }

    std::uint32_t getSize() const override {
      return elements_;
    }

    void setSize(std::uint32_t new_size) {
      CHECK_GT(max_elements_, new_size);
      elements_ = new_size;
    }

    std::uint32_t getRemainingElements() const {
      return max_elements_ - elements_;
    }

    /**
     * Make a vertical slice of the data given the column indices to select.
     *
     * @param start_index inclusive column index.
     * @param end_index exclusive column index. Negative indices are treated as
     *              distances from the end index.
     * @return A new datablock containing the specified columns and all of the rows.
     */
    DenseDataBlock* slice(std::int32_t start_index, std::int32_t end_index) const;

    /**
     * Returns a new datablock containing only rows whose columns pass the filter function.
     * @param filter A function which takes a double attribute value and returns whether or not
     *              this attribute is selected.
     * @param col
     * @return
     */
    DenseDataBlock* filter(std::function<bool(double)> &filter, unsigned col) const;

    void matchRows(std::function<bool(double)> &filter, unsigned col, std::vector<unsigned> &matches) const;

    DenseDataBlock* sliceRows(const std::vector<unsigned>& rows) const;

    void setWidth(std::uint32_t width) {
      LOG_IF(WARNING, (0 != elements_ % width)) << "Incompatible width.";

      num_columns_ = width;
    }

    std::uint32_t getNumRows() const override {
      return elements_/num_columns_;
    }

    double *getRow(unsigned row) const override;


  private:
    std::uint32_t elements_;
    std::uint32_t max_elements_;

    friend std::ostream& operator<<(std::ostream& os, const DenseDataBlock& block);
    friend class Loader;
  };

  std::ostream& operator<<(std::ostream& os, const DenseDataBlock& block);

}  // namespace obamadb

#endif //OBAMADB_DATABLOCK_H_
