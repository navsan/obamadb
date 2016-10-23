#ifndef OBAMADB_UTILS_H
#define OBAMADB_UTILS_H

#include "glog/logging.h"

namespace  obamadb {

  /*
   * A vector which stores index->entry pairs.
   *
   * This is a very simple implementation and assumes that people inserted
   * it in lowest to highest value pairs. Running the verify() method will
   * check this.
   */
  template<class T>
  class svector {
  public:
    svector(int size) :
      num_elements_(0),
      alloc_size_(size),
      index_(new int[size]),
      values_(new T[size]) {	}

    void push_back(int idx, const T& value) {
      DCHECK_LT(num_elements_, alloc_size_);

      index_[num_elements_] = idx;
      values_[num_elements_] = value;
      num_elements_++;
    }

    /**
     * This includes the sparse entries which are implicitly null.
     * @return The total number of logical elements.
     */
    int size() const {
      if (0 == num_elements_)
        return 0;
      return index_[num_elements_-1];
    }

    int numElements() const {
      return num_elements_;
    }

    /**
     * Use binary search to find a specific entry.
     * @param idx Index of element.
     * @return nullptr if entry does not exist for that index.
     */
    T* get(int idx) const {
      if (num_elements_ == 0) {
        return nullptr;
      }

      if (num_elements_ == 1) {
        return index_[0] == idx ? values_ : nullptr;
      }

      int h = num_elements_;
      int l = 0;
      while(h - l > 1) {
        int c = (h-l)/2 + l;
        int const idxc = index_[c];
        if (idxc == idx) {
          return values_ + c;
        } else if (idxc > idx) {
          h = c;
        } else {
          l = c;
        }
      }

      return nullptr;
    }

    int num_elements_;
    int alloc_size_;
    int * index_;
    T * values_;
  };

}  // namespace obamadb

#endif //OBAMADB_UTILS_H
