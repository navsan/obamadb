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

    /**
     * Create an svector which owns its own memory.
     * @param size Maximum number of non-null elements this can contain.
     */
    svector(int size) :
      num_elements_(0),
      alloc_size_(size),
      index_(new int[size]),
      values_(new T[size]),
      owns_memory_(true) {	}

    /**
     * Creates an empty svector with a default amount of space.
     */
    svector() : svector(8) {}

    /**
     * Creates an svector using existing memory. Does not take ownership.
     * @param size The number of non-null elements in the region of memory.
     * @param vloc region where the index_ and T values are stored.
     */
    svector(int size, void * vloc)
      : num_elements_(size),
        alloc_size_(size),
        index_(reinterpret_cast<int*>(vloc)),
        values_(reinterpret_cast<T*>(index_ + num_elements_)),
        owns_memory_(false) { }

    /**
     * Copy constructor.
     */
    svector(const svector& other)
      : num_elements_(other.num_elements_),
        alloc_size_(other.alloc_size_),
        index_(other.index_),
        values_(other.values_),
        owns_memory_(false) { }

    ~svector() {
      if (owns_memory_) {
        delete index_;
        delete values_;
      }
    }

    /**
     * Copies the core data structure. Does not copy the numElements. Copies exactly as many elements as there are
     * stored. This means if the svector was overallocated, it does not include the empty space.
     *
     * @param dst Destination memory.
     */
    void copyTo(void * dst) const {
      memcpy(dst, index_, sizeof(int) * num_elements_);
      memcpy(static_cast<int*>(dst) + num_elements_, values_, sizeof(T) * num_elements_);
    }

    /*
     * Deletes all the elements if it owns the data. Does not reliquish memory.
     * Does nothing if it does not own memory.
     */
    void clear() {
      if(owns_memory_) {
        num_elements_ = 0;
        // TODO: should this call the destructor of the elements which it owns?
        // Certainly not in the case of doubles but for other types, one could imagine leaks..
      }
    }

    void push_back(int idx, const T& value) {

      if (alloc_size_ == num_elements_) {
        // double the size.
        int * tidx = new int[alloc_size_ * 2];
        T * tvalues = new T[alloc_size_ * 2];
        memcpy(tidx, index_, sizeof(int) * alloc_size_);
        memcpy(tvalues, values_, sizeof(T) * alloc_size_);
        index_ = tidx;
        values_ = tvalues;
        alloc_size_ *= 2;
      }

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

    bool owns_memory_;
  };

}  // namespace obamadb

#endif //OBAMADB_UTILS_H
