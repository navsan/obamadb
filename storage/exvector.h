#ifndef OBAMADB_EXVECTOR_H
#define OBAMADB_EXVECTOR_H

#include <cstring>

#include "glog/logging.h"

namespace obamadb {
  /**
   * Virtual base class for a training example (ex)vector.
   * Each vector is a map of attribute index->attribute value and a slot for classification.
   */
  template<class T>
  class exvector {
  public:
    virtual void clear() = 0;

    virtual void push_back(const T &value) = 0;

    virtual T *get(int index) const = 0;

    virtual int size() const = 0;

    virtual int sizeBytes() const = 0;

    virtual void setMemory(int size, void *src) = 0;
  };

  /**
   * Dense example vector. A physical element i is represented at index i.
   */
  template<class T>
  class dvector : public exvector<T> {
  public:
    /**
     * Create a dvector which owns its own memory.
     * @param size Maximum number of non-null elements this can contain.
     */
    dvector(int size) :
      values_(new T[size]),
      class_(new T),
      num_elements_(0),
      alloc_size_(size),
      owns_memory_(true) {}

    /**
     * Copy constructor.
     */
    dvector(const dvector<T> &other)
      : num_elements_(other.num_elements_),
        alloc_size_(other.alloc_size_),
        values_(other.values_),
        class_(other.class_),
        owns_memory_(false) {}

    /**
     * Creates a de_vector which does not own the memory under it. It expects th
     * memory to be laid out like [values (# size)][classification]
     * @param size Size of the data set (number of elements excluding classification)
     * @param valuesPtr Pointer to start of the values array.
     */
    dvector(int size, void *valuesPtr) :
      num_elements_(size),
      alloc_size_(size),
      values_(reinterpret_cast<T *>(valuesPtr)),
      class_(values_ + num_elements_),
      owns_memory_(false) {}

    /**
     * Creates an empty svector with a default amount of space.
     */
    dvector() :
      dvector(8) {}

    ~dvector() {
      release();
    }

    /**
     * Expects the memory to be laid out like [values (# size)][classification]
     * @param size Size of the data set (number of elements excluding classification)
     * @param valuesPtr Pointer to start of the values array.
     */
    void setMemory(int size, void *valuesPtr) {
      release();
      num_elements_ = size;
      values_ = reinterpret_cast<T *>(valuesPtr);
      class_ = values_ + num_elements_;
    }

    void clear() {
      CHECK(owns_memory_) << "Attempted to clear non-owned memory.";
      num_elements_ = 0;
    }

    void push_back(const T &value) {
      if (num_elements_ == alloc_size_) {
        doubleAllocation();
      }
      values_[num_elements_++] = value;
    }

    T *get(int index) const {
      DCHECK_LT(index, num_elements_);
      return values_ + index;
    }

    /**
     * Number of elements. May have allocated more memory.
     * @return Num elements.
     */
    int size() const {
      return num_elements_;
    }

    /**
     * @return Minimum number of bytes needed to store this object.
     */
    int sizeBytes() const {
      return sizeof(T) * (num_elements_ + 1);
    }

    T *values_;
    T *class_;

  private:
    void doubleAllocation() {
      T *tempValues = new T[alloc_size_ * 2];
      memcpy(tempValues, values_, sizeof(T) * alloc_size_);
      delete[] values_;
      values_ = tempValues;
      alloc_size_ *= 2;
    }

    void release() {
      if (owns_memory_) {
        delete[] values_;
        delete class_;
        owns_memory_ = false;
      }
    }

    int num_elements_;
    int alloc_size_;
    bool owns_memory_;
  };

  /*
   * A vector which stores index->entry pairs.
   *
   * This is a very simple implementation and assumes that people inserted
   * it in lowest to highest value pairs. Running the verify() method will
   * check this.
   */
  template<class T>
  class svector : public exvector<T> {
  public:

    /**
     * Create an svector which owns its own memory.
     * @param size Maximum number of non-null elements this can contain.
     */
    svector(int size) :
      index_(new int[size]),
      values_(new T[size]),
      class_(new T),
      num_elements_(0),
      alloc_size_(size),
      owns_memory_(true) {}

    /**
     * Creates an empty svector with a default amount of space.
     */
    svector() :
      svector(8) {}

    /**
     * Creates an svector using existing memory. Does not take ownership.
     * Expects the memory to be laid out like [index][values][class] where
     * there are no gaps (empty space) in allocation.
     *
     * @param size The number of non-null elements in the region of memory.
     * @param valuePtr region where the index_ and T values are stored.
     */
    svector(int size, void *valuePtr)
      : index_(nullptr),
        values_(nullptr),
        class_(nullptr),
        num_elements_(size),
        alloc_size_(size),
        owns_memory_(false) {
      setMemory(size, valuePtr);
    }

    /**
     * Copy constructor.
     */
    svector(const svector<T> &other)
      : num_elements_(other.num_elements_),
        alloc_size_(other.alloc_size_),
        index_(other.index_),
        values_(other.values_),
        class_(other.class_),
        owns_memory_(false) {}

    ~svector() {
      release();
    }

    void setMemory(int size, void *src) {
      if (owns_memory_) {
        delete[] index_;
        delete[] values_;
        delete class_;
        owns_memory_ = false;
      }

      num_elements_ = size;
      index_ = reinterpret_cast<int *>(src);
      values_ = reinterpret_cast<T *>(index_ + size);
      class_ = reinterpret_cast<T *>(values_ + size);
    }

    bool owns_memory() const {
      return owns_memory_;
    }

    /**
     * Copies the core data structure. Does not copy the numElements. Copies exactly as many elements as there are
     * stored. This means if the svector was overallocated, it does not include the empty space.
     *
     * @param dst Destination memory.
     */
    void copyTo(void *dst) const {
      memcpy(dst, index_, sizeof(int) * num_elements_);

      T *valuesPtr = reinterpret_cast<T *>(reinterpret_cast<int *>(dst) + num_elements_);
      memcpy(valuesPtr, values_, sizeof(T) * num_elements_);

      T *classPtr = valuesPtr + num_elements_;
      memcpy(classPtr, class_, sizeof(T));
    }

    void setClassification(T *classification) {
      if (owns_memory_) {
        memcpy(class_, classification, sizeof(T));
      } else {
        class_ = classification;
      }
    }

    T *getClassification() {
      return class_;
    }

    /**
     * @return Minimum number of bytes needed to store this object. Makes no difference whether it owns
     * the memory or not.
     */
    int sizeBytes() const {
      return (num_elements_ * sizeof(int)) + (sizeof(T) * (num_elements_ + 1));
    }

    /*
     * Deletes all the elements if it owns the data. Does not relinquish memory.
     */
    void clear() {
      CHECK(owns_memory_) << "Attempted to clear somebody else's memory.";
      num_elements_ = 0;
    }

    /**
     * Appends the value to the array in the next logical index.
     * @param value To append.
     */
    void push_back(const T &value) {
      int index = 0;
      if (num_elements_ > 0) {
        index = index_[num_elements_ - 1] + 1;
      }
      push_back(index, value);
    }

    /**
     * Appends the value as if it were in the specified index.
     * @param idx Location of the value.
     * @param value To append.
     */
    void push_back(int idx, const T &value) {
      if (alloc_size_ == num_elements_) {
        doubleAllocation();
      }
      index_[num_elements_] = idx;
      values_[num_elements_] = value;
      num_elements_++;
    }

    /**
     * This includes the sparse entries which are implicitly null.
     *
     * @return The total number of logical elements. (The last logical index + 1)
     */
    int size() const {
      if (0 == num_elements_)
        return 0;
      return index_[num_elements_ - 1] + 1;
    }

    int numElements() const {
      return num_elements_;
    }

    /**
     * Use binary search to find a specific entry.
     * @param idx Index of element.
     * @return nullptr if entry does not exist for that index.
     */
    T *get(int idx) const {
      for (int i = 0; i < num_elements_; i++) {
        if (index_[i] == idx) {
          return &values_[i];
        }
      }
      return nullptr;
    }

    int *index_;
    T *values_;
    T *class_;

    int num_elements_;

  private:

    void release() {
      if (owns_memory_) {
        delete[] index_;
        delete[] values_;
        delete class_;
      }
    }

    void doubleAllocation() {
      DCHECK(owns_memory_);

      int *tempIdx = new int[alloc_size_ * 2];
      T *tempValues = new T[alloc_size_ * 2];
      memcpy(tempIdx, index_, sizeof(int) * alloc_size_);
      memcpy(tempValues, values_, sizeof(T) * alloc_size_);
      delete[] index_;
      delete[] values_;
      index_ = tempIdx;
      values_ = tempValues;
      alloc_size_ *= 2;
    }

    int alloc_size_;
    bool owns_memory_;
  };
}

#endif //OBAMADB_EXVECTOR_H
