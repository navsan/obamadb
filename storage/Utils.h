#ifndef OBAMADB_UTILS_H
#define OBAMADB_UTILS_H

#include <chrono>

#include "glog/logging.h"

#define PRINT_TIMING(block) { auto time_start = std::chrono::steady_clock::now();\
                              block\
                              auto time_end = std::chrono::steady_clock::now();\
                              std::chrono::duration<double, std::milli> time_ms = time_end - time_start;\
                              printf("[%s:%d] %s elapsed time %f\n",__FILE__, __LINE__, __FUNCTION__, time_ms.count()); \
                            }

#define DISABLE_COPY_AND_ASSIGN(CLASS) \
            CLASS & operator=(const CLASS&) = delete;\
            CLASS(const CLASS&) = delete

namespace  obamadb {
  /**
   * Virtual base class for a training example vector.
   * Each vector is a map of attribute index->attribute value and a slot for classification.
   */
  template<class T>
  class e_vector {
  public:
    virtual void clear() = 0;

    virtual void push_back(const T& value) = 0;

    virtual T* get(int index) const = 0;

    virtual T* getLast() const = 0;

    virtual void setClassification(T* classification) = 0;

    virtual T* getClassification() = 0;

    virtual int size() const = 0;

    virtual int sizeBytes() const = 0;

    virtual void setMemory(int size, void * src) = 0;

  };

  template<class T>
  class de_vector : public e_vector<T> {
  public:
    /**
     * Create a dvector which owns its own memory.
     * @param size Maximum number of non-null elements this can contain.
     */
    de_vector(int size) :
      values_(new T[size]),
      num_elements_(0),
      alloc_size_(size),
      owns_memory_(true) {	}

    /**
     * Copy constructor.
     */
    de_vector(const de_vector<T>& other)
      : num_elements_(other.num_elements_),
        alloc_size_(other.alloc_size_),
        values_(other.values_),
        owns_memory_(false) { }

    de_vector(int size, void * vloc) :
      num_elements_(size),
      alloc_size_(size),
      values_(reinterpret_cast<T*>(vloc)),
      owns_memory_(false) {}

    ~de_vector() {
      if (owns_memory_)
        delete[] values_;
    }

    /**
     * Creates an empty svector with a default amount of space.
     */
    de_vector() : de_vector(8) {}

    void setMemory(int size, void * src) {
      if(owns_memory_) {
        delete[] values_;
        owns_memory_ = false;
      }
      num_elements_ = size;
      values_ = reinterpret_cast<T*>(src);
    }

    void clear() {
      num_elements_ = 0;
    }

    void setClassification(T* classification) {
      CHECK(false) << "TODO";
    }

    T* getClassification() {
      CHECK(false) << "TODO";
      return nullptr;
    }

    void push_back(const T& value) {
      if (num_elements_ == alloc_size_) {
        T * tvalues = new T[alloc_size_ * 2];
        memcpy(tvalues, values_, sizeof(T) * alloc_size_);
        delete[] values_;
        values_ = tvalues;
        alloc_size_ *= 2;
      }

      values_[num_elements_] = value;
      num_elements_++;
    }

    T* get(int index) const  {
      DCHECK_LT(index, num_elements_);
      return values_ + index;
    }

    T* getLast() const {
      return values_ + num_elements_ - 1;
    }

    int size() const  {
      return num_elements_;
    }

    /**
    * @return Minimum number of bytes needed to store this object.
    */
    int sizeBytes() const {
      return sizeof(T) * (num_elements_ + 1);
    }

    T* values_;

  private:
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
  class se_vector : public e_vector<T> {
  public:

    /**
     * Create an svector which owns its own memory.
     * @param size Maximum number of non-null elements this can contain.
     */
    se_vector(int size) :
      index_(new int[size]),
      values_(new T[size]),
      class_(new T),
      num_elements_(0),
      alloc_size_(size),
      owns_memory_(true) {	}

    /**
     * Creates an empty svector with a default amount of space.
     */
    se_vector() : se_vector(8) {}

    /**
     * Creates an svector using existing memory. Does not take ownership.
     * @param size The number of non-null elements in the region of memory.
     * @param vloc region where the index_ and T values are stored.
     */
    se_vector(int size, void * vloc)
      : index_(nullptr),
        values_(nullptr),
        class_(nullptr),
        num_elements_(size),
        alloc_size_(size),
        owns_memory_(false) {
      setMemory(size, vloc);
    }

    /**
     * Copy constructor.
     */
    se_vector(const se_vector<T>& other)
      : num_elements_(other.num_elements_),
        alloc_size_(other.alloc_size_),
        index_(other.index_),
        values_(other.values_),
        class_(other.class_),
        owns_memory_(false) { }

    ~se_vector() {
      if (owns_memory_) {
        delete[] index_;
        delete[] values_;
        delete class_;
      }
    }

    void setMemory(int size, void * src) {
      if (owns_memory_){
        delete[] index_;
        delete[] values_;
        delete class_;
        owns_memory_ = false;
      }

      num_elements_ = size;
      index_ = reinterpret_cast<int*>(src);
      values_ = reinterpret_cast<T*>(index_ + size);
      class_ = reinterpret_cast<T*>(values_ + size);
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
    void copyTo(void * dst) const {
      memcpy(dst, index_, sizeof(int) * num_elements_);

      T* vptr = reinterpret_cast<T*>(reinterpret_cast<int*>(dst) + num_elements_);
      memcpy(vptr, values_, sizeof(T) * num_elements_);

      T* cptr = vptr + num_elements_;
      memcpy(cptr, class_, sizeof(T));
    }

    void setClassification(T* classification) {
      if (owns_memory_) {
       memcpy(class_, classification, sizeof(T));
      } else {
       class_ = classification;
      }
    }

    T* getClassification() {
      return class_;
    }

    /**
     * @return Minimum number of bytes needed to store this object.
     */
    int sizeBytes() const {
      return (num_elements_ * sizeof(int)) + (sizeof(T) * (num_elements_ + 1));
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

    void push_back(const T& value) {
      if (alloc_size_ == num_elements_) {
        doubleAllocation();
      }

      int index = 0;
      if (num_elements_ > 0 ) {
        index = index_[num_elements_ - 1] + 1;
      }

      index_[num_elements_] = index;
      values_[num_elements_] = value;
      num_elements_++;
    }

    void push_back(int idx, const T& value) {

      if (alloc_size_ == num_elements_) {
        doubleAllocation();
      }

      index_[num_elements_] = idx;
      values_[num_elements_] = value;
      num_elements_++;
    }

    /**
     * This includes the sparse entries which are implicitly null.
     * @return The total number of logical elements. (The last logical index + 1)
     */
    int size() const {
      if (0 == num_elements_)
        return 0;
      return index_[num_elements_-1] + 1;
    }

    int numElements() const {
      return num_elements_;
    }

    /**
     * Within the sparse structure, get the element index which is referred to.
     * Useful for doing a sparse iteration iteration.
     * @param index Physical index within the data structure.
     * @return Element index.
     */
    int elementAtIndex(int index) {
      DCHECK_LT(index, num_elements_);
      return index_[index];
    }

    /**
     * Use binary search to find a specific entry.
     * @param idx Index of element.
     * @return nullptr if entry does not exist for that index.
     */
    T* get(int idx) const {
      for (int i = 0; i < num_elements_; i++) {
        if (index_[i] == idx) {
          return &values_[i];
        }
      }
      return nullptr;
    }

    T* getLast() const {
      return values_ + num_elements_ - 1;
    }

    int * index_;
    T * values_;
    T * class_;

    int num_elements_;

  private:

    void doubleAllocation() {
      DCHECK(owns_memory_);
      // double the size.
      int * tidx = new int[alloc_size_ * 2];
      T * tvalues = new T[alloc_size_ * 2];
      memcpy(tidx, index_, sizeof(int) * alloc_size_);
      memcpy(tvalues, values_, sizeof(T) * alloc_size_);
      delete[] index_;
      delete[] values_;
      index_ = tidx;
      values_ = tvalues;
      alloc_size_ *= 2;
    }

    int alloc_size_;
    bool owns_memory_;
  };

  /**
   * Simple vector for floating type numbers.
   */
  struct f_vector {
    f_vector(unsigned dimension)
      : dimension_(dimension) {
      values_ = new float_t[dimension_];
    }

    f_vector(const f_vector& other) {
      dimension_ = other.dimension_;
      values_ = new float_t[dimension_];
      memcpy(values_, other.values_, sizeof(float_t) * dimension_);
    }

    ~f_vector() {
      delete[] values_;
    }

    float_t& operator[](int idx) const {
      DCHECK_GT(dimension_, idx);

      return values_[idx];
    }

    void clear() {
      memset(values_,0, sizeof(float_t) * dimension_);
    }

    unsigned dimension_;
    float_t *values_;

  };

  class QuickRandom {
  public:
    QuickRandom() : x(15486719), y(19654991), z(16313527), char_index(0) {
      LOG_IF(FATAL, sizeof(std::uint64_t) != 8) << "Expect 64_t to be 8 bytes long.";
      // warm up 5 cycles.
      for (int i = 0; i < 5; i++) {
        nextInt64();
      }
    }

    inline std::uint64_t nextInt64() {
      // TODO: there are many other ways to generate random numbers,
      // http://stackoverflow.com/questions/1640258/need-a-fast-random-generator-for-c
      // has several more techniques to choose from and includes this one.
      std::uint64_t t;
      x ^= x << 16;
      x ^= x >> 5;
      x ^= x << 1;

      t = x;
      x = y;
      y = z;
      z = t ^ x ^ y;

      return z;
    }

    inline std::uint32_t nextInt32() {
      if (char_index == 0) {
        char_index = 4;
        return *reinterpret_cast<uint32_t *>(&z);
      } else if (char_index == 4) {
        char_index = 8;
        return reinterpret_cast<uint32_t *>(&z)[1];
      } else {
        nextInt64();
        char_index = 4;
        return *reinterpret_cast<uint32_t *>(&z);
      }
    }

    inline unsigned char nextChar() {
      if (char_index >= 8) {
        nextInt64();
        char_index = 0;
      }
      return reinterpret_cast<unsigned char*>(&z)[char_index++];
    }

    std::uint64_t x, y, z;
    int char_index;
  };

}  // namespace obamadb

#endif //OBAMADB_UTILS_H
