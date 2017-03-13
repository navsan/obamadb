#ifndef OBAMADB_UTILS_H
#define OBAMADB_UTILS_H

#include "storage/StorageConstants.h"

#include <chrono>
#include <cmath>

#include "glog/logging.h"
#include "gflags/gflags.h"

DECLARE_bool(verbose);

#define PRINT_TIMING(block) { \
                              if (FLAGS_verbose) { \
                                auto time_start = std::chrono::steady_clock::now();\
                                {block}\
                                auto time_end = std::chrono::steady_clock::now();\
                                std::chrono::duration<double, std::milli> time_ms = time_end - time_start;\
                                printf("[TIMING][%s:%d] elapsed time %.2f ms\n",__FILE__, __LINE__, time_ms.count()); \
                              } else {\
                                block \
                              } \
                            }

#define DISABLE_COPY_AND_ASSIGN(CLASS) \
            CLASS & operator=(const CLASS&) = delete;\
            CLASS(const CLASS&) = delete

namespace obamadb {


  /**
   * Simple vector for floating type numbers.
   */
  struct fvector {
    fvector(unsigned dimension)
      : dimension_(dimension) {
      values_ = new num_t[dimension_];
    }

    fvector(const fvector &other) {
      dimension_ = other.dimension_;
      values_ = new num_t[dimension_];
      memcpy(values_, other.values_, sizeof(num_t) * dimension_);
    }

    /**
     * An fvector filled with values [-1,1]
     * @param dim Number of elements in the new fvector.
     * @return An fvector filled with random floats.
     */
    static fvector GetRandomFVector(int const dim);

    ~fvector() {
      delete[] values_;
    }

    num_t &operator[](int idx) const {
      DCHECK_GT(dimension_, idx);

      return values_[idx];
    }

    void clear() {
      memset(values_, 0, sizeof(num_t) * dimension_);
    }

    unsigned dimension_;
    num_t *values_;

  };

  inline int randomInt() {
    return rand();
  }

  inline float randomFloat() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  }

  namespace stats {
    template<class T>
    double mean(std::vector<T> values) {
      double sum = 0;
      for (T& t : values) {
        sum += t;
      }
      return sum / values.size();
    }

    template<class T>
    double variance(std::vector<T> values) {
      double m = mean<T>(values);
      double sq_res = 0;
      for (T& t : values) {
        double diff = m - t;
        sq_res += diff * diff;
      }
      return sq_res/values.size();
    }

    template<class T>
    double stddev(std::vector<T> values) {
      double var = variance<T>(values);
      return sqrt(var);
    }

    template<class T>
    double stderr(std::vector<T> values) {
      double stdv = stddev<T>(values);
      return stdv/sqrt(values.size()); // stderr grows smaller as the number of samples increases
    }
  }

  /**
   * Takes a list like "1,2,3,42" and converts it into a vector of integers.
   * @param list - Integers to convert.
   * @return A vector of the represented ints.
   */
  std::vector<int> GetIntList(std::string const & list);

}  // namespace obamadb

#endif //OBAMADB_UTILS_H
