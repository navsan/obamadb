#ifndef OBAMADB_UTILS_H
#define OBAMADB_UTILS_H

#include "storage/StorageConstants.h"

#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>

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

  class Scanner {
  public:
    Scanner(std::string const & file_name) :
      fd_(-1),
      buff_(new char[BUFFER_SIZE + 1]),
      scan_ptr_(nullptr),
      last_delimiter_('\0') {
      fd_ = open(file_name.c_str(), O_RDONLY);
      CHECK_NE(fd_, -1) << "Error opening file: " << file_name;
#ifndef __APPLE__
      // Advise the kernel of our access pattern.
      posix_fadvise(fd, 0, 0, 1);  // FDADVICE_SEQUENTIAL
#endif
      readChunk();
    }

    ~Scanner() {
      delete[] buff_;
      close(fd_);
    }

    double nextDouble() {
      if (eof_) {
        return 0;
      }
      return readDouble();
    }

    bool eof() const {
      return eof_;
    }

    char lastDelimiter() const {
      return last_delimiter_;
    }

    std::vector<double> scanLine() {
      std::vector<double> line;
      scanToFloat();
      while(*scan_ptr_ != '\n' && !eof_) {
        line.push_back(nextDouble());
      }
      return line;
    }

  private:
    double readDouble() {
      scanToFloat();
      double value = 0;
      bool decimal = false;
      int decimal_places = 0;
      bool negative = false;
      char c = *scan_ptr_;
      DCHECK(isDecimalChar(c));
      while (isDecimalChar(c)) {
        if (c == '-') {
          DCHECK_EQ(negative, false);
          negative = true;
        } else if (c == '+') {
          DCHECK_EQ(negative, false);
        } else if (c == '.') {
          DCHECK_EQ(decimal, false);
          decimal = true;
        } else {
          char val = c - 48;
          DCHECK_LT(val, 10);
          value *= 10;
          value += val;
          decimal_places += decimal;
        }
        c = nextChar();
      }
      value = value / pow(10,decimal_places);
      if (negative) {
        value *= -1;
      }
      return value;
    }

    char nextChar() {
      scan_ptr_++;
      if (*scan_ptr_ == '\0') {
        readChunk();
        scan_ptr_ = buff_;
      }
      DCHECK_NE(buff_ + BUFFER_SIZE, scan_ptr_) << "Scanner over the end of the buffer";
      return *scan_ptr_;
    }

    void scanToFloat() {
      char c = *scan_ptr_;
      while (!isDecimalChar(c) && !eof_) {
        last_delimiter_ = c;
        c = nextChar();
      }
    }

    inline bool isDecimalChar(char c) {
      return (c >= 48 && c < 58) ||
             (c == '.') ||
             (c == '-') ||
             (c == '+');
    }

    /**
     *
     * @param offset Offset into the buffer to read.
     * @return True if we read in more data. False if EOF.
     */
    void readChunk() {
      int bytes_read = read(fd_, buff_, BUFFER_SIZE);
      CHECK_NE(bytes_read, -1) << "Error reading file";
      buff_[bytes_read] = '\0';
      scan_ptr_ = buff_;
      eof_ = bytes_read == 0;
    }

    int fd_;
    char *buff_;
    char *scan_ptr_;
    char last_delimiter_;

    bool eof_;

    static std::size_t const BUFFER_SIZE;
  };

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
