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
      posix_fadvise(fd_, 0, 0, 1);  // FDADVICE_SEQUENTIAL
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

    /**
     * Scans a line of input. All sequences of non-decimal characters ([^0-9.-]) are treated
     * as delimiters between double values. Reads up through the newline.
     * @return A vector of doubles.
     */
    std::vector<double> scanLine() {
      std::vector<double> line;
      scanToDouble();
      while(!eof_) {
        line.push_back(nextDouble());
        scanToDouble();
        if (last_delimiter_ == '\n')
          break;
      }
      return line;
    }

  private:
    double readDouble() {
      scanToDouble();
      double value = 0;
      char c = *scan_ptr_;
      DCHECK(isDecimalChar(c)) << "no double available for scan.";
      value = readInt<false>();
      if (*scan_ptr_ == '.') {
        nextChar();
        value += (std::signbit(value) == 0 ? 1 : -1) * readInt<true>();
      }
      if (*scan_ptr_ == 'e' || *scan_ptr_ == 'E') {
        nextChar();
        double exp = readInt<false>();
        value = std::pow(10, exp) * value;
      }
      return value;
    }

    /**
     * Reads an integer. scan_ptr must begin on an integer character [0-9]+|-. Advances
     * the scan_ptr_ until a non-integer character is reached.
     * @tparam DECIMAL If the returned value should be as the decimal part of a fractional
     *                 number.
     */
    template<bool DECIMAL = false>
    double readInt() {
      double value = 0;
      int chars_read = 0;
      bool negative;
      if (!DECIMAL)
        negative = isNegative();
      while (isIntChar(*scan_ptr_)) {
        value *= 10;
        value += *scan_ptr_ - 48;
        chars_read++;
        nextChar();
      }
      if (DECIMAL) {
        return value / std::pow(10, chars_read);
      } else {
        value *= negative ? -1 : 1;
        return value;
      }
    }

    /**
     * Advances the scan_ptr if the current scan_ptr is -|+.
     * To be called when starting to scan a decimal.
     * @return True if the scan_ptr is '-', false otherwise
     */
    bool isNegative() {
      char c = *scan_ptr_;
      if (c == '-' || c == '+'){
        nextChar();
        return c == '-';
      }
      return false;
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

    void scanToDouble() {
      char c = *scan_ptr_;
      while (!isDecimalChar(c) && !eof_) {
        last_delimiter_ = c;
        c = nextChar();
      }
    }

    inline bool isIntChar(char c) {
      return (c >= 48 && c < 58);
    }

    inline bool isDecimalChar(char c) {
      return isIntChar(c) ||
             (c == '.') ||
             (c == '-') ||
             (c == '+') ||
             (c == 'e') ||
             (c == 'E');
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
