#include "UnorderedMatrix.h"

namespace obamadb {
#ifdef USE_SPOOKY_MATRIX
  std::ostream& operator<<(std::ostream& os, const UnorderedMatrix& matrix) {
    int size_mb = (matrix.maxSize_ * sizeof(MatrixEntry)) / 1e6;
    os << "(" << matrix.numRows() << ", " << matrix.numColumns() << ") "
       << matrix.size_ << " entries, approx " << size_mb << "mb";
    return os;
  }
#else

  std::ostream& operator<<(std::ostream& os, const UnorderedMatrix& matrix) {
    int size_mb = (matrix.entries_.size() * sizeof(int) * 3)/1e6;
    os << "(" << matrix.numRows() << ", " << matrix.numColumns() << ") "
       << matrix.entries_.size() << " entries, approx " << size_mb << "mb";
    return os;
  }
#endif

}