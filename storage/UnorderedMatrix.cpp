#include "UnorderedMatrix.h"

namespace obamadb {
  std::ostream& operator<<(std::ostream& os, const UnorderedMatrix& matrix) {
    int size_mb = (matrix.maxSize_ * sizeof(MatrixEntry)) / 1e6;
    os << "(" << matrix.numRows() << ", " << matrix.numColumns() << ") "
       << matrix.size_ << " entries, approx " << size_mb << "mb";
    return os;
  }

}