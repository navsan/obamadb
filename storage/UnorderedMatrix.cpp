#include "UnorderedMatrix.h"

namespace obamadb {

  std::ostream& operator<<(std::ostream& os, const UnorderedMatrix& matrix) {
    int size_mb = (matrix.entries_.size() * sizeof(int) * 3)/1e6;
    os << "(" << matrix.numRows() << ", " << matrix.numColumns() << ") "
       << matrix.entries_.size() << " entries, approx " << size_mb << "mb";
    return os;
  }

}