#include "UnorderedMatrix.h"

namespace obamadb {

  std::ostream& operator<<(std::ostream& os, const UnorderedMatrix& matrix) {
    int size_mb = 0;
    os << "(" << matrix.numRows() << ", " << matrix.numColumns() << ") "
       << 0 << " entries, approx " << size_mb << "mb";
    return os;
  }


}