#include <iostream>

#include "storage/Matrix.h"

namespace obamadb {

  std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
  {
    os << "Matrix: (" << matrix.numRows_ << ", " << matrix.numColumns_ << ") "
       << matrix.blocks_.size() << " training blocks, "
       << (long) (matrix.sizeBytes() / 1e6) << "mb, "
       << matrix.numRows_  << " rows, " << matrix.getNNZ() << " elements, "
       << matrix.getSparsity() << " sparsity\n";
    return os;
  }

} // namespace obamadb
