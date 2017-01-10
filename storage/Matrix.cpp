#include <iostream>
#include <cstdio>

#include "storage/Matrix.h"

namespace obamadb {

  std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
  {
    char buff[1000]; // TODO: buffer overflow possible.
    snprintf(buff, sizeof(buff),
             "Matrix: %lu training blocks for a total size of %ldmb with %d examples (%d nnz elements) with %f sparsity\n",
             matrix.blocks_.size(),
             (long) (matrix.sizeBytes() / 1e6),
             matrix.numRows_,
             matrix.getNNZ(),
             matrix.getSparsity());
    std::string buffAsStdStr = buff;
    os << buffAsStdStr;
    return os;
  }

} // namespace obamadb
