#include "storage/SparseDataBlock.h"

#include "storage/exvector.h"

namespace obamadb {

  std::ostream &operator<<(std::ostream &os, const SparseDataBlock<num_t> &block) {
    svector<num_t> row;
    int cols_max = block.getNumColumns();
    for (int i = 0; i < block.getNumRows(); i++) {
      block.getRowVector(i, &row);
      for (int i = 0; i < cols_max; i++) {
        num_t * dptr = row.get(i);
        if (dptr == nullptr) {
          os << 0 << ",";
        } else {
          os << *dptr << ",";
        }
      }
      os << *row.getClassification() << "\n";
    }
    return os;
  }

}
