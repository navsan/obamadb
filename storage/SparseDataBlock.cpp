#include "storage/SparseDataBlock.h"
#include "storage/DataBlock.h"

namespace obamadb {

  std::ostream &operator<<(std::ostream &os, const SparseDataBlock<double> &block) {
    se_vector<double> row;
    int cols_max = block.getNumColumns();
    for (int i = 0; i < block.getNumRows(); i++) {
      block.getRowVector(i, &row);
      for (int i = 0; i < cols_max; i++) {
        double * dptr = row.get(i);
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
