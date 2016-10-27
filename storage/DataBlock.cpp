#include "storage/DataBlock.h"

#include <cstring>
#include <iostream>
#include <memory>

namespace obamadb {

  template<class T>
  std::ostream& operator<<(std::ostream &os, const DataBlock<T> &block) {
    os << "DataBlock[" << std::to_string(block.num_rows_) << " " << std::to_string(block.num_columns_) << "]";
    return os;
  }

}
