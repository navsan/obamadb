#ifndef OBAMADB_STORAGECONSTANTS_H_
#define OBAMADB_STORAGECONSTANTS_H_

#include <cinttypes>

namespace obamadb {

  typedef float float_t;

  const std::uint64_t kStorageBlockSize = 16e6;  // 4 megabytes.

}  // namespace obamadb

#endif //OBAMADB_STORAGECONSTANTS_H_
