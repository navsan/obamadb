#ifndef OBAMADB_STORAGECONSTANTS_H_
#define OBAMADB_STORAGECONSTANTS_H_

#include <cinttypes>

namespace obamadb {

  typedef float float_t;
  typedef float alignas(64) aligned_float_t;

  const std::uint64_t kStorageBlockSize = 2e6;  // 2 megabytes.

}  // namespace obamadb

#endif //OBAMADB_STORAGECONSTANTS_H_
