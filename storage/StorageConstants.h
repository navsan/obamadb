#ifndef OBAMADB_STORAGECONSTANTS_H_
#define OBAMADB_STORAGECONSTANTS_H_

#include <cinttypes>

namespace obamadb {

  // The basic machine learning number type. Since it can be either floating point or
  // integer type, we call it num(ber) type.
  typedef float num_t;

  const std::uint64_t kStorageBlockSize = 2e6;  // 2 megabytes.

}  // namespace obamadb

#endif //OBAMADB_STORAGECONSTANTS_H_
