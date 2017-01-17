#ifndef OBAMADB_STORAGECONSTANTS_H_
#define OBAMADB_STORAGECONSTANTS_H_

#include <cinttypes>

namespace obamadb {

  typedef std::int32_t int_t;

  const std::uint64_t kStorageBlockSize = 2e6;  // 2 megabytes.

  // Originally our data was in float TF/IDF values [0,1]. When we convert these
  // values to ints, we multiply them by this constant.
  // Scaling by 1e4 seems to be the highest we can go (while staying a power of 10)
  // without running into overflow issues.
  const int_t kScaleFloats = 1e4;

}  // namespace obamadb

#endif //OBAMADB_STORAGECONSTANTS_H_
