#ifndef OBAMADB_LOGUTILS_H_
#define OBAMADB_LOGUTILS_H_

#include "glog/logging.h"
#include "gflags/gflags.h"

#include <chrono>
#include <iostream>

DECLARE_bool(verbose);

#define VPRINT(str) { if(FLAGS_verbose) { printf(str); } }
#define VPRINTF(str, ...) { if(FLAGS_verbose) { printf(str, __VA_ARGS__); } }
#define VSTREAM(obj) {if(FLAGS_verbose){ std::cout << obj <<std::endl; }}

#define PRINT_TIMING(block)                                                \
  {                                                                        \
    if (FLAGS_verbose) {                                                   \
      auto time_start = std::chrono::steady_clock::now();                  \
      { block }                                                            \
      auto time_end = std::chrono::steady_clock::now();                    \
      std::chrono::duration<double, std::milli> time_ms =                  \
          time_end - time_start;                                           \
      printf("[TIMING][%s:%d] elapsed time %.2f ms\n", __FILE__, __LINE__, \
             time_ms.count());                                             \
    } else {                                                               \
      block                                                                \
    }                                                                      \
  }

#endif // #ifndef OBAMADB_LOGUTILS_H_
