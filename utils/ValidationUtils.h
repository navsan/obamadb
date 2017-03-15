#ifndef OBAMADB_VALIDATIONUTILS_H_
#define OBAMADB_VALIDATIONUTILS_H_

#include <algorithm>
#include <string>
#include <unistd.h>
#include <vector>


bool ValidateThreads(const char* flagname, std::int64_t value) {
  int const max = 256;
  if (value > 0 && value <= max) {
    return true;
  }
  printf("The number of threads should be between 0 and %d\n", max);
  return false;
}

bool ValidateAlgorithm(const char* flagname, std::string const & value) {
  std::vector<std::string> valid_algorithms = {"svm", "mc"};
  if (std::find(valid_algorithms.begin(), valid_algorithms.end(), value) != valid_algorithms.end()) {
    return true;
  } else {
    printf("Invalid algorithm choice. Choices are:\n");
    for (std::string& alg : valid_algorithms) {
      printf("\t%s\n", alg.c_str());
    }
    return false;
  }
}

#endif // #ifndef OBAMADB_VALIDATIONUTILS_H_
