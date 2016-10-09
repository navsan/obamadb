#ifndef OBAMADB_STORAGE_LOADER_H_
#define OBAMADB_STORAGE_LOADER_H_

#include "storage/DataBlock.h"
#include <cctype>
#include <fstream>
#include <istream>
#include <iterator>
#include <string>

namespace obamadb {

  class Loader {
  public:
    /**
     * Loads a CSV file into a DataSet.
     * @param file_name
     * @return nullptr if datafile did not exist or was corrupt.
     */
    static DataBlock* load(const std::string& file_name);

  private:

    static DataBlock* loadFileToDataSet(const std::string& file_name);

  };

}

#endif //  OBAMADB_STORAGE_LOADER_H_
