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
     * Loads a CSV file into an array of datablocks.
     *
     * @param file_name
     * @return nullptr if datafile did not exist or was corrupt.
     */
    static std::vector<DataBlock*> load(const std::string& file_name);

  private:

    static void loadFileToDataBlocks(
      const std::string &file_name,
      std::vector<DataBlock*>& blocks);

    /**
     * Scans a single line of input and appends it to the given datablock.
     *
     * @param line
     * @param block
     * @return Number of elements which were scanned from the line.
     */
    static int scanLine(const std::string &line, DataBlock *block);
  };

}

#endif //  OBAMADB_STORAGE_LOADER_H_
