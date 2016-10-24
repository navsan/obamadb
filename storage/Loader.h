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
     * @param sparse If this is true, we assume the dataset is in the same format as HogWild! authors
     *  give their RCV1 data.
     * @return nullptr if datafile did not exist or was corrupt.
     */
    static std::vector<DenseDataBlock*> load(const std::string& file_name, bool sparse);

    /**
     * Saves a datablock to the specified filename.
     *
     * @param file_name
     * @param datablock
     */
    static void save(const std::string& file_name, const DenseDataBlock& datablock);

   /**
    * Expects the data to be in the format:
    * [ID][attribute index][value]
    * where IDs are in increasing order and where on a new ID row, it will contain the class (-1,1)
    * of the training example in the [value] position.
    * @param file_name The file.
    * @param blocks A vector which the method will populate.
    */
    static void loadFileToSparseDataBlocks(
      const std::string &file_name,
      std::vector<SparseDataBlock*>& blocks);

  private:

    static void loadFileToDataBlocks(
      const std::string &file_name,
      std::vector<DenseDataBlock*>& blocks);



    /**
     * Scans a single line of input and appends it to the given datablock.
     *
     * @param line
     * @param block
     * @return Number of elements which were scanned from the line.
     */
    static int scanLine(const std::string &line, DenseDataBlock *block);
  };

}

#endif //  OBAMADB_STORAGE_LOADER_H_
