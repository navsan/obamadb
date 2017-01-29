#ifndef OBAMADB_STORAGE_IO_H_
#define OBAMADB_STORAGE_IO_H_

#include "storage/exvector.h"
#include "storage/Matrix.h"
#include "storage/SparseDataBlock.h"
#include "storage/StorageConstants.h"

#include <cctype>
#include <fstream>
#include <istream>
#include <iterator>
#include <string>

namespace obamadb {

  class Matrix;

  namespace IO {

   /**
    * Expects the data to be in the format:
    * [ID][attribute index][value]
    * where IDs are in increasing order and where on a new ID row, it will contain the class (-1,1)
    * of the training example in the [value] position.
    *
    * @param file_name
    * @return nullptr if datafile did not exist or was corrupt.
    */
    template<class T>
    std::vector<SparseDataBlock<T>*> loadBlocks(const std::string &file_name);

    /**
     * Load a sparse file representation of a dataset into a matrix.
     * @param filename The sparse datafile.
     * @return Caller-owned matrix.
     */
    Matrix* load(const std::string &filename);

    void save(const std::string& file_name, const Matrix& mat);

    /**
     * Saves a datablock to the specified filename.
     *
     * @param file_name
     * @param datablock
     */
    template<class T>
    void save(const std::string& file_name, const DataBlock<T>* datablock);

    template<class T>
    void save(const std::string &file_name, const obamadb::DataBlock<T>* datablock) {
      std::ofstream file;
      file.open(file_name, std::ios::out | std::ios::binary);

      CHECK(file.is_open()) << "Unable to open " << file_name << " for output.";

      if (datablock->getDataBlockType == obamadb::DataBlockType::kSparse) {
        CHECK(false) << "Not implemented";
      } else {
        CHECK(false) << "Unknown block type";
      }

      file.close();
    }

    /**
     * Save some number of blocks from a set of SparseBlocks.
     * @param file_name The file to save to.
     * @param blocks List of SparseBlocks
     * @param nblocks The number of blocks which you would like to save.
     */
    template<class T>
    void save(const std::string& file_name, std::vector<SparseDataBlock<T>*> blocks, int const numBlocks) {
      std::ofstream file;
      file.open(file_name, std::ios::out | std::ios::binary);
      CHECK(file.is_open()) << "Unable to open " << file_name << " for output.";

      int max_columns = maxColumns<T>(blocks);

      DCHECK_LE(numBlocks, blocks.size());

      for (int i = 0; i < numBlocks; ++i) {
        const SparseDataBlock<T> &block = *blocks[i];
        svector<T> row;
        for (int j = 0; j < block.getNumRows(); j++) {
          block.getRowVector(j, &row);
          for (int k = 0; k < max_columns; k++) {
            T * dptr = row.get(k);
            if (dptr == nullptr) {
              file << 0 << ",";
            } else {
              file << std::to_string(*dptr) << ",";
            }
          }
          file << *row.getClassification() << "\n";
        }
      }

      file.close();
    }

  }  // namespace IO
}

#endif //  OBAMADB_STORAGE_IO_H_
