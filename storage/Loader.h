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
    DataBlock* load(const std::string& file_name) {
      if (!checkFileExists(file_name)) {
        return nullptr;
      }
      return nullptr;
    }

  private:

    bool checkFileExists(const std::string& name) {
      std::ifstream f(name.c_str());
      return f.good();
    }

    DataBlock* loadFileToDataSet(const std::string& file_name) {
      // DCHECK(checkFileExists(file_name));

      std::ifstream infile;
      std::string line;
      infile.open(file_name.c_str(), std::ios::binary | std::ios::in);
      DataBlock *block = new DataBlock();
      while (std::getline(infile, line)) {
        const char *cstr = line.c_str();
        unsigned i = 0;
        while (cstr[i] != '\0') {
          scanThroughWhitespace(cstr, &i);
          if (cstr[i] == '\0')
            break;
          double value = 0;
          scanForDigits(cstr, &i, &value);
          block->append(&value);
        }
      }
      infile.close();
      return block;
    }

    inline unsigned scanForDigits(const char *str, unsigned *cursor, double *value) {
      unsigned count = 0;
      while (isdigit(str[*cursor])) {
        *value +=
        *cursor++;
      }
      return count;
    }

    inline void scanThroughWhitespace(const char *str, unsigned *cursor) {
      while (isspace(str[*cursor])) {
        *cursor++;
      }
    }

    bool ParseIntString(const std::string &int_string,
                        const char delimiter,
                        std::vector<int> *parsed_output) {
      std::vector<int>::size_type original_size = parsed_output->size();

      std::size_t pos = 0;
      while (pos < int_string.size()) {
        char *endptr = nullptr;
        int element = std::strtol(int_string.c_str() + pos, &endptr, 10);
        if ((endptr > int_string.c_str() + pos)
            && ((*endptr == delimiter) || (*endptr == '\0'))) {
          parsed_output->push_back(element);
          pos = endptr - int_string.c_str() + 1;
        } else {
          parsed_output->resize(original_size);
          return false;
        }
      }

      return true;
    }



  };

}

#endif //  OBAMADB_STORAGE_LOADER_H_
