
#include "storage/Loader.h"
#include "storage/DataBlock.h"

#include <iostream>

#include "glog/logging.h"


namespace obamadb {

  inline unsigned scanForDigits(const char *str, unsigned *cursor, double *value) {
    unsigned count = 0;
    while (isdigit(str[*cursor])) {
      *value = *value * 10;
      *value = *value + static_cast<double>(str[*cursor] - 48);
      *cursor = *cursor + 1;
      count++;
    }
    return count;
  }

  inline void scanThroughWhitespace(const char *str, unsigned *cursor) {
    while (isspace(str[*cursor])) {
      *cursor = *cursor + 1;
    }
  }

  bool checkFileExists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
  }

  DataBlock* Loader::loadFileToDataSet(const std::string& file_name) {
    DCHECK(checkFileExists(file_name));

    std::ifstream infile;
    std::string line;
    infile.open(file_name.c_str(), std::ios::binary | std::ios::in);
    unsigned elements_per_line = 0;
    unsigned line_number = 0;
    DataBlock *block = new DataBlock();
    while (std::getline(infile, line)) {
      const char *cstr = line.c_str();
      unsigned elements_this_line = 0;
      unsigned i = 0;
      while (cstr[i] != '\0') {
        scanThroughWhitespace(cstr, &i);
        if (cstr[i] == '\0')
          break;
        double value = 0;
        scanForDigits(cstr, &i, &value);
        block->append(&value);
        elements_this_line++;
      }
      if (0 == elements_per_line) {
        elements_per_line = elements_this_line;
      } else if (elements_per_line != elements_this_line) {
        LOG(WARNING) << std::to_string(line_number) << "scanned line element count not consistent with previous lines";
      }
      line_number++;
    }
    infile.close();
    block->setWidth(elements_per_line);
    return block;
  }

  DataBlock* Loader::load(const std::string& file_name) {
    if (!checkFileExists(file_name)) {
      return nullptr;
    }

    return Loader::loadFileToDataSet(file_name);
  }

}