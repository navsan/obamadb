#include "storage/Loader.h"
#include "storage/DataBlock.h"

#include <iostream>
#include <cmath>

#include "glog/logging.h"


namespace obamadb {

  inline unsigned scanForInt(const char *str, unsigned int *cursor, double *value) {
    while (isdigit(str[*cursor])) {
      *value = *value * 10;
      *value = *value + static_cast<double>(str[*cursor] - 48);
      *cursor = *cursor + 1;
    }
  }

  inline void scanForDouble(const char *str, unsigned *cursor, double *value) {
    scanForInt(str, cursor, value);
    if (str[*cursor] == '.') {
      double decimal = 0;
      unsigned count = scanForInt(str, cursor, &decimal);
      *value = *value + (decimal / pow(10, count));
      *cursor = *cursor + 1;
    }
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

  void Loader::loadFileToDataBlocks(const std::string &file_name, std::vector<DenseDataBlock*>& blocks) {
    DCHECK(checkFileExists(file_name));

    std::ifstream infile;
    std::string line;
    infile.open(file_name.c_str(), std::ios::binary | std::ios::in);

    unsigned line_number = 0;
    DenseDataBlock *current_block = new DenseDataBlock();
    bool new_block = true;

    while (std::getline(infile, line)) {
      int elements_this_line = scanLine(line, current_block);

      if (-1 == elements_this_line) {
        blocks.push_back(current_block);
        current_block = new DenseDataBlock();
        new_block = true;
      } else if (new_block) {
        current_block->setWidth(elements_this_line);
        new_block = false;
      } else if (current_block->getNumColumns() != elements_this_line) {
        LOG(WARNING) << std::to_string(line_number)
                     << "scanned line element count not consistent with previous lines.";
      }
      line_number++;
    }
    blocks.push_back(current_block);

    infile.close();
  }

  int Loader::scanLine(const std::string &line, DenseDataBlock *block) {
    const char *cstr = line.c_str();
    int elements_this_line = 0;
    unsigned i = 0;
    while (cstr[i] != '\0' && block->getRemainingElements() > 0) {
      scanThroughWhitespace(cstr, &i);
      if (cstr[i] == '\0')
        break;
      double value = 0;
      scanForDouble(cstr, &i, &value);
      block->append(&value);
      elements_this_line++;
    }

    if (cstr[i] != '\0') {
      block->elements_ = block->elements_ - elements_this_line;
      return -1;
    }

    return elements_this_line;
  }

  std::vector<DenseDataBlock*> Loader::load(const std::string& file_name, bool sparse) {
    std::vector<DenseDataBlock*> blocks;

    if (!checkFileExists(file_name)) {
      return blocks;
    }

    if (sparse) {
      CHECK(false) << "Haven't implemented this yet" << std::endl;
    }

    Loader::loadFileToDataBlocks(file_name, blocks);
    return blocks;
  }

  void Loader::save(const std::string &file_name, const DenseDataBlock &datablock) {
    std::ofstream file;
    file.open(file_name, std::ios::out | std::ios::binary);

    CHECK(file.is_open()) << "Unable to open " << file_name << " for output.";

    for(int i = 0; i < datablock.getNumRows(); i++) {
      double *row = datablock.getRow(i);
      for(int j = 0; j < datablock.getNumColumns(); j++) {
       file << row[j] << ", ";
      }
      file << "\n";
    }
    file.close();
  }

}