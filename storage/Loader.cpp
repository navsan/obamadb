#include "storage/Loader.h"
#include "storage/DataBlock.h"

#include <iostream>
#include <cmath>

#include "glog/logging.h"


namespace obamadb {

  inline unsigned scanForInt(const char *str, unsigned int *cursor, double *value) {
    unsigned count = 0;

    bool negate = '-' == str[*cursor];
    if (negate) {
      *cursor += 1;
    }

    while (isdigit(str[*cursor])) {
      *value = *value * 10;
      *value = *value + static_cast<double>(str[*cursor] - 48);
      *cursor = *cursor + 1;
      count++;
    }

    if (negate) {
      *value = *value * -1;
    }

    return count;
  }

  inline void scanForDouble(const char *str, unsigned *cursor, double *value) {
    scanForInt(str, cursor, value);
    if (str[*cursor] == '.') {
      *cursor = *cursor + 1;
      double decimal = 0;
      unsigned count = scanForInt(str, cursor, &decimal);
      decimal = *value < 0 ? decimal * -1 : decimal;
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

  void Loader::save(const std::string &file_name, const DataBlock *datablock) {
    std::ofstream file;
    file.open(file_name, std::ios::out | std::ios::binary);

    CHECK(file.is_open()) << "Unable to open " << file_name << " for output.";

    if (datablock->getDataBlockType() == DataBlockType::kDense) {
      for (int i = 0; i < datablock->getNumRows(); i++) {
        double *row = datablock->getRow(i);
        for (int j = 0; j < datablock->getNumColumns(); j++) {
          file << row[j] << ", ";
        }
        file << "\n";
      }
    } else {
      const SparseDataBlock* sdatablock = dynamic_cast<const SparseDataBlock*>(datablock);
      file << *sdatablock;
    }

    file.close();
  }

  void scanSparseRowData(const std::string& line, double * row_id, double * attr_id, double * value) {
    // Assume it populates it here.
    char const *cstr = line.c_str();
    unsigned i = 0;
    scanThroughWhitespace(cstr, &i);
    CHECK_NE('\0', cstr[i]) << "Line of whitespace detected.";
    scanForDouble(cstr, &i, row_id);
    scanThroughWhitespace(cstr, &i);
    CHECK_NE('\0', cstr[i]) << "Malformed line.";
    scanForDouble(cstr, &i, attr_id);
    scanThroughWhitespace(cstr, &i);
    CHECK_NE('\0', cstr[i]) << "Malformed line.";
    scanForDouble(cstr, &i, value);
  }

  void Loader::loadFileToSparseDataBlocks(const std::string &file_name, std::vector<DataBlock *> &blocks) {
    if (!checkFileExists(file_name)) {
      return;
    }
    std::ifstream infile;
    std::string line;
    infile.open(file_name.c_str(), std::ios::binary | std::ios::in);

    SparseDataBlock *current_block = new SparseDataBlock();
    svector<double> temp_row;
    bool new_line = true;

    double last_id = -1;
    double id = -1;
    double idx = -1;
    double value = -1;
    double classification = -1;

    std::getline(infile, line);
    while (true) {
      if (new_line) {
        if (id != -1){
          // write vector to block.
          temp_row.push_back(classification);
          bool appended = current_block->appendRow(temp_row);
          if (!appended) {
            current_block->finalize();
            blocks.push_back(current_block);
            current_block = new SparseDataBlock();
            current_block->appendRow(temp_row);
          }
          temp_row.clear();
        }
        last_id = 0;
        idx = 0;
        classification = 0;
        scanSparseRowData(line, &last_id, &idx, &classification);
        new_line = false;
      } else {
        if(!std::getline(infile, line)) {
          // TODO: not DRY
          temp_row.push_back(classification);
          bool appended = current_block->appendRow(temp_row);
          if (!appended) {
            current_block->finalize();
            blocks.push_back(current_block);
            current_block = new SparseDataBlock();
            current_block->appendRow(temp_row);
          }
          break;
        }
        id = 0;
        idx = 0;
        value = 0;
        scanSparseRowData(line, &id, &idx, &value);
        if (id != last_id) {
          new_line = true;
        } else {
          temp_row.push_back(idx, value);
        }
      }
    }

    blocks.push_back(current_block);
    current_block->finalize();

    infile.close();
  }

}