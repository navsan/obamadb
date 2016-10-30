#include "storage/IO.h"
#include "storage/DataBlock.h"
#include "storage/SparseDataBlock.h"

#include <iostream>
#include <fstream>
#include <cmath>

#include "glog/logging.h"


namespace obamadb {

  namespace IO {

    inline unsigned scanForInt(const char *str, unsigned int *cursor, float_t *value) {
      unsigned count = 0;

      bool negate = '-' == str[*cursor];
      if (negate) {
        *cursor += 1;
      }

      while (isdigit(str[*cursor])) {
        *value = *value * 10;
        *value = *value + static_cast<float_t>(str[*cursor] - 48);
        *cursor = *cursor + 1;
        count++;
      }

      if (negate) {
        *value = *value * -1;
      }

      return count;
    }

    inline void scanForDouble(const char *str, unsigned *cursor, float_t *value) {
      scanForInt(str, cursor, value);
      if (str[*cursor] == '.') {
        *cursor = *cursor + 1;
        float_t decimal = 0;
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

    void scanSparseRowData(const std::string& line, float_t * row_id, float_t * attr_id, float_t * value) {
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

    template<class T>
    std::vector<obamadb::SparseDataBlock<T>*> load(const std::string &file_name) {
      CHECK(false) << "Not implemented for the general case.";
    }

    template<>
    std::vector<obamadb::SparseDataBlock<float_t>*> load(const std::string &file_name) {
      std::vector<obamadb::SparseDataBlock<float_t>*> blocks;

      if (!checkFileExists(file_name)) {
        DCHECK(false) << "Could not open file for reading: " << file_name;
        return blocks;
      }

      std::ifstream infile;
      std::string line;
      infile.open(file_name.c_str(), std::ios::binary | std::ios::in);
      CHECK(infile.is_open());

      obamadb::SparseDataBlock<float_t> *current_block = new SparseDataBlock<float_t>();
      se_vector<float_t> temp_row;
      bool new_line = true;

      float_t last_id = -1;
      float_t id = -1;
      float_t idx = -1;
      float_t value = -1;
      float_t classification = -1;

      std::getline(infile, line);
      while (true) {
        if (new_line) {
          if (id != -1){
            // write vector to block.
            temp_row.setClassification(&classification);

            bool appended = current_block->appendRow(temp_row);
            if (!appended) {
              current_block->finalize();
              blocks.push_back(current_block);
              current_block = new SparseDataBlock<float_t>();
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
            temp_row.setClassification(&classification);
            bool appended = current_block->appendRow(temp_row);
            if (!appended) {
              current_block->finalize();
              blocks.push_back(current_block);
              current_block = new SparseDataBlock<float_t>();
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

      infile.close();

      blocks.push_back(current_block);
      current_block->finalize();
      return blocks;
    }

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

    void save(const std::string& file_name, std::vector<SparseDataBlock<float_t>*> blocks, int nblocks) {
      std::ofstream file;
      file.open(file_name, std::ios::out | std::ios::binary);
      CHECK(file.is_open()) << "Unable to open " << file_name << " for output.";

      int max_columns = maxColumns<float_t>(blocks);

      DCHECK_LT(nblocks, blocks.size());

      for (int i = 0; i < nblocks; ++i) {
        const SparseDataBlock<float_t> &block = *blocks[i];
        se_vector<float_t> row;
        for (int j = 0; j < block.getNumRows(); j++) {
          block.getRowVector(j, &row);
          for (int k = 0; k < max_columns; k++) {
            float_t * dptr = row.get(k);
            if (dptr == nullptr) {
              file << 0 << ",";
            } else {
              file << *dptr << ",";
            }
          }
          file << *row.getClassification() << "\n";
        }
      }

      file.close();
    }

  } // namespace IO

}