#include "storage/exvector.h"
#include "storage/DataBlock.h"
#include "storage/Matrix.h"
#include "storage/SparseDataBlock.h"

#include <cmath>
#include <iostream>
#include <fstream>

#include "glog/logging.h"

#include "storage/IO.h"

namespace obamadb {

  namespace IO {

    inline unsigned scanForInt(const char *str, unsigned int *cursor, int_t *value) {
      unsigned count = 0;

      bool negate = '-' == str[*cursor];
      if (negate) {
        *cursor += 1;
      }

      while (isdigit(str[*cursor])) {
        *value = *value * 10;
        *value = *value + static_cast<int_t>(str[*cursor] - 48);
        *cursor = *cursor + 1;
        count++;
      }

      if (negate) {
        *value = *value * -1;
      }

      return count;
    }

    inline void scanForDouble(const char *str, unsigned *cursor, int_t *value) {
      scanForInt(str, cursor, value);
      if (str[*cursor] == '.') {
        *cursor = *cursor + 1;
        int_t decimal = 0;
        unsigned count = scanForInt(str, cursor, &decimal);
        decimal = *value < 0 ? decimal * -1 : decimal;

        // hack for integer-type test
        double mantissa = *value + (decimal / pow(10, count));
        *value = (int_t) (mantissa * kScaleFloats);

        *cursor = *cursor + 1;
      } else {
        // hack for integer-type test
        *value = (*value) * kScaleFloats;
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

    void scanSparseRowData(const std::string& line, int_t * row_id, int_t * attr_id, int_t * value) {
      // Assume it populates it here.
      char const *cstr = line.c_str();
      unsigned i = 0;
      scanThroughWhitespace(cstr, &i);
      CHECK_NE('\0', cstr[i]) << "Line of whitespace detected.";
      scanForInt(cstr, &i, row_id);
      scanThroughWhitespace(cstr, &i);
      CHECK_NE('\0', cstr[i]) << "Malformed line.";
      scanForInt(cstr, &i, attr_id);
      scanThroughWhitespace(cstr, &i);
      CHECK_NE('\0', cstr[i]) << "Malformed line.";

      // hack to read in normalized TF/IDF values as integers.
      scanForDouble(cstr, &i, value);
      //CHECK_GE(*value, 0);
    }

    template<class T>
    std::vector<SparseDataBlock<T>*> load_csv(const std::string& file_name){
      CHECK(false) << "Not implemented for the general case.";
    }

    template<>
    std::vector<obamadb::SparseDataBlock<int_t>*> load_csv(const std::string &file_name) {
      std::vector<obamadb::SparseDataBlock<int_t> *> blocks;
      // TODO: load a normal, "dense" CSV
      return blocks;
    }


    template<class T>
    std::vector<obamadb::SparseDataBlock<T>*> load_blocks(const std::string &file_name) {
      CHECK(false) << "Not implemented for the general case.";
    }

    template<>
    std::vector<obamadb::SparseDataBlock<int_t>*> load_blocks(const std::string &file_name) {
      std::vector<obamadb::SparseDataBlock<int_t>*> blocks;

      if (!checkFileExists(file_name)) {
        DCHECK(false) << "Could not open file for reading: " << file_name;
        return blocks;
      }

      std::ifstream infile;
      std::string line;
      infile.open(file_name.c_str(), std::ios::binary | std::ios::in);
      CHECK(infile.is_open());

      obamadb::SparseDataBlock<int_t> *current_block = new SparseDataBlock<int_t>();
      svector<int_t> temp_row;
      bool new_line = true;

      int_t last_id = -1;
      int_t id = -1;
      int_t idx = -1;
      int_t value = -1;
      int_t classification = -1;

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
              current_block = new SparseDataBlock<int_t>();
              current_block->appendRow(temp_row);
            }
            temp_row.clear();
          }
          last_id = 0;
          idx = 0;
          classification = 0;
          scanSparseRowData(line, &last_id, &idx, &classification);
          classification /= kScaleFloats;
          //CHECK(classification == -1 || classification == 1);
          new_line = false;
        } else {
          if(!std::getline(infile, line)) {
            // TODO: not DRY
            temp_row.setClassification(&classification);
            bool appended = current_block->appendRow(temp_row);
            if (!appended) {
              current_block->finalize();
              blocks.push_back(current_block);
              current_block = new SparseDataBlock<int_t>();
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

    void save(const std::string& file_name, const Matrix& mat) {
      save(file_name, mat.blocks_, mat.blocks_.size());
    }

    Matrix *load(const std::string &filename) {
      std::vector<obamadb::SparseDataBlock<int_t>*> blocks = load_blocks<int_t>(filename);
      Matrix *mat = new Matrix(blocks);
      return mat;
    }

  } // namespace IO

}