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

    inline unsigned scanForInt(const char *str, unsigned int *cursor, num_t *value) {
      unsigned count = 0;

      bool negate = '-' == str[*cursor];
      if (negate) {
        *cursor += 1;
      }

      while (isdigit(str[*cursor])) {
        *value = *value * 10;
        *value = *value + static_cast<num_t>(str[*cursor] - 48);
        *cursor = *cursor + 1;
        count++;
      }

      if (negate) {
        *value = *value * -1;
      }

      return count;
    }

    inline void scanForDouble(const char *str, unsigned *cursor, float *value) {
      scanForInt(str, cursor, value);
      if (str[*cursor] == '.') {
        *cursor = *cursor + 1;
        float decimal = 0;
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

    void scanSparseRowData(const std::string& line, num_t * row_id, num_t * attr_id, num_t * value) {
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
    std::vector<obamadb::SparseDataBlock<T>*> loadBlocks(const std::string &file_name) {
      CHECK(false) << "Not implemented for the general case.";
    }

    /**
     * Checks if the data looks like it is for the sparse SVM.
     * @param infile
     * @return true if for an SVM.
     */
    bool isSvmLike(std::ifstream& infile) {
      bool svmLike = false;
      std::string line;
      std::getline(infile, line);
      unsigned cursor = 0;
      float val = 0;
      scanForDouble(line.c_str(), &cursor, &val);
      scanThroughWhitespace(line.c_str(), &cursor);
      val = 0;
      scanForDouble(line.c_str(), &cursor, &val); // the second value will be -1, a class.
      if (val < 0) {
        svmLike = true;
      }
      infile.seekg(0);
      return svmLike;
    }

    /**
     * Helper parser function which expects classifications to be set for each row.
     * @param infile The open file handle
     * @param blocks Vector to dump finished blocks into.
     */
    void loadSvmBlocks(std::ifstream& infile, std::vector<obamadb::SparseDataBlock<num_t>*> &blocks) {
      std::string line;
      obamadb::SparseDataBlock<num_t> *current_block = new SparseDataBlock<num_t>();
      svector<num_t> temp_row;
      bool new_line = true;

      num_t last_id = -1;
      num_t id = -1;
      num_t idx = -1;
      num_t value = -1;
      num_t classification = -1;

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
              current_block = new SparseDataBlock<num_t>();
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
              current_block = new SparseDataBlock<num_t>();
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
    }

    /**
     * Load Matrix Completion blocks.
     * Helper parser function which expects classifications to be set for each row.
     * @param infile The open file handle
     * @param blocks Vector to dump finished blocks into.
     */
    void loadMcBlocks(std::ifstream& infile, std::vector<obamadb::SparseDataBlock<num_t>*> &blocks) {
      // helper function to add a row.
      auto addRow = [&blocks](obamadb::SparseDataBlock<num_t>** currentBlock, svector<num_t>& row) -> void {
        if (!(*currentBlock)->appendRow(row)) {
          blocks.push_back(*currentBlock);
          *currentBlock = new SparseDataBlock<num_t>();
          CHECK((*currentBlock)->appendRow(row));
        }
      };

      // returns if we have reached the end of the file.
      auto scanForRow = [](std::ifstream& is, svector<num_t>& row_vector, int* row_id) -> bool {
        std::string line;
        int currentRow = -1;
        bool eof = true;
        while(std::getline(is, line)) {
          num_t row = 0;
          num_t column = 0;
          num_t val = 0;
          unsigned c = 0;
          char const * cstr = line.c_str();
          scanForInt(cstr, &c, &row);
          scanThroughWhitespace(cstr, &c);
          scanForInt(cstr, &c, &column);
          scanThroughWhitespace(cstr, &c);
          scanForInt(cstr, &c, &val);

          row_vector.push_back(column, val);

          if (currentRow == -1) {
            currentRow = row;
            *row_id = currentRow;
            row_vector.setClassification((num_t*)&currentRow);
          } else if (row != currentRow) {
            is.seekg(-1 * (int)c, is.cur);
            eof = false;
            break;
          }
        }
        return eof;
      };

//      obamadb::SparseDataBlock<num_t> *current_block = new obamadb::SparseDataBlock<num_t>();
//      svector<num_t> temp_row;
//      int row_id = 0;
//      while(scanForRow(infile, temp_row, &row_id)) {
//        while(current_block.)
//      }
//      current_block->finalize();
    }

    template<>
    std::vector<obamadb::SparseDataBlock<num_t>*> loadBlocks(const std::string &file_name) {
      std::vector<obamadb::SparseDataBlock<num_t>*> blocks;

      if (!checkFileExists(file_name)) {
        CHECK(false) << "Could not open file for reading: " << file_name;
        return blocks;
      }

      std::ifstream infile;
      infile.open(file_name.c_str(), std::ios::binary | std::ios::in);
      CHECK(infile.is_open());

      if (isSvmLike(infile)) {
        DLOG(INFO) << "Loading file as SVM-like matrix";
        loadSvmBlocks(infile, blocks);
      } else {
        DLOG(INFO) << "Loading file as Matrix Completion-like matrix";
        loadMcBlocks(infile, blocks);
      }

      infile.close();
      return blocks;
    }

    std::vector<obamadb::SparseDataBlock<num_t> *> load_synthetic_blocks(std::string const & file_name) {
      std::vector<obamadb::SparseDataBlock<num_t>*> blocks;

      // extract params (m n sigma)
      if (!checkFileExists(file_name)) {
        DCHECK(false) << "Could not open file for reading: " << file_name;
        return blocks;
      }

      std::ifstream infile;
      std::string line;
      infile.open(file_name.c_str(), std::ios::binary | std::ios::in);
      CHECK(infile.is_open());

      unsigned cursor = 0;
      std::getline(infile, line);
      float m = 0;
      float n = 0;
      float sigma = 0;
      scanForDouble(line.c_str(), &cursor, &m);
      scanThroughWhitespace(line.c_str(), &cursor);
      scanForDouble(line.c_str(), &cursor, &n);
      scanThroughWhitespace(line.c_str(), &cursor);
      scanForDouble(line.c_str(), &cursor, &sigma);
      infile.close();

      CHECK_GT(m, 0);
      CHECK_GT(n, 0);
      CHECK(sigma <= 1.0 && sigma > 0.0);
      int total_rows = 0;
      while (total_rows < m) {
        blocks.push_back(obamadb::GetRandomSparseDataBlock(kStorageBlockSize, n, 1.0 - sigma));
        total_rows += blocks.back()->num_rows_;
      }
      blocks.back()->trimRows(total_rows - m);
      return blocks;
    }

    Matrix *load(const std::string &filename) {
      std::string const synth_str("_synth_svm_");
      Matrix *mat = nullptr;
      if (filename.find(synth_str) != std::string::npos) {
        // this file contains synthetic data params
        std::vector<obamadb::SparseDataBlock<num_t> *> blocks = load_synthetic_blocks(filename);
        mat = new Matrix(blocks);
      } else {
        std::vector<obamadb::SparseDataBlock<num_t> *> blocks = loadBlocks<num_t>(filename);
        mat = new Matrix(blocks);
      }
      return mat;
    }

    void save(const std::string& file_name, const Matrix& mat) {
      save(file_name, mat.blocks_, mat.blocks_.size());
    }

  } // namespace IO

}