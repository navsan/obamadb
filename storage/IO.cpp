#include "storage/exvector.h"
#include "storage/DataBlock.h"
#include "storage/Matrix.h"
#include "storage/MLTask.h"
#include "storage/SparseDataBlock.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <set>

#include "glog/logging.h"

#include "storage/IO.h"
#include "DenseDataBlock.h"

DECLARE_bool(liblinear);

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

    struct SynthMcParams {
      SynthMcParams(const std::string& file_name)
        : rows(0), cols(0), nnz(0), rank(0), seed(0) {
        std::ifstream infile;
        std::string line;
        infile.open(file_name.c_str(), std::ios::binary | std::ios::in);
        CHECK(infile.is_open());

        unsigned cursor = 0;
        std::getline(infile, line);
        // m,n,nnz,rank,seed
        int const num_params = 5;
        float params[num_params];
        for (int i = 0; i < num_params; i++) {
          params[i] = 0;
          scanForDouble(line.c_str(), &cursor, &params[i]);
          scanThroughWhitespace(line.c_str(), &cursor);
        }
        infile.close();

        CHECK_GT(params[0], 0);
        CHECK_GT(params[1], 0);
        CHECK_GT(params[2], 1);
        CHECK((params[3] > 1 && params[3] < 1000) || params[3] == -1.0) << "The rank must be greater than 1.";

        rows = params[0];
        cols = params[1];
        nnz = params[2];
        rank = params[3];
        seed = params[4];
      }

      int rows;
      int cols;
      int nnz;
      int rank;
      int seed;
    };

    void create_synth_mc_matrix(SynthMcParams const & params, obamadb::UnorderedMatrix *derived_mat) {
      int const rank = params.rank;
      int const rows = params.rows;
      int const cols = params.cols;
      int nnz = params.nnz;

      DenseDataBlock<num_t> lmat(rows, rank);
      DenseDataBlock<num_t> rmat(cols, rank);
      lmat.randomize();
      rmat.randomize();
      dvector<num_t> lrow_vec(0, nullptr);
      dvector<num_t> rrow_vec(0, nullptr);
      while (nnz-- != 0) {
        int rrow = rand() % rows;
        int rcol = rand() % cols;
        //dot, and add to derived mat.
        lmat.getRowVectorFast(rrow, &lrow_vec);
        rmat.getRowVectorFast(rcol, &rrow_vec);
        // We could also opt to add some noise at this step.
        derived_mat->append(rrow, rcol, obamadb::ml::dot(lrow_vec, rrow_vec.values_));
      }
      lmat.getRowVectorFast(rows, &lrow_vec);
      rmat.getRowVectorFast(cols, &rrow_vec);
      if (derived_mat->numRows() < rows || derived_mat->numColumns() < cols) {
        derived_mat->append(rows, cols, obamadb::ml::dot(lrow_vec, rrow_vec.values_));
      }
    }

    /**
     * Creates a synthetic dataset for matrix completion testing.
     * @param file_name
     * @return a matrix completion matrix to the standards of
     */
    UnorderedMatrix* load_synth_MC(const std::string& file_name) {
      SynthMcParams params(file_name);
      srand(params.seed);
      int const rank = params.rank;
      int const rows = params.rows;
      int const cols = params.cols;
      int nnz = params.nnz;

      obamadb::UnorderedMatrix *derived_mat = new obamadb::UnorderedMatrix();
      if (rank != -1) {
        // creates a matrix with a particular rank
        create_synth_mc_matrix(params, derived_mat);
      } else {
        while (nnz-- != 0) {
          int rrow = rand() % rows;
          int rcol = rand() % cols;
          derived_mat->append(rrow, rcol, std::fmod(rand() / 1e-6, 10.0));
        }
        // technically off by one, but who cares.
        if (derived_mat->numRows() < rows || derived_mat->numColumns() < cols) {
          derived_mat->append(rows, cols, std::fmod(rand() / 1e-6, 10.0));
        }
      }
      return derived_mat;
    }

    /**
     * Loads a libsvm style file into a sparse representation.
     * class (index:value )*\n
     */
    void loadLibsvmSparseBlocks(std::string const & file_name,
                                std::vector<obamadb::SparseDataBlock<num_t>*> &blocks) {
      std::size_t const BUFFER_SIZE = 32 * 1024;
      int fd = open(file_name.c_str(), O_RDONLY);
      CHECK_NE(fd, -1) << "Error opening file: " << file_name;
#ifndef __APPLE__
      // Advise the kernel of our access pattern.
      posix_fadvise(fd, 0, 0, 1);  // FDADVICE_SEQUENTIAL
#endif

      auto scanFloat = [](char* & cptr, char const * LIM, char const DELIM, float* val) -> bool {
        double i = 0;
        double neg = 1;
        int dp = 0;
        bool decimal = false;

        if (*cptr == '-') {
          neg = -1;
          cptr++;
        } else if (*cptr == '+') {
          cptr++;
        }

        while(cptr < LIM && *cptr != DELIM && *cptr != '\n') {
          if (*cptr == '.') {
            DCHECK_NE(true, decimal);
            decimal = true;
            cptr++;
            continue;
          }
          dp += decimal;

          i *= 10;
          i += *cptr - 48;
          cptr++;
        }

        if (cptr == LIM) {
          return true;
        } else {
          i *= neg;
          *val = (float) (i / pow(10.0, dp));
          if (*cptr != '\n')
            cptr++;
          return false;
        }
      };

      auto addToBlock = [](
        std::vector<obamadb::SparseDataBlock<num_t>*> &blocks,
        obamadb::SparseDataBlock<num_t>* &block,
        obamadb::svector<num_t> &sparse_row) {
        if (!block->appendRow(sparse_row)) {
          blocks.push_back(block);
          block = new SparseDataBlock<num_t>();
        }
      };

      obamadb::SparseDataBlock<num_t>* block = new obamadb::SparseDataBlock<num_t>();
      obamadb::svector<num_t> sparse_row;

      char buf[BUFFER_SIZE + 1];
      char * READ_LIM = buf + BUFFER_SIZE; // stopping point for new data was read in by the read call
      int offset = 0;
      while (size_t bytes_read = read(fd, buf + offset, BUFFER_SIZE - offset)) {
        CHECK_NE(bytes_read, (size_t) -1) << "Error reading file " << file_name;

        if (!bytes_read) {
          break;
        }

        READ_LIM = buf + bytes_read + offset;
        char *p = buf; // pointer to current char
        char *nl = p;  // pointer to start of last line
        while (true) {
          num_t cls = 0;
          nl = p;
          if(scanFloat(p, READ_LIM, ' ', &cls)) {
            break;
          } else {
            // ignore blank lines
            if (*p == '\n') {
              p++;
              continue;
            } else {
              sparse_row.setClassification(&cls);
            }
          }
          DCHECK(cls == -1 || cls == 1);

          bool eof = false;
          bool eol = false;
          while(!eol) {
            float idx = 0;
            num_t value = 0;
            eof = scanFloat(p, READ_LIM, ':', &idx);
            if (eof)
              break;
            eof = scanFloat(p, READ_LIM, ' ', &value);
            if (eof)
              break;
            sparse_row.push_back((int)idx, value);
            if(*p == '\n') {
              eol = true;
              p++;
            }
          }
          if (eof)
            break;
          else {
            DCHECK_EQ(true, eol);
          }
          addToBlock(blocks, block, sparse_row);
          sparse_row.clear();
        }
        sparse_row.clear();


        offset = READ_LIM - nl;
        memcpy(buf, nl, offset);
      }
      if (block->num_rows_ > 0) {
        blocks.push_back(block);
      }
    }

    /**
     * Load examples as an unordered matrix.
     * Scans a TSV file of the format
     * int1\tint2\tint3\n
     * where int1 specifies a row
     * int2 specifies a column
     * int3 specifies a value
     * Helper parser function which expects classifications to be set for each row.
     */
    UnorderedMatrix* loadUnorderedMatrix(const std::string& file_name) {
      if (file_name.find("_synth_mc_") != std::string::npos) {
        LOG(INFO) << "Loading a synthetic dataset";
        return load_synth_MC(file_name);
      }
      char const * fname = file_name.c_str();
      UnorderedMatrix* mat = new UnorderedMatrix();

      std::size_t const BUFFER_SIZE = 16 * 1024;
      int fd = open(fname, O_RDONLY);
      CHECK_NE(fd, -1) << "Error opening file: " << fname;
#ifndef __APPLE__
      // Advise the kernel of our access pattern.
      posix_fadvise(fd, 0, 0, 1);  // FDADVICE_SEQUENTIAL
#endif

      auto scanInt = [](char* & cptr, char const * LIM, char const DELIM, int* val) -> bool {
        while(cptr < LIM && *cptr != DELIM) {
          *val *= 10;
          *val += *cptr - 48;
          cptr++;
        }
        if (cptr == LIM) {
          return false;
        } else {
          cptr++;
          return true;
        }
      };

      char buf[BUFFER_SIZE + 1];
      char * READ_LIM = buf + BUFFER_SIZE; // stopping point for new data was read in by the read call
      int offset = 0;
      while (size_t bytes_read = read(fd, buf + offset, BUFFER_SIZE - offset)) {
        CHECK_NE(bytes_read, (size_t) -1) << "Error reading file " << fname;

        if (!bytes_read) {
          break;
        }

        READ_LIM = buf + bytes_read + offset;
        char *p = buf; // pointer to current char
        char *nl = p;  // pointer to start of last line
        while (true) {
          int i0 = 0;
          int i1 = 0;
          int i2 = 0;
          nl = p;
          if(!scanInt(p,READ_LIM,'\t', &i0))
            break;

          if(!scanInt(p,READ_LIM,'\t', &i1))
            break;

          if(!scanInt(p,READ_LIM,'\n', &i2))
            break;

          mat->append(i0,i1,i2);
        }
        offset = READ_LIM - nl;
        memcpy(buf, nl, offset);
      }

      return mat;
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

      if (FLAGS_liblinear) {
        DLOG(INFO) << "Loading file as liblinear style dataset";
        loadLibsvmSparseBlocks(file_name, blocks);
      } else {
        DLOG(INFO) << "Loading file as SVM-like matrix";
        loadSvmBlocks(infile, blocks);
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
        LOG(INFO) << "Loading a synthetic dataset";
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