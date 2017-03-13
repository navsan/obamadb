#include "storage/IO.h"

#include "storage/exvector.h"
#include "storage/DataBlock.h"
#include "storage/DenseDataBlock.h"
#include "storage/Matrix.h"
#include "storage/MLTask.h"
#include "storage/SparseDataBlock.h"
#include "storage/Utils.h"

#include <cmath>
#include <set>

#include "glog/logging.h"

namespace obamadb {

  typedef std::vector<obamadb::SparseDataBlock<num_t> *> BlockVector;

  namespace IO {
    /**
     * Creates a synthetic matrix with approximately some true rank.
     * @param rows
     * @param cols
     * @param nnz Number of non-zero entries
     * @param rank
     * @param derived_mat Matrix to add entries to.
     */
    void createSyntheticMcMatrix(
      int rows, int cols,
      int nnz, int rank,
      obamadb::UnorderedMatrix *derived_mat) {

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
     * @return a matrix completion matrix with the file's spec.
     */
    UnorderedMatrix* loadSyntheticMcMatrix(const std::string& file_name) {
      Scanner scanner(file_name);
      std::vector<double> params = scanner.scanLine();
      CHECK_EQ(5, params.size());
      CHECK_GT(params[0], 0);
      CHECK_GT(params[1], 0);
      CHECK_GT(params[2], 1);
      CHECK((params[3] > 1 && params[3] < 1000) || params[3] == -1.0) << "The rank must be greater than 1.";

      int const rows = static_cast<int>(params[0]);
      int const cols = static_cast<int>(params[1]);
      int nnz = static_cast<int>(params[2]);
      int const rank = static_cast<int>(params[3]);
      int const seed = static_cast<int>(params[4]);

      srand(seed);

      obamadb::UnorderedMatrix *derived_mat = new obamadb::UnorderedMatrix();
      if (rank != -1) {
        // creates a matrix with a particular rank
        createSyntheticMcMatrix(rows, cols, nnz, rank, derived_mat);
      } else {
        while (nnz-- != 1) {
          int rrow = rand() % rows;
          int rcol = rand() % cols;
          derived_mat->append(rrow, rcol, std::fmod(rand() / 1e-6, 10.0));
        }
        derived_mat->append(rows, cols, std::fmod(rand() / 1e-6, 10.0));
      }
      return derived_mat;
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
        LOG(INFO) << "Loading a synthetic dataset: " << file_name;
        return loadSyntheticMcMatrix(file_name);
      }
      UnorderedMatrix* mat = new UnorderedMatrix();

      Scanner scanner(file_name);
      std::vector<double> row = scanner.scanLine();
      while(row.size() > 0) {
        DCHECK_EQ(3, row.size());
        mat->append(static_cast<int>(row[0]), static_cast<int>(row[1]), row[2]);
        row = scanner.scanLine();
      }

      return mat;
    }

    /**
     * Loads blocks which are in the liblinear format
     * class index:value index:value ...
     * @param file_name
     * @return Block vector
     */
    template<>
    BlockVector loadBlocks(const std::string &file_name) {
      BlockVector blocks;

      // Helper function to add row to block.
      auto insertHelper = [&blocks](
        obamadb::SparseDataBlock<num_t>* &block,
        obamadb::svector<num_t> &sparse_row) {
        if (!block->appendRow(sparse_row)) {
          blocks.push_back(block);
          block = new SparseDataBlock<num_t>();
        }
      };

      obamadb::SparseDataBlock<num_t>* block = new obamadb::SparseDataBlock<num_t>();
      obamadb::svector<num_t> sparse_row;

      Scanner scanner(file_name);
      std::vector<double> line = scanner.scanLine();
      while(line.size() > 0) {
        DCHECK_EQ(1, line.size() % 2);
        sparse_row.setClassification(line[0]);
        for (int i = 1; i < line.size(); i+=2) {
          sparse_row.push_back(line[i], line[i+1]);
        }
        insertHelper(block, sparse_row);
        sparse_row.clear();
        line = scanner.scanLine();
      }

      if (block->num_rows_ > 0) {
        blocks.push_back(block);
      }
      return blocks;
    }

    BlockVector loadSyntheticBlocks(std::string const & file_name) {
      BlockVector blocks;
      Scanner scanner(file_name);

      std::vector<double> line = scanner.scanLine();

      double m = static_cast<int>(line[0]);
      double n = static_cast<int>(line[1]);
      double sigma = line[2];

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
        std::vector<obamadb::SparseDataBlock<num_t> *> blocks = loadSyntheticBlocks(filename);
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