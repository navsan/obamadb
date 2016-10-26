#include "gtest/gtest.h"
#include "storage/Loader.h"
#include "storage/DataBlock.h"

#include "storage/tests/StorageTestHelpers.h"

namespace obamadb {

  TEST(LoaderTest, TestLoad) {
    const unsigned size_dataset = sizeof(iris_data)/sizeof(double);
    ASSERT_EQ(150 * 5, size_dataset);

    Loader loader;
    // There's only one block returned, so keep it in a unique ptr.
    std::unique_ptr<DenseDataBlock> block(*loader.load(test_file, false).begin());

    EXPECT_EQ(size_dataset, block->getSize());
    double *store = block->getStore();
    for (unsigned i = 0; i < size_dataset; i++) {
      EXPECT_EQ(iris_data[i], store[i]);
    }
  }

  TEST(LoaderTest, TestLoadSparse) {
    Loader load;
    std::vector<DataBlock*> blocks;
    load.loadFileToSparseDataBlocks("sparse.dat", blocks);
    ASSERT_EQ(1, blocks.size());
    std::unique_ptr<SparseDataBlock> block(dynamic_cast<SparseDataBlock*>(blocks[0]));

    EXPECT_EQ(2, block->getNumRows());
    EXPECT_EQ(24, block->getNumColumns());

    svector<double> row;
    block->getRowVector(0, &row);
    EXPECT_EQ(-1, row.values_[row.numElements() - 1]);

    std::cout << *block;
  }
}