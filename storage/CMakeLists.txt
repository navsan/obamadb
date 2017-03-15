add_library(obamadb_storage_DataBlock
        DataBlock.cpp
        DataBlock.h)
add_library(obamadb_storage_DataView
        DataView.cpp
        DataView.h)
add_library(obamadb_storage_DenseDataBlock
        DenseDataBlock.cpp
        DenseDataBlock.h)
add_library(obamadb_storage_exvector
        exvector.cpp
        exvector.h)
add_library(obamadb_storage_IO
        IO.cpp
        IO.h)
add_library(obamadb_storage_Matrix
        Matrix.cpp
        Matrix.h)
add_library(obamadb_storage_SparseDataBlock
        SparseDataBlock.cpp
        SparseDataBlock.h)
add_library(obamadb_storage_StorageConstants
        StorageConstants.h
        StorageConstants.cpp)
add_library(obamadb_storage_ThreadPool
        ThreadPool.cpp
        ThreadPool.h)
add_library(obamadb_storage_UnorderedMatrix
        UnorderedMatrix.cpp
        UnorderedMatrix.h)
add_library(obamadb_storage_Utils
        Utils.cpp
        Utils.h)

target_link_libraries(obamadb_storage_DataBlock
        glog
        obamadb_storage_exvector
        obamadb_storage_StorageConstants
        obamadb_storage_Utils)
target_link_libraries(obamadb_storage_DataView
        glog
        obamadb_storage_DataBlock
        obamadb_storage_exvector
        obamadb_storage_SparseDataBlock)
target_link_libraries(obamadb_storage_DenseDataBlock
        glog
        obamadb_storage_DataBlock
        obamadb_storage_exvector
        obamadb_storage_StorageConstants
        obamadb_storage_Utils)
target_link_libraries(obamadb_storage_exvector
        glog)
target_link_libraries(obamadb_storage_IO
        glog
        gflags
        obamadb_storage_DataBlock
        obamadb_storage_exvector
        obamadb_storage_Matrix
        obamadb_ml_MLTask
        obamadb_storage_SparseDataBlock
        obamadb_storage_StorageConstants
        obamadb_storage_UnorderedMatrix
        obamadb_storage_Utils)
target_link_libraries(obamadb_storage_Matrix
        glog
        obamadb_storage_DataBlock
        obamadb_storage_exvector
        obamadb_storage_SparseDataBlock
        obamadb_storage_StorageConstants
        obamadb_storage_ThreadPool
        obamadb_storage_Utils)
target_link_libraries(obamadb_storage_SparseDataBlock
        glog
        obamadb_storage_DataBlock
        obamadb_storage_exvector)
target_link_libraries(obamadb_storage_ThreadPool
        glog)
target_link_libraries(obamadb_storage_UnorderedMatrix
        glog
        obamadb_storage_StorageConstants)
target_link_libraries(obamadb_storage_Utils
        glog
        gflags)

add_executable(DenseDataBlock_unittest
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/DenseDataBlock_unittest.cpp")
target_link_libraries(DenseDataBlock_unittest
        gtest
        gtest_main
        obamadb_storage_DenseDataBlock
        obamadb_storage_exvector
        obamadb_storage_IO
        obamadb_storage_Utils
        ${LIBS})
add_test(DenseDataBlock_unittest DenseDataBlock_unittest)

add_executable(IO_unittest
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/IO_unittest.cpp")
target_link_libraries(IO_unittest
        gtest
        gtest_main
        gflags
        obamadb_storage_DataBlock
        obamadb_storage_exvector
        obamadb_storage_IO
        obamadb_storage_SparseDataBlock
        ${LIBS})
add_test(IO_unittest IO_unittest)

add_executable(Matrix_unittest
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/Matrix_unittest.cpp")
target_link_libraries(Matrix_unittest
        gtest
        gtest_main
        gflags
        obamadb_storage_exvector
        obamadb_storage_IO
        obamadb_storage_Matrix
        obamadb_storage_SparseDataBlock
        obamadb_storage_Utils
        ${LIBS})
add_test(Matrix_unittest Matrix_unittest)

add_executable(SparseDataBlock_unittest
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/SparseDataBlock_unittest.cpp")
target_link_libraries(SparseDataBlock_unittest
        gtest
        gtest_main
        obamadb_storage_DataBlock
        obamadb_storage_exvector
        obamadb_storage_IO
        obamadb_storage_Utils
        ${LIBS})
add_test(SparseDataBlock_unittest SparseDataBlock_unittest)

add_executable(Utils_unittest
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/Utils_unittest.cpp")
target_link_libraries(Utils_unittest
        gtest
        gtest_main
        obamadb_storage_DataBlock
        obamadb_storage_exvector
        obamadb_storage_IO
        obamadb_storage_Utils
        ${LIBS})
add_test(Utils_unittest Utils_unittest)

configure_file(tests/iris.dat iris.dat COPYONLY)
configure_file(tests/sparse.dat sparse.dat COPYONLY)
configure_file(tests/heart_scale.dat heart_scale.dat COPYONLY)
