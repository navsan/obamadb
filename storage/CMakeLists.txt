add_library(obamadb_storage_DataBlock
        DataBlock.cpp
        DataBlock.h)
add_library(obamadb_storage_DataView
        DataView.cpp
        DataView.h)
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
add_library(obamadb_storage_Task
        Task.cpp
        Task.h)
add_library(obamadb_storage_ThreadPool
        ThreadPool.cpp
        ThreadPool.h)
add_library(obamadb_storage_Utils
        ../empty_src.cpp
        Utils.h)
add_library(obamadb_storage_tests_StorageTestHelpers
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/StorageTestHelpers.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/StorageTestHelpers.h")

target_link_libraries(obamadb_storage_DataBlock
        obamadb_storage_StorageConstants
        obamadb_storage_Utils
        glog)
target_link_libraries(obamadb_storage_DataView
        glog
        obamadb_storage_DataBlock
        obamadb_storage_SparseDataBlock)
target_link_libraries(obamadb_storage_IO
        glog
        obamadb_storage_DataBlock
        obamadb_storage_SparseDataBlock
        obamadb_storage_StorageConstants)
target_link_libraries(obamadb_storage_Matrix
        glog
        obamadb_storage_DataBlock
        obamadb_storage_SparseDataBlock
        obamadb_storage_StorageConstants
        obamadb_storage_Utils)
target_link_libraries(obamadb_storage_SparseDataBlock
        glog
        obamadb_storage_DataBlock)
target_link_libraries(obamadb_storage_Task
        glog
        obamadb_storage_DataBlock
        obamadb_storage_SparseDataBlock
        obamadb_storage_Utils)
target_link_libraries(obamadb_storage_ThreadPool
        glog
        obamadb_storage_Task)
target_link_libraries(obamadb_storage_Utils
        glog)
target_link_libraries(obamadb_storage_tests_StorageTestHelpers
        glog)

add_executable(IO_unittest
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/IO_unittest.cpp")
target_link_libraries(IO_unittest
        gtest
        gtest_main
        obamadb_storage_DataBlock
        obamadb_storage_IO
        obamadb_storage_SparseDataBlock
        obamadb_storage_tests_StorageTestHelpers
        ${LIBS})
add_test(IO_unittest IO_unittest)

add_executable(Matrix_unittest
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/Matrix_unittest.cpp")
target_link_libraries(Matrix_unittest
        gtest
        gtest_main
        obamadb_storage_IO
        obamadb_storage_Matrix
        obamadb_storage_SparseDataBlock
        obamadb_storage_Utils
        obamadb_storage_tests_StorageTestHelpers
        ${LIBS})
add_test(Matrix_unittest Matrix_unittest)

add_executable(SparseDataBlock_unittest
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/SparseDataBlock_unittest.cpp")
target_link_libraries(SparseDataBlock_unittest
        gtest
        gtest_main
        obamadb_storage_DataBlock
        obamadb_storage_IO
        obamadb_storage_Utils
        obamadb_storage_tests_StorageTestHelpers
        ${LIBS})
add_test(SparseDataBlock_unittest SparseDataBlock_unittest)

add_executable(Utils_unittest
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/Utils_unittest.cpp")
target_link_libraries(Utils_unittest
        gtest
        gtest_main
        obamadb_storage_DataBlock
        obamadb_storage_IO
        obamadb_storage_Utils
        obamadb_storage_tests_StorageTestHelpers
        ${LIBS})
add_test(Utils_unittest Utils_unittest)

configure_file(tests/iris.dat iris.dat COPYONLY)
configure_file(tests/sparse.dat sparse.dat COPYONLY)