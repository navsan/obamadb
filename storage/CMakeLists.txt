add_library(obamadb_storage_DataBlock
        DataBlock.cpp
        DataBlock.h)
add_library(obamadb_storage_DataView
        DataView.cpp
        DataView.h)
add_library(obamadb_storage_Loader
        Loader.cpp
        Loader.h)
add_library(obamadb_storage_LinearMath
        LinearMath.cpp
        LinearMath.h)
add_library(obamadb_storage_Task
        Task.cpp
        Task.h)
add_library(obamadb_storage_ThreadPool
        ThreadPool.cpp
        ThreadPool.h)
add_library(obamadb_storage_Utils
        ../empty_source.cpp
        Utils.h)
add_library(obamadb_storage_tests_StorageTestHelpers
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/StorageTestHelpers.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/StorageTestHelpers.h")

target_link_libraries(obamadb_storage_DataBlock
        obamadb_storage_Utils
        glog)
target_link_libraries(obamadb_storage_DataView
        glog
        obamadb_storage_DataBlock)
target_link_libraries(obamadb_storage_LinearMath
        glog
        obamadb_storage_DataBlock)
target_link_libraries(obamadb_storage_Loader
        glog
        obamadb_storage_DataBlock)
target_link_libraries(obamadb_storage_Task
        glog
        obamadb_storage_DataBlock
        obamadb_storage_LinearMath)
target_link_libraries(obamadb_storage_ThreadPool
        glog
        obamadb_storage_Task)
target_link_libraries(obamadb_storage_Utils
        glog)
target_link_libraries(obamadb_storage_tests_StorageTestHelpers
        glog
        obamadb_storage_DataBlock)

add_dependencies(obamadb_storage_Loader
        obamadb_storage_DataBlock)

add_executable(DataBlock_unittest
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/DataBlock_unittest.cpp")
target_link_libraries(DataBlock_unittest
        gtest
        gtest_main
        obamadb_storage_DataBlock
        obamadb_storage_LinearMath
        obamadb_storage_Loader
        obamadb_storage_Utils
        obamadb_storage_tests_StorageTestHelpers
        ${LIBS})
add_test(DataBlock_unittest DataBlock_unittest)

add_executable(Loader_unittest
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/Loader_unittest.cpp")
target_link_libraries(Loader_unittest
        gtest
        gtest_main
        obamadb_storage_Loader
        obamadb_storage_tests_StorageTestHelpers
        ${LIBS})
add_test(Loader_unittest Loader_unittest)

configure_file(tests/iris.dat iris.dat COPYONLY)
configure_file(tests/sparse.dat sparse.dat COPYONLY)