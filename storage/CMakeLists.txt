add_library(obamadb_storage_Loader
        Loader.cpp
        Loader.h)
add_library(obamadb_storage_Datablock
        DataBlock.cpp
        DataBlock.h)
add_library(obamadb_storage_tests_StorageTestHelpers
        ../empty_src.cpp
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/StorageTestHelpers.h")

target_link_libraries(obamadb_storage_Datablock
        glog)
target_link_libraries(obamadb_storage_Loader
        glog)

add_dependencies(obamadb_storage_Loader
        obamadb_storage_Datablock)

add_executable(DataBlock_unittest
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/DataBlock_unittest.cpp")
target_link_libraries(DataBlock_unittest
        gtest
        gtest_main
        obamadb_storage_Datablock
        obamadb_storage_Loader
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