add_library(obamadb_storage_loader
        Loader.cpp
        Loader.h)
add_library(obamadb_storage_datablock
        DataBlock.cpp
        DataBlock.h)

add_dependencies(obamadb_storage_loader
        glog
        obamadb_storage_datablock)
add_dependencies(obamadb_storage_datablock
        glog)


add_executable(DataBlock_unittest
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/DataBlock_unittest.cpp")
target_link_libraries(DataBlock_unittest
        gtest
        gtest_main
        obamadb_storage_datablock
        ${LIBS})
add_executable(Loader_unittest
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/Loader_unittest.cpp")
target_link_libraries(Loader_unittest
        gtest
        gtest_main
        obamadb_storage_loader
        ${LIBS})

configure_file(tests/iris.dat iris.dat COPYONLY)