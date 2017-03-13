#include "gtest/gtest.h"
#include "storage/DataBlock.h"
#include "storage/exvector.h"
#include "storage/IO.h"
#include "storage/Utils.h"

#include <cstdlib>
#include <memory>
#include <unordered_set>

namespace obamadb {

  TEST(UtilsTest, TestSVector) {
    svector<int> vec(10);
    for (int i = 0; i < 10; i++) {
      vec.push_back(i * 100, i);
    }
    EXPECT_EQ(901, vec.size());
    EXPECT_EQ(10, vec.numElements());
    EXPECT_EQ(nullptr, vec.get(2));
    EXPECT_EQ(2, *vec.get(200));
  }

  TEST(UtilsTest, TestSVectorMemory) {
    svector<int> vec(10);
    for (int i = 0; i < 10; i++) {
      vec.push_back(i * 100, i);
    }

    int cfn = -1;
    vec.setClassification(&cfn);

    std::unique_ptr<char> buffer(new char[vec.sizeBytes()]);
    vec.copyTo(buffer.get());
    // Read back out the vector.
    svector<int> vec2(vec.numElements(), buffer.get());

    EXPECT_EQ(901, vec2.size());
    EXPECT_EQ(10, vec2.numElements());
    EXPECT_EQ(nullptr, vec2.get(2));
    EXPECT_EQ(2, *vec2.get(200));
    EXPECT_EQ(*vec.class_, *vec2.class_);

    svector<int> vec3 = vec2;

    EXPECT_EQ(901, vec3.size());
    EXPECT_EQ(10, vec3.numElements());
    EXPECT_EQ(nullptr, vec3.get(2));
    EXPECT_EQ(2, *vec3.get(200));
    EXPECT_EQ(*vec.class_, *vec3.class_);
  }
}


