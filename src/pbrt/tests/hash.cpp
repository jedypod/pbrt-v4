
#include <gtest/gtest.h>

#include <pbrt/util/hash.h>

using namespace pbrt;

TEST(Hash, VarArgs) {
    int buf[] = { 1, -12511, 31415821, 37 };
    EXPECT_EQ(HashBuffer(buf, 4 * sizeof(int)),
              Hash(buf[0], buf[1], buf[2], buf[3]));
}
