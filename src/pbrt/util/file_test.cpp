// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>
#include <pbrt/util/file.h>

using namespace pbrt;

static std::string inTestDir(const std::string &path) {
    return path;
}

TEST(File, HasExtension) {
    EXPECT_TRUE(HasExtension("foo.exr", "exr"));
    EXPECT_TRUE(HasExtension("foo.Exr", "exr"));
    EXPECT_TRUE(HasExtension("foo.Exr", "exR"));
    EXPECT_TRUE(HasExtension("foo.EXR", "exr"));
    EXPECT_FALSE(HasExtension("foo.xr", "exr"));
    EXPECT_FALSE(HasExtension("/foo/png", "ppm"));
}

TEST(File, RemoveExtension) {
    EXPECT_EQ(RemoveExtension("foo.exr"), "foo");
    EXPECT_EQ(RemoveExtension("fooexr"), "fooexr");
    EXPECT_EQ(RemoveExtension("foo.exr.png"), "foo.exr");
}

TEST(File, ReadWriteFile) {
    std::string fn = inTestDir("readwrite.txt");
    std::string str = "this is a test.";
    EXPECT_TRUE(WriteFile(fn, str));
    auto contents = ReadFileContents(fn);
    EXPECT_TRUE(contents.has_value());
    EXPECT_EQ(str, *contents);
    EXPECT_EQ(0, remove(fn.c_str()));
}

TEST(File, Success) {
    std::string fn = inTestDir("floatfile_good.txt");
    EXPECT_TRUE(WriteFile(fn, R"(1
# comment 6632
-2.5
#6502
3e2       -4.75E-1       5.25




6
)"));

    auto floats = ReadFloatFile(fn);
    EXPECT_TRUE(floats.has_value());
    EXPECT_EQ(6, floats->size());
    const float expected[] = {1.f, -2.5f, 300.f, -.475f, 5.25f, 6.f};
    for (int i = 0; i < PBRT_ARRAYSIZE(expected); ++i)
        EXPECT_EQ(expected[i], (*floats)[i]);

    EXPECT_EQ(0, remove(fn.c_str()));
}

TEST(File, Failures) {
    auto floats = ReadFloatFile("NO_SUCH_FILE_64622");
    EXPECT_FALSE(floats.has_value());

    std::string fn = inTestDir("floatfile_malformed.txt");
    EXPECT_TRUE(WriteFile(fn, R"(1
2 3 4
l5l
6


)"));
    floats = ReadFloatFile(fn);
    EXPECT_FALSE(floats.has_value());

    EXPECT_EQ(0, remove(fn.c_str()));
}
