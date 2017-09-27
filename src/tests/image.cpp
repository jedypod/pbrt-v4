
#include "tests/gtest/gtest.h"

#include "pbrt.h"
#include "image.h"
#include "util/rng.h"
#include "mipmap.h"
#include "half.h"

#include <algorithm>

using namespace pbrt;

// TODO:
// for tga and png i/o: test mono and rgb; make sure mono is smaller
// pixel bounds stuff... (including i/o paths...)
// basic lookups, bilerps, etc
//   also clamp, repeat, etc...
// resize?
// round trip: init, write, read, check
// FlipY()

TEST(Image, Basics) {
    Image y8(PixelFormat::Y8, {4, 8});
    EXPECT_EQ(y8.nChannels(), 1);
    EXPECT_EQ(y8.BytesUsed(), y8.resolution[0] * y8.resolution[1]);

    Image sy8(PixelFormat::SY8, {4, 8});
    EXPECT_EQ(sy8.nChannels(), 1);
    EXPECT_EQ(sy8.BytesUsed(), sy8.resolution[0] * sy8.resolution[1]);

    Image y16(PixelFormat::Y16, {4, 8});
    EXPECT_EQ(y16.nChannels(), 1);
    EXPECT_EQ(y16.BytesUsed(), 2 * y16.resolution[0] * y16.resolution[1]);

    Image y32(PixelFormat::Y32, {4, 8});
    EXPECT_EQ(y32.nChannels(), 1);
    EXPECT_EQ(y32.BytesUsed(), 4 * y32.resolution[0] * y32.resolution[1]);

    Image rgb8(PixelFormat::RGB8, {4, 8});
    EXPECT_EQ(rgb8.nChannels(), 3);
    EXPECT_EQ(rgb8.BytesUsed(), 3 * rgb8.resolution[0] * rgb8.resolution[1]);

    Image srgb8(PixelFormat::SRGB8, {4, 8});
    EXPECT_EQ(srgb8.nChannels(), 3);
    EXPECT_EQ(srgb8.BytesUsed(), 3 * srgb8.resolution[0] * srgb8.resolution[1]);

    Image rgb16(PixelFormat::RGB16, {4, 16});
    EXPECT_EQ(rgb16.nChannels(), 3);
    EXPECT_EQ(rgb16.BytesUsed(),
              2 * 3 * rgb16.resolution[0] * rgb16.resolution[1]);

    Image rgb32(PixelFormat::RGB32, {4, 32});
    EXPECT_EQ(rgb32.nChannels(), 3);
    EXPECT_EQ(rgb32.BytesUsed(),
              4 * 3 * rgb32.resolution[0] * rgb32.resolution[1]);
}

static Float sRGBRoundTrip(Float v) {
    if (v < 0) return 0;
    else if (v > 1) return 1;
    uint8_t encoded = LinearToSRGB8(v);
    return SRGB8ToLinear(encoded);
}

static std::vector<uint8_t> GetInt8Pixels(Point2i res, int nc) {
    std::vector<uint8_t> r;
    for (int y = 0; y < res[1]; ++y)
        for (int x = 0; x < res[0]; ++x)
            for (int c = 0; c < nc; ++c) r.push_back((x * y + c) % 255);
    return r;
}

static std::vector<Float> GetFloatPixels(Point2i res, int nc) {
    std::vector<Float> p;
    for (int y = 0; y < res[1]; ++y)
        for (int x = 0; x < res[0]; ++x)
            for (int c = 0; c < nc; ++c)
                p.push_back(-.25 +
                            2. * (c + 3 * x + 3 * y * res[0]) /
                                (res[0] * res[1]));
    return p;
}

static Float modelQuantization(Float value, PixelFormat format) {
    switch (format) {
    case PixelFormat::SY8:
    case PixelFormat::SRGB8:
        return sRGBRoundTrip(value);
    case PixelFormat::Y8:
    case PixelFormat::RGB8:
        return Clamp((value * 255.f) + 0.5f, 0, 255) * (1.f / 255.f);
    case PixelFormat::Y16:
    case PixelFormat::RGB16:
        return Float(Half(value));
    case PixelFormat::Y32:
    case PixelFormat::RGB32:
        return value;
    default:
        LOG(FATAL) << "Unhandled pixel format";
    }
}

TEST(Image, GetSetY) {
    Point2i res(9, 3);
    std::vector<Float> yPixels = GetFloatPixels(res, 1);

    for (auto format : {PixelFormat::Y8, PixelFormat::SY8, PixelFormat::Y16,
                        PixelFormat::Y32}) {
        Image image(format, res);
        for (int y = 0; y < res[1]; ++y)
            for (int x = 0; x < res[0]; ++x) {
                image.SetChannel({x, y}, 0, yPixels[y * res[0] + x]);
            }
        for (int y = 0; y < res[1]; ++y)
            for (int x = 0; x < res[0]; ++x) {
                Float v = image.GetChannel({x, y}, 0);
                EXPECT_EQ(v, image.GetY({x, y}));
                if (format == PixelFormat::Y8)
                    EXPECT_LT(std::abs(v - Clamp(yPixels[y * res[0] + x], 0, 1)),
                              0.501f / 255.f);
                else
                    EXPECT_EQ(v, modelQuantization(yPixels[y * res[0] + x], format));
            }
    }
}

TEST(Image, GetSetRGB) {
    Point2i res(7, 32);
    std::vector<Float> rgbPixels = GetFloatPixels(res, 3);

    for (auto format : {PixelFormat::RGB8, PixelFormat::SRGB8,
                        PixelFormat::RGB16, PixelFormat::RGB32}) {
        Image image(format, res);
        for (int y = 0; y < res[1]; ++y)
            for (int x = 0; x < res[0]; ++x)
                for (int c = 0; c < 3; ++c)
                    image.SetChannel({x, y}, c,
                                     rgbPixels[3 * y * res[0] + 3 * x + c]);

        for (int y = 0; y < res[1]; ++y)
            for (int x = 0; x < res[0]; ++x) {
                Spectrum s = image.GetSpectrum({x, y});
                std::array<Float, 3> rgb = s.ToRGB();

                for (int c = 0; c < 3; ++c) {
                    // This is assuming Spectrum==RGBSpectrum, which is bad.
                    ASSERT_EQ(sizeof(RGBSpectrum), sizeof(Spectrum));

                    EXPECT_EQ(rgb[c], image.GetChannel({x, y}, c));
                    int offset = 3 * y * res[0] + 3 * x + c;
                    if (format == PixelFormat::RGB8)
                        EXPECT_LT(std::abs(rgb[c] - Clamp(rgbPixels[offset], 0, 1)),
                                  0.501f / 255.f);
                    else {
                        Float qv = modelQuantization(rgbPixels[offset], format);
                        EXPECT_EQ(rgb[c], qv);
                    }
                }
            }
    }
}

TEST(Image, CopyRectOut) {
    Point2i res(29, 14);

    for (auto format : {PixelFormat::SY8, PixelFormat::Y8,
                PixelFormat::SRGB8, PixelFormat::RGB8,
                PixelFormat::Y16, PixelFormat::RGB16,
                PixelFormat::Y32, PixelFormat::RGB32}) {
        int nc = nChannels(format);
        std::vector<Float> orig = GetFloatPixels(res, nc);

        Image image(format, res);
        auto origIter = orig.begin();
        for (int y = 0; y < res[1]; ++y)
            for (int x = 0; x < res[0]; ++x)
                for (int c = 0; c < nc; ++c, ++origIter)
                    image.SetChannel({x, y}, c, *origIter);

        Bounds2i extent(Point2i(2, 3), Point2i(5, 10));
        std::vector<Float> buf(extent.Area() * nc);

        image.CopyRectOut(extent, absl::MakeSpan(buf));

        // Iterate through the points in the extent and the buffer
        // together.
        auto bufIter = buf.begin();
        for (auto pIter = begin(extent); pIter != end(extent); ++pIter) {
            for (int c = 0; c < nc; ++c) {
                ASSERT_FALSE(bufIter == buf.end());
                EXPECT_EQ(*bufIter, image.GetChannel(*pIter, c));
                ++bufIter;
            }
        }
    }
}

TEST(Image, CopyRectIn) {
    Point2i res(17, 32);
    RNG rng;

    for (auto format : {PixelFormat::SY8, PixelFormat::Y8,
                PixelFormat::SRGB8, PixelFormat::RGB8,
                PixelFormat::Y16, PixelFormat::RGB16,
                PixelFormat::Y32, PixelFormat::RGB32}) {
        int nc = nChannels(format);
        std::vector<Float> orig = GetFloatPixels(res, nc);

        Image image(format, res);
        auto origIter = orig.begin();
        for (int y = 0; y < res[1]; ++y)
            for (int x = 0; x < res[0]; ++x)
                for (int c = 0; c < nc; ++c, ++origIter)
                    image.SetChannel({x, y}, c, *origIter);

        Bounds2i extent(Point2i(10, 23), Point2i(17, 28));
        std::vector<Float> buf(extent.Area() * nc);
        std::generate(buf.begin(), buf.end(),
                      [&rng]() { return rng.UniformFloat(); });

        image.CopyRectIn(extent, buf);

        // Iterate through the points in the extent and the buffer
        // together.
        auto bufIter = buf.begin();
        for (auto pIter = begin(extent); pIter != end(extent); ++pIter) {
            for (int c = 0; c < nc; ++c) {
                ASSERT_FALSE(bufIter == buf.end());
                if (format == PixelFormat::Y8 || format == PixelFormat::RGB8) {
                    Float err = std::abs(image.GetChannel(*pIter, c) -
                                         Clamp(*bufIter, 0, 1));
                    EXPECT_LT(err, 0.501f / 255.f);
                } else {
                    Float qv = modelQuantization(*bufIter, format);
                    EXPECT_EQ(qv, image.GetChannel(*pIter, c));
                }
                ++bufIter;
            }
        }
    }
}

TEST(Image, PfmIO) {
    Point2i res(16, 49);
    std::vector<Float> rgbPixels = GetFloatPixels(res, 3);

    Image image(rgbPixels, PixelFormat::RGB32, res);
    EXPECT_TRUE(image.Write("test.pfm"));
    absl::optional<Image> read = Image::Read("test.pfm");
    EXPECT_TRUE((bool)read);

    EXPECT_EQ(image.resolution, read->resolution);
    EXPECT_EQ(read->format, PixelFormat::RGB32);

    for (int y = 0; y < res[1]; ++y)
        for (int x = 0; x < res[0]; ++x)
            for (int c = 0; c < 3; ++c)
                EXPECT_EQ(image.GetChannel({x, y}, c),
                          read->GetChannel({x, y}, c));

    EXPECT_EQ(0, remove("test.pfm"));
}

TEST(Image, ExrIO) {
    Point2i res(16, 49);
    std::vector<Float> rgbPixels = GetFloatPixels(res, 3);

    for (auto format : { PixelFormat::Y8, PixelFormat::RGB8,
                PixelFormat::Y16, PixelFormat::RGB16,
                PixelFormat::RGB32, PixelFormat::RGB32 }) {
        Image image(format, res);
        image.CopyRectIn({{0, 0}, res}, rgbPixels);
        EXPECT_TRUE(image.Write("test.exr"));
        absl::optional<Image> read = Image::Read("test.exr");
        EXPECT_TRUE((bool)read);

        EXPECT_EQ(image.resolution, read->resolution);
        if (!Is8Bit(format))
            EXPECT_EQ(read->format, format);

        for (int y = 0; y < res[1]; ++y)
            for (int x = 0; x < res[0]; ++x)
                for (int c = 0; c < image.nChannels(); ++c)
                    if (Is8Bit(format))
                        EXPECT_EQ(Float(Half(image.GetChannel({x, y}, c))),
                                  read->GetChannel({x, y}, c));
                    else if (Is16Bit(format))
                        EXPECT_EQ(Float(Half(image.GetChannel({x, y}, c))),
                                  read->GetChannel({x, y}, c));
                    else
                        EXPECT_EQ(image.GetChannel({x, y}, c), read->GetChannel({x, y}, c));

        EXPECT_EQ(0, remove("test.exr"));
    }
}

TEST(Image, PngRgbIO) {
    Point2i res(11, 50);
    std::vector<Float> rgbPixels = GetFloatPixels(res, 3);

    Image image(rgbPixels, PixelFormat::RGB32, res);
    EXPECT_TRUE(image.Write("test.png"));
    absl::optional<Image> read = Image::Read("test.png");
    EXPECT_TRUE((bool)read);

    EXPECT_EQ(image.resolution, read->resolution);
    EXPECT_EQ(read->format, PixelFormat::SRGB8);

    for (int y = 0; y < res[1]; ++y)
        for (int x = 0; x < res[0]; ++x)
            for (int c = 0; c < 3; ++c)
                EXPECT_FLOAT_EQ(sRGBRoundTrip(image.GetChannel({x, y}, c)),
                                read->GetChannel({x, y}, c))
                    << " x " << x << ", y " << y << ", c " << c << ", orig "
                    << rgbPixels[3 * y * res[0] + 3 * x + c];

    EXPECT_EQ(0, remove("test.png"));
}

TEST(Image, ToSRGB_LUTAccuracy) {
    const int n = 1024 * 1024;
    double sumErr = 0, maxErr = 0;
    RNG rng;
    for (int i = 0; i < n; ++i) {
        Float v = (i + rng.UniformFloat()) / n;
        Float lut = LinearToSRGB(v);
        Float precise = LinearToSRGBFull(v);
        double err = std::abs(lut - precise);
        sumErr += err;
        maxErr = std::max(err, maxErr);
    }
    // These bounds were measured empirically.
    EXPECT_LT(sumErr / n, 6e-6);  // average error
    EXPECT_LT(maxErr, 0.0015);
}

TEST(Image, SRGB8ToLinear) {
    for (int v = 0; v < 255; ++v) {
        float err = std::abs(SRGBToLinear(v / 255.f) - SRGB8ToLinear(v));
        EXPECT_LT(err, 1e-6);
    }
}

// Monotonicity between the individual segments actually isn't enforced
// when we do the piecewise linear fit, but it should happen naturally
// since the derivative of the underlying function doesn't change fit.
TEST(Image, ToSRGB_LUTMonotonic) {
    for (int i = 1; i < LinearToSRGBPiecewiseSize; ++i) {
        // For each break in the function, we'd like to find a pair of floats
        // such that the second uses the next segment after the one used by
        // the first. To deal with fp rounding error, move down a bunch of floats
        // from the computed split point and then step up one float at a time.
        Float v = Float(i) / LinearToSRGBPiecewiseSize;
        int slop = 100;
        v = NextFloatDown(v, slop);
        bool spanned = true;
        for (int j = 0; j < 2 * slop; ++j) {
            EXPECT_LE(LinearToSRGB(v), LinearToSRGB(NextFloatUp(v)));
            spanned |= int(v * LinearToSRGBPiecewiseSize) !=
                int(NextFloatUp(v) * LinearToSRGBPiecewiseSize);
            v = NextFloatUp(v);
        }
        // Make sure we actually did cross segments at some point.
        EXPECT_TRUE(spanned);
    }
}

///////////////////////////////////////////////////////////////////////////

TEST(ImageTexelProvider, Y32) {
    Point2i res(32, 8);

    // Must be a power of 2, so that the base image isn't resampled when
    // generating the MIP levels.
    ASSERT_TRUE(IsPowerOf2(res[0]) && IsPowerOf2(res[1]));
    PixelFormat format = PixelFormat::Y32;
    ASSERT_EQ(1, nChannels(format));

    std::vector<Float> pixels = GetFloatPixels(res, nChannels(format));
    Image image(pixels, format, res);
    ImageTexelProvider provider(image, WrapMode::Clamp,
                                SpectrumType::Reflectance);

    for (Point2i p : Bounds2i({0, 0}, res)) {
        Float pv = provider.TexelFloat(0, p);
        EXPECT_EQ(image.GetY(p), pv);
        EXPECT_EQ(pixels[p.x + p.y * res.x], pv);
    }
}

TEST(ImageTexelProvider, RGB32) {
    Point2i res(2, 4); //16, 32);
    // Must be a power of 2, so that the base image isn't resampled when
    // generating the MIP levels.
    ASSERT_TRUE(IsPowerOf2(res[0]) && IsPowerOf2(res[1]));
    PixelFormat format = PixelFormat::RGB32;
    ASSERT_EQ(3, nChannels(format));

    std::vector<Float> pixels = GetFloatPixels(res, nChannels(format));
    Image image(pixels, format, res);
    ImageTexelProvider provider(image, WrapMode::Clamp,
                                SpectrumType::Reflectance);

    for (Point2i p : Bounds2i({0, 0}, res)) {
        Spectrum is = image.GetSpectrum(p);
        Spectrum ps = provider.TexelSpectrum(0, p);
        EXPECT_EQ(is, ps) << "At pixel " << p << ", image gives : " << is <<
            ", image provider gives " << ps;
        std::array<Float, 3> rgb = is.ToRGB();
        for (int c = 0; c < 3; ++c) {
            EXPECT_EQ(pixels[3 * (p.x + p.y * res.x) + c], rgb[c]);
        }
    }
}

#if 0
TEST(TiledTexelProvider, Y32) {
  Point2i res(32, 8);
  TestFloatProvider<TiledTexelProvider>(res, PixelFormat::Y32);
}
#endif

#if 0
TEST(TiledTexelProvider, RGB32) {
    Point2i res(16, 32);

    ASSERT_TRUE(IsPowerOf2(res[0]) && IsPowerOf2(res[1]));
    PixelFormat format = PixelFormat::RGB32;
    ASSERT_EQ(3, nChannels(format));

    std::vector<Float> pixels = GetFloatPixels(res, nChannels(format));
    Image image(pixels, format, res);
    const char *fn = "tiledprovider.pfm";
    image.Write(fn);
    TiledTexelProvider provider(fn, WrapMode::Clamp, SpectrumType::Reflectance,
                                false);

    for (Point2i p : Bounds2i({0, 0}, res)) {
        Spectrum is = image.GetSpectrum(p);
        Spectrum ps = provider.TexelSpectrum(0, p);
        // FIXME: this doesn't work with the flip above :-p
        // CO    EXPECT_EQ(is, ps) << is << "vs " << ps;
        Float rgb[3];
        ps.ToRGB(rgb);
        for (int c = 0; c < 3; ++c) {
            EXPECT_EQ(pixels[3 * (p.x + p.y * res.x) + c], rgb[c]);
        }
    }

    EXPECT_EQ(0, remove(fn));
}
#endif
