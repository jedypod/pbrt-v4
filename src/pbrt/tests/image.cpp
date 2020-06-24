
#include <gtest/gtest.h>

#include <pbrt/pbrt.h>

#include <pbrt/mipmap.h>
#include <pbrt/util/array2d.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/file.h>
#include <pbrt/util/float.h>
#include <pbrt/util/image.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>

#include <algorithm>
#include <array>
#include <cmath>

using namespace pbrt;

// TODO:
// for png i/o: test mono and rgb; make sure mono is smaller
// pixel bounds stuff... (including i/o paths...)
// basic lookups, bilerps, etc
//   also clamp, repeat, etc...
// resize?
// round trip: init, write, read, check
// FlipY()

TEST(Image, Basics) {
    const ColorEncoding *encoding = ColorEncoding::Linear;
    Image y8(PixelFormat::U256, {4, 8}, { "Y" }, encoding);
    EXPECT_EQ(y8.NChannels(), 1);
    EXPECT_EQ(y8.BytesUsed(), y8.Resolution()[0] * y8.Resolution()[1]);

    Image y16(PixelFormat::Half, {4, 8}, { "Y" });
    EXPECT_EQ(y16.NChannels(), 1);
    EXPECT_EQ(y16.BytesUsed(), 2 * y16.Resolution()[0] * y16.Resolution()[1]);

    Image y32(PixelFormat::Float, {4, 8}, { "Y" });
    EXPECT_EQ(y32.NChannels(), 1);
    EXPECT_EQ(y32.BytesUsed(), 4 * y32.Resolution()[0] * y32.Resolution()[1]);

    Image rgb8(PixelFormat::U256, {4, 8}, { "R", "G", "B" }, encoding);
    EXPECT_EQ(rgb8.NChannels(), 3);
    EXPECT_EQ(rgb8.BytesUsed(), 3 * rgb8.Resolution()[0] * rgb8.Resolution()[1]);

    Image rgb16(PixelFormat::Half, {4, 16}, { "R", "G", "B" });
    EXPECT_EQ(rgb16.NChannels(), 3);
    EXPECT_EQ(rgb16.BytesUsed(),
              2 * 3 * rgb16.Resolution()[0] * rgb16.Resolution()[1]);

    Image rgb32(PixelFormat::Float, {4, 32}, { "R", "G", "B" });
    EXPECT_EQ(rgb32.NChannels(), 3);
    EXPECT_EQ(rgb32.BytesUsed(),
              4 * 3 * rgb32.Resolution()[0] * rgb32.Resolution()[1]);
}

static Float sRGBRoundTrip(Float v, Float dither = 0) {
    if (v < 0) return 0;
    else if (v > 1) return 1;
    uint8_t encoded = LinearToSRGB8(v, dither);
    return SRGB8ToLinear(encoded);
}

static pstd::vector<uint8_t> GetInt8Pixels(Point2i res, int nc) {
    pstd::vector<uint8_t> r;
    for (int y = 0; y < res[1]; ++y)
        for (int x = 0; x < res[0]; ++x)
            for (int c = 0; c < nc; ++c) r.push_back((x * y + c) % 255);
    return r;
}

static pstd::vector<float> GetFloatPixels(Point2i res, int nc) {
    pstd::vector<float> p;
    for (int y = 0; y < res[1]; ++y)
        for (int x = 0; x < res[0]; ++x)
            for (int c = 0; c < nc; ++c)
                if (c == 0)
                    p.push_back(std::sin(x/4.) * std::cos(y/8.));
                else
                    p.push_back(-.25 +
                                2. * (c + 3 * x + 3 * y * res[0]) /
                                (res[0] * res[1]));
    return p;
}

static Float modelQuantization(Float value, PixelFormat format) {
    switch (format) {
    case PixelFormat::U256:
        return Clamp((value * 255.f) + 0.5f, 0, 255) * (1.f / 255.f);
    case PixelFormat::Half:
        return Float(Half(value));
    case PixelFormat::Float:
        return value;
    default:
        LOG_FATAL("Unhandled pixel format");
    }
}

TEST(Image, GetSetY) {
    Point2i res(9, 3);
    pstd::vector<float> yPixels = GetFloatPixels(res, 1);

    for (auto format : {PixelFormat::U256, PixelFormat::Half, PixelFormat::Float}) {
        Image image(format, res, { "Y" }, ColorEncoding::Linear);
        for (int y = 0; y < res[1]; ++y)
            for (int x = 0; x < res[0]; ++x) {
                image.SetChannel({x, y}, 0, yPixels[y * res[0] + x]);
            }

        pstd::optional<ImageChannelDesc> yDesc = image.GetChannelDesc({ "Y" });
        EXPECT_TRUE(bool(yDesc));
        EXPECT_FALSE(bool(image.GetChannelDesc({ "Y0L0" })));

        for (int y = 0; y < res[1]; ++y)
            for (int x = 0; x < res[0]; ++x) {
                Float v = image.GetChannel({x, y}, 0);
                ImageChannelValues cv = image.GetChannels({x, y});
                EXPECT_EQ(1, cv.size());
                EXPECT_EQ(v, cv[0]);

                cv = image.GetChannels({x, y}, *yDesc);
                EXPECT_EQ(1, cv.size());
                EXPECT_EQ(v, cv[0]);

                if (format == PixelFormat::U256)
                    EXPECT_LT(std::abs(v - Clamp(yPixels[y * res[0] + x], 0, 1)),
                              0.501f / 255.f) << v << " vs " << Clamp(yPixels[y * res[0] + x], 0, 1);
                else
                    EXPECT_EQ(v, modelQuantization(yPixels[y * res[0] + x], format));
            }
    }
}

TEST(Image, GetSetRGB) {
    Point2i res(7, 32);
    pstd::vector<float> rgbPixels = GetFloatPixels(res, 3);

    for (auto format : {PixelFormat::U256, PixelFormat::Half, PixelFormat::Float}) {
        Image image(format, res, { "R", "G", "B" }, ColorEncoding::Linear);
        for (int y = 0; y < res[1]; ++y)
            for (int x = 0; x < res[0]; ++x)
                for (int c = 0; c < 3; ++c)
                    image.SetChannel({x, y}, c,
                                     rgbPixels[3 * y * res[0] + 3 * x + c]);

        pstd::optional<ImageChannelDesc> rgbDesc = image.GetChannelDesc({ "R", "G", "B" });
        EXPECT_TRUE(bool(rgbDesc));
        EXPECT_FALSE(bool(image.GetChannelDesc({ "R", "Gxxx", "B" })));

        for (int y = 0; y < res[1]; ++y)
            for (int x = 0; x < res[0]; ++x) {
                ImageChannelValues rgb = image.GetChannels({x, y});
                EXPECT_EQ(3, rgb.size());

                ImageChannelValues rgb2 = image.GetChannels({x, y}, *rgbDesc);
                EXPECT_EQ(rgb[0], rgb2[0]);
                EXPECT_EQ(rgb[1], rgb2[1]);
                EXPECT_EQ(rgb[2], rgb2[2]);

                for (int c = 0; c < 3; ++c) {
                    EXPECT_EQ(rgb[c], image.GetChannel({x, y}, c));
                    int offset = 3 * y * res[0] + 3 * x + c;
                    if (format == PixelFormat::U256)
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

TEST(Image, GetSetBGR) {
    Point2i res(7, 32);
    pstd::vector<float> rgbPixels = GetFloatPixels(res, 3);

    for (auto format : {PixelFormat::U256, PixelFormat::Half, PixelFormat::Float}) {
        Image image(format, res, { "R", "G", "B" }, ColorEncoding::Linear);
        for (int y = 0; y < res[1]; ++y)
            for (int x = 0; x < res[0]; ++x)
                for (int c = 0; c < 3; ++c)
                    image.SetChannel({x, y}, c,
                                     rgbPixels[3 * y * res[0] + 3 * x + c]);

        pstd::optional<ImageChannelDesc> bgrDesc = image.GetChannelDesc({ "B", "G", "R" });
        EXPECT_TRUE(bool(bgrDesc));

        for (int y = 0; y < res[1]; ++y)
            for (int x = 0; x < res[0]; ++x) {
                ImageChannelValues rgb = image.GetChannels({x, y});
                EXPECT_EQ(3, rgb.size());

                ImageChannelValues bgr = image.GetChannels({x, y}, *bgrDesc);
                EXPECT_EQ(rgb[0], bgr[2]);
                EXPECT_EQ(rgb[1], bgr[1]);
                EXPECT_EQ(rgb[2], bgr[0]);

                for (int c = 0; c < 3; ++c) {
                    EXPECT_EQ(rgb[c], image.GetChannel({x, y}, c));
                    int offset = 3 * y * res[0] + 3 * x + c;
                    if (format == PixelFormat::U256)
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

    for (auto format : {PixelFormat::U256, PixelFormat::Half, PixelFormat::Float}) {
        //for (int nc : { 1, 3 }) {
        for (int nc = 1; nc < 4; ++nc) {
            pstd::vector<float> orig = GetFloatPixels(res, nc);

            std::vector<std::string> channelNames = { "A" };
            for (int i = 1; i < nc; ++i)
                channelNames.push_back(std::string(1, 'A' + 1));

            Image image(format, res, channelNames, ColorEncoding::Linear);

            auto origIter = orig.begin();
            for (int y = 0; y < res[1]; ++y)
                for (int x = 0; x < res[0]; ++x)
                    for (int c = 0; c < nc; ++c, ++origIter)
                        image.SetChannel({x, y}, c, *origIter);

            Bounds2i extent(Point2i(2, 3), Point2i(5, 10));
            std::vector<float> buf(extent.Area() * nc);

            image.CopyRectOut(extent, pstd::MakeSpan(buf));

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
}

TEST(Image, CopyRectIn) {
    Point2i res(17, 32);
    RNG rng;

    for (auto format : {PixelFormat::U256, PixelFormat::Half, PixelFormat::Float}) {
        for (int nc = 1; nc < 4; ++nc) {
            pstd::vector<float> orig = GetFloatPixels(res, nc);

            std::vector<std::string> channelNames = { "A" };
            for (int i = 1; i < nc; ++i)
                channelNames.push_back(std::string(1, 'A' + 1));

            Image image(format, res, channelNames, ColorEncoding::Linear);
            auto origIter = orig.begin();
            for (int y = 0; y < res[1]; ++y)
                for (int x = 0; x < res[0]; ++x)
                    for (int c = 0; c < nc; ++c, ++origIter)
                        image.SetChannel({x, y}, c, *origIter);

            Bounds2i extent(Point2i(10, 23), Point2i(17, 28));
            std::vector<float> buf(extent.Area() * nc);
            std::generate(buf.begin(), buf.end(),
                          [&rng]() { return rng.Uniform<Float>(); });

            image.CopyRectIn(extent, buf);

            // Iterate through the points in the extent and the buffer
            // together.
            auto bufIter = buf.begin();
            for (auto pIter = begin(extent); pIter != end(extent); ++pIter) {
                for (int c = 0; c < nc; ++c) {
                    ASSERT_FALSE(bufIter == buf.end());
                    if (format == PixelFormat::U256) {
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
}

TEST(Image, PfmIO) {
    Point2i res(16, 49);
    pstd::vector<float> rgbPixels = GetFloatPixels(res, 3);

    Image image(rgbPixels, res, { "R", "G", "B" });
    EXPECT_TRUE(image.Write("test.pfm"));
    pstd::optional<ImageAndMetadata> read = Image::Read("test.pfm");
    EXPECT_TRUE((bool)read);
    EXPECT_EQ(*RGBColorSpace::sRGB, *read->metadata.GetColorSpace());

    EXPECT_EQ(image.Resolution(), read->image.Resolution());
    EXPECT_EQ(PixelFormat::Float, read->image.Format());
    EXPECT_EQ(3, read->image.NChannels());

    for (int y = 0; y < res[1]; ++y)
        for (int x = 0; x < res[0]; ++x)
            for (int c = 0; c < 3; ++c)
                EXPECT_EQ(image.GetChannel({x, y}, c),
                          read->image.GetChannel({x, y}, c));

    EXPECT_EQ(0, remove("test.pfm"));
}

TEST(Image, ExrIO) {
    Point2i res(16, 49);
    pstd::vector<float> rgbPixels = GetFloatPixels(res, 3);

    for (auto format : { PixelFormat::U256, PixelFormat::Half, PixelFormat::Float }) {
        Image image(format, res, { "R", "G", "B" }, ColorEncoding::Linear);
        image.CopyRectIn({{0, 0}, res}, rgbPixels);

        // Check CopyRectIn()
        int offset = 0;
        for (int y = 0; y < res.y; ++y)
            for (int x = 0; x < res.x; ++x)
                for (int c = 0; c < 3; ++c, ++offset) {
                    if (format == PixelFormat::U256)
                        ;
                    else if (format == PixelFormat::Half)
                        ASSERT_EQ(Float(Half(rgbPixels[offset])), image.GetChannel({x, y}, c));
                    else if (format == PixelFormat::Float)
                        ASSERT_EQ(rgbPixels[offset], image.GetChannel({x, y}, c));
                }

        ImageMetadata metadata;
        metadata.colorSpace = RGBColorSpace::ACES2065_1;
        EXPECT_TRUE(image.Write("test.exr", metadata));

        pstd::optional<ImageAndMetadata> read = Image::Read("test.exr");
        ASSERT_TRUE((bool)read);

        EXPECT_EQ(image.Resolution(), read->image.Resolution());
        EXPECT_EQ(*RGBColorSpace::ACES2065_1, *read->metadata.GetColorSpace());
        if (!Is8Bit(format))
            EXPECT_EQ(read->image.Format(), format);

        pstd::optional<ImageChannelDesc> rgbDesc = read->image.GetChannelDesc({ "R", "G", "B" });
        ASSERT_TRUE(bool(rgbDesc));

        for (int y = 0; y < res[1]; ++y)
            for (int x = 0; x < res[0]; ++x) {
                ImageChannelValues v = read->image.GetChannels({x, y}, *rgbDesc);
                for (int c = 0; c < 3; ++c)
                    if (Is8Bit(format))
                        EXPECT_EQ(Float(Half(image.GetChannel({x, y}, c))), v[c]);
                    else if (Is16Bit(format))
                        EXPECT_EQ(Float(Half(image.GetChannel({x, y}, c))), v[c]);
                    else
                        EXPECT_EQ(image.GetChannel({x, y}, c), v[c]) <<
                            " @ (" << x << ", " << y << ", ch " << c << ")";
            }

        EXPECT_EQ(0, remove("test.exr"));
    }
}

TEST(Image, ExrNoMetadata) {
    Point2i res(16, 32);
    pstd::vector<float> rgbPixels = GetFloatPixels(res, 3);
    Image image(rgbPixels, res, { "R", "G", "B" });

    std::string filename = "nometadata.exr";
    ImageMetadata writeMetadata;
    writeMetadata.colorSpace = RGBColorSpace::ACES2065_1;
    EXPECT_TRUE(image.Write(filename, writeMetadata));

    pstd::optional<ImageAndMetadata> read = Image::Read(filename);
    EXPECT_TRUE((bool)read);
    ImageMetadata &metadata = read->metadata;
    EXPECT_EQ(*RGBColorSpace::ACES2065_1, *metadata.GetColorSpace());

    // All of the metadata should be unset
    EXPECT_FALSE((bool)metadata.renderTimeSeconds);
    EXPECT_FALSE((bool)metadata.cameraFromWorld);
    EXPECT_FALSE((bool)metadata.NDCFromWorld);
    EXPECT_TRUE((bool)metadata.pixelBounds);
    EXPECT_EQ(*metadata.pixelBounds, Bounds2i({0, 0}, res));
    EXPECT_TRUE((bool)metadata.fullResolution);
    EXPECT_EQ(*metadata.fullResolution, res);
    EXPECT_EQ(0, metadata.stringVectors.size());

    EXPECT_EQ(0, remove(filename.c_str()));
}

TEST(Image, ExrMetadata) {
    Point2i res(16, 32);
    pstd::vector<float> rgbPixels = GetFloatPixels(res, 3);
    Image image(rgbPixels, res, { "R", "G", "B" });

    std::string filename = "metadata.exr";
    ImageMetadata outMetadata;
    outMetadata.colorSpace = RGBColorSpace::Rec2020;
    outMetadata.renderTimeSeconds = 1234;
    SquareMatrix<4> w2c(3, 1, 4, 1,
                5, 9, 2, Pi,
                2, 7, 1, 8,
                2, 8, 1, std::exp(1.f));
    SquareMatrix<4> w2n(1.5, 2.5, 3.5, 4.75,
                5.333, 6.2135, -351.2, -552.,
                63.2, 47.2, Pi, std::cos(1.f),
                0, -14, 6, 1e-10f);
    // Must be the same area as image resolution.
    Bounds2i pb(Point2i(2, 10), Point2i(18, 42));
    Point2i fullRes(1000, 200);
    std::map<std::string, std::vector<std::string>> stringVectors;
    stringVectors["yolo"] = { "foo", "bar" };

    outMetadata.cameraFromWorld = w2c;
    outMetadata.NDCFromWorld = w2n;
    outMetadata.pixelBounds = pb;
    outMetadata.fullResolution = fullRes;
    outMetadata.stringVectors = stringVectors;
    EXPECT_TRUE(image.Write(filename, outMetadata));

    pstd::optional<ImageAndMetadata> read = Image::Read(filename);
    EXPECT_TRUE((bool)read);
    ImageMetadata &inMetadata = read->metadata;

    EXPECT_EQ(*RGBColorSpace::Rec2020, *inMetadata.GetColorSpace());

    EXPECT_TRUE((bool)inMetadata.renderTimeSeconds);
    EXPECT_EQ(1234, *inMetadata.renderTimeSeconds);

    EXPECT_TRUE((bool)inMetadata.cameraFromWorld);
    EXPECT_EQ(*inMetadata.cameraFromWorld, w2c);

    EXPECT_TRUE((bool)inMetadata.NDCFromWorld);
    EXPECT_EQ(*inMetadata.NDCFromWorld, w2n);

    EXPECT_TRUE((bool)inMetadata.pixelBounds);
    EXPECT_EQ(*inMetadata.pixelBounds, pb);

    EXPECT_TRUE((bool)inMetadata.fullResolution);
    EXPECT_EQ(*inMetadata.fullResolution, fullRes);

    EXPECT_EQ(1, inMetadata.stringVectors.size());
    auto iter = stringVectors.find("yolo");
    EXPECT_TRUE(iter != stringVectors.end());
    EXPECT_EQ("foo", iter->second[0]);
    EXPECT_EQ("bar", iter->second[1]);

    EXPECT_EQ(0, remove(filename.c_str()));
}

TEST(Image, PngRgbIO) {
    Point2i res(11, 50);
    pstd::vector<float> rgbPixels = GetFloatPixels(res, 3);

    Image image(rgbPixels, res, { "R", "G", "B" });
    EXPECT_TRUE(image.Write("test.png"));
    pstd::optional<ImageAndMetadata> read = Image::Read("test.png");
    EXPECT_TRUE((bool)read);

    EXPECT_EQ(image.Resolution(), read->image.Resolution());
    EXPECT_EQ(read->image.Format(), PixelFormat::U256);
    ASSERT_TRUE(read->image.Encoding() != nullptr);
    //EXPECT_EQ(*read->image.Encoding(), *ColorEncoding::sRGB);
    ASSERT_TRUE((bool)read->metadata.colorSpace);
    ASSERT_TRUE(*read->metadata.colorSpace != nullptr);
    EXPECT_EQ(*RGBColorSpace::sRGB, *read->metadata.GetColorSpace());

    for (int y = 0; y < res[1]; ++y)
        for (int x = 0; x < res[0]; ++x)
            for (int c = 0; c < 3; ++c) {
                EXPECT_LE(sRGBRoundTrip(image.GetChannel({x, y}, c), -.5f),
                          read->image.GetChannel({x, y}, c))
                    << " x " << x << ", y " << y << ", c " << c << ", orig "
                    << rgbPixels[3 * y * res[0] + 3 * x + c];
                EXPECT_LE(read->image.GetChannel({x, y}, c),
                          sRGBRoundTrip(image.GetChannel({x, y}, c), 0.5f))
                    << " x " << x << ", y " << y << ", c " << c << ", orig "
                    << rgbPixels[3 * y * res[0] + 3 * x + c];
            }

    EXPECT_EQ(0, remove("test.png"));
}

TEST(Image, ToSRGB_LUTAccuracy) {
    const int n = 1024 * 1024;
    double sumErr = 0, maxErr = 0;
    RNG rng;
    for (int i = 0; i < n; ++i) {
        Float v = (i + rng.Uniform<Float>()) / n;
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
            EXPECT_GT(NextFloatUp(v), v);
            EXPECT_LE(LinearToSRGB(v), LinearToSRGB(NextFloatUp(v))) <<
                StringPrintf("%f @ %d = %f, next %f @ %d = %f",
                             v, int(v * LinearToSRGBPiecewiseSize), LinearToSRGB(v),
                             NextFloatUp(v), int(NextFloatUp(v) *LinearToSRGBPiecewiseSize),
                             LinearToSRGB(NextFloatUp(v)));
            spanned |= int(v * LinearToSRGBPiecewiseSize) !=
                int(NextFloatUp(v) * LinearToSRGBPiecewiseSize);
            v = NextFloatUp(v);
        }
        // Make sure we actually did cross segments at some point.
        EXPECT_TRUE(spanned);
    }
}

TEST(Image, SampleSimple) {
    pstd::vector<float> texels = {Float(0), Float(1), Float(0), Float(0)};
    Image zeroOne(texels, {2,2}, { "Y" });
    Distribution2D distrib(zeroOne.ComputeSamplingDistribution(zeroOne.AllChannelsDesc(), 2,
                                                               Bounds2f(Point2f(0,0), Point2f(1,1)), Norm::L1));
    RNG rng;
    for (int i = 0; i < 1000; ++i) {
        Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());
        Float pdf;
        Point2f p = distrib.SampleContinuous(u, &pdf);
        // Due to bilerp on lookup, the non-zero range goes out a bit.
        EXPECT_GE(p.x, 0.25);
        EXPECT_LE(p.y, 0.75);
    }
}

TEST(Image, SampleLinear) {
    int w = 500, h = 500;
    pstd::vector<float> v;
    for (int y = 0; y < h; ++y) {
        Float fy = (y + .5) / h;
        for (int x = 0; x < w; ++x) {
            Float fx = (x + .5) / w;
            // This integrates to 1 over [0,1]^2
            Float f = fx + fy;
            v.push_back(f);
        }
    }

    Image image(v, {w, h}, { "Y" });
    Distribution2D distrib(image.ComputeSamplingDistribution(image.AllChannelsDesc(), 2,
                                                             Bounds2f(Point2f(0,0), Point2f(1,1)), Norm::L1));
    RNG rng;
    for (int i = 0; i < 1000; ++i) {
        Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());
        Float pdf;
        Point2f p = distrib.SampleContinuous(u, &pdf);
        Float f = p.x + p.y;
        // Allow some error since Distribution2D uses a piecewise constant
        // sampling distribution.
        EXPECT_LE(std::abs(f - pdf), 1e-3) << u << ", f: " << f << ", pdf: " << pdf;
    }
}

TEST(Image, SampleSinCos) {
    int w = 500, h = 500;
    auto f = [](Point2f p) {
        return std::abs(std::sin(3. * p.x) * Sqr(std::cos(4. * p.y)));
    };
    // Integral of f over [0,1]^2
    Float integral = 1./24. * Sqr(std::sin(1.5)) * (8 + std::sin(8.));

    pstd::vector<float> v;
    for (int y = 0; y < h; ++y) {
        Float fy = (y + .5) / h;
        for (int x = 0; x < w; ++x) {
            Float fx = (x + .5) / w;
            v.push_back(f({fx, fy}));
        }
    }

    Image image(v, {w, h}, { "Y" });
    Distribution2D distrib(image.ComputeSamplingDistribution(image.AllChannelsDesc(), 2,
                                                             Bounds2f(Point2f(0,0), Point2f(1,1)), Norm::L1));
    RNG rng;
    for (int i = 0; i < 1000; ++i) {
        Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());
        Float pdf;
        Point2f p = distrib.SampleContinuous(u, &pdf);
        Float fp = f(p);
        // Allow some error since Distribution2D uses a piecewise constant
        // sampling distribution.
        EXPECT_LE(std::abs(fp - pdf * integral), 3e-3) << u << ", fp: " << fp << ", pdf: " << pdf;
    }
}

TEST(Image, L1Sample) {
    Point2i res(8, 15);
    pstd::vector<float> pixels = GetFloatPixels(res, 1);
    for (float &p : pixels) p = std::abs(p);
    // Put a spike in the middle
    pixels[27] = 10000;

    Image image(pixels, res, { "Y" });
    Distribution2D imageDistrib(image.ComputeSamplingDistribution(image.AllChannelsDesc(), 1,
                                                                  Bounds2f(Point2f(0,0), Point2f(1,1)), Norm::L1));

    auto bilerp = [&](Float x, Float y) {
                      return image.Bilerp({x, y}).MaxValue();
                  };
    int nSamples = 65536;
    auto values = Sample2DFunction(bilerp, res[0], res[1], nSamples,
                                   Bounds2f(Point2f(0,0), Point2f(1,1)), Norm::L1);
    Distribution2D sampledDistrib(values, res[0], res[1],
                                  Bounds2f(Point2f(0,0), Point2f(1,1)));

    Distribution2D::TestCompareDistributions(imageDistrib, sampledDistrib, 1e-3f);
}

TEST(Image, L2Sample) {
    Point2i res(8, 15);
    pstd::vector<float> pixels = GetFloatPixels(res, 1);
    for (float &p : pixels) p = std::abs(p);
    // Put a spike in the middle
    pixels[27] = 10000;

    Image image(pixels, res, { "Y" });
    Distribution2D imageDistrib(image.ComputeSamplingDistribution(image.AllChannelsDesc(), 1,
                                                                  Bounds2f(Point2f(0,0), Point2f(1,1)), Norm::L2));

    auto bilerp = [&](Float x, Float y) {
                      return image.Bilerp({x, y}).MaxValue();
                  };
    int nSamples = 65536;
    auto values = Sample2DFunction(bilerp, res[0], res[1], nSamples,
                                   Bounds2f(Point2f(0,0), Point2f(1,1)), Norm::L2);
    Distribution2D sampledDistrib(values, res[0], res[1]);

    Distribution2D::TestCompareDistributions(imageDistrib, sampledDistrib, 2e-4f);
}

TEST(Image, LInfinitySample) {
    Point2i res(8, 15);
    pstd::vector<float> pixels = GetFloatPixels(res, 1);
    for (float &p : pixels) p = std::abs(p);

    Image image(pixels, res, { "Y" });
    int resScale = 1;
    Distribution2D imageDistrib(
        image.ComputeSamplingDistribution(image.AllChannelsDesc(), resScale,
                                          Bounds2f(Point2f(0,0), Point2f(1,1)), Norm::LInfinity));

    auto bilerp = [&](Float x, Float y) {
                      return image.Bilerp({x, y}).MaxValue();
                  };
    int nSamples = 4*65536;
    auto values = Sample2DFunction(bilerp, resScale * res[0], resScale * res[1],
                                   nSamples, Bounds2f(Point2f(0,0), Point2f(1,1)),
                                   Norm::LInfinity);
    Distribution2D sampledDistrib(values, resScale * res[0], resScale * res[1]);

    Distribution2D::TestCompareDistributions(imageDistrib, sampledDistrib);
}

TEST(Image, Wrap2D) {
    pstd::vector<float> texels = {Float(0), Float(1), Float(0),
                                 Float(0), Float(0), Float(0),
                                 Float(0), Float(0), Float(0)};
    Image zeroOne(texels, {3,3}, { "Y" });

    EXPECT_EQ(1, zeroOne.GetChannel({1, -1}, 0, {WrapMode::Clamp, WrapMode::Clamp}));
    EXPECT_EQ(1, zeroOne.GetChannel({1, -1}, 0, {WrapMode::Black, WrapMode::Clamp}));
    EXPECT_EQ(0, zeroOne.GetChannel({1, -1}, 0, {WrapMode::Black, WrapMode::Repeat}));
    EXPECT_EQ(0, zeroOne.GetChannel({1, -1}, 0, {WrapMode::Clamp, WrapMode::Black}));

    EXPECT_EQ(0, zeroOne.GetChannel({1, 3}, 0, {WrapMode::Clamp, WrapMode::Clamp}));
    EXPECT_EQ(0, zeroOne.GetChannel({1, 3}, 0, {WrapMode::Repeat, WrapMode::Clamp}));
    EXPECT_EQ(1, zeroOne.GetChannel({1, 3}, 0, {WrapMode::Black, WrapMode::Repeat}));
    EXPECT_EQ(0, zeroOne.GetChannel({1, 3}, 0, {WrapMode::Clamp, WrapMode::Black}));

    EXPECT_EQ(0.5, zeroOne.BilerpChannel(Point2f(0.5, 0.), 0, WrapMode::Repeat));
    EXPECT_EQ(0.5, zeroOne.BilerpChannel(Point2f(0.5, 0.), 0, WrapMode::Black));
    EXPECT_EQ(1, zeroOne.BilerpChannel(Point2f(0.5, 0.), 0, WrapMode::Clamp));
}

TEST(Image, Select) {
    Point2i res(4, 9);
    pstd::vector<float> pix = GetFloatPixels(res, 4);
    Image image(pix, res, { "A", "B", "G", "R" });

    auto aDesc = image.GetChannelDesc({ "A" });
    EXPECT_TRUE(bool(aDesc));
    Image aImage = image.SelectChannels(*aDesc);

    EXPECT_EQ(aImage.Resolution(), image.Resolution());
    EXPECT_EQ(1, aImage.NChannels());
    EXPECT_EQ(1, aImage.ChannelNames().size());
    EXPECT_EQ("A", aImage.ChannelNames()[0]);
    for (int y = 0; y < res.y; ++y)
        for (int x = 0; x < res.x; ++x)
            EXPECT_EQ(aImage.GetChannel({x, y}, 0), image.GetChannel({x, y}, 0));

    auto rgDesc = image.GetChannelDesc({ "R", "G" });
    EXPECT_TRUE(bool(rgDesc));
    Image rgImage = image.SelectChannels(*rgDesc);
    EXPECT_EQ(rgImage.Resolution(), image.Resolution());
    EXPECT_EQ(2, rgImage.NChannels());
    EXPECT_EQ(2, rgImage.ChannelNames().size());
    EXPECT_EQ("R", rgImage.ChannelNames()[0]);
    EXPECT_EQ("G", rgImage.ChannelNames()[1]);
    for (int y = 0; y < res.y; ++y)
        for (int x = 0; x < res.x; ++x) {
            EXPECT_EQ(rgImage.GetChannel({x, y}, 0), image.GetChannel({x, y}, 3));
            EXPECT_EQ(rgImage.GetChannel({x, y}, 1), image.GetChannel({x, y}, 2));
        }
}

///////////////////////////////////////////////////////////////////////////

static std::string inTestDir(const std::string &path) { return path; }

static void TestRoundTrip(const char *fn) {
    Point2i res(16, 29);
    Image image(PixelFormat::Float, res, { "R", "G", "B" });
    for (int y = 0; y < res[1]; ++y)
        for (int x = 0; x < res[0]; ++x)
            image.SetChannels({x, y}, {Float(x) / Float(res[0] - 1),
                                       Float(y) / Float(res[1] - 1),
                                       Float(-1.5)});

    std::string filename = inTestDir(fn);
    ASSERT_TRUE(image.Write(filename));

    auto readImage = Image::Read(filename);
    ASSERT_TRUE((bool)readImage);
    ASSERT_EQ(readImage->image.Resolution(), res);

    pstd::optional<ImageChannelDesc> rgbDesc =
        readImage->image.GetChannelDesc({ "R", "G", "B" });
    ASSERT_TRUE((bool)rgbDesc);

    for (int y = 0; y < res[1]; ++y)
        for (int x = 0; x < res[0]; ++x) {
            ImageChannelValues rgb = readImage->image.GetChannels({x, y},
                                                                  *rgbDesc);

            for (int c = 0; c < 3; ++c) {
                float wrote = image.GetChannel({x, y}, c);
                float delta = wrote - rgb[c];
                if (HasExtension(filename, "pfm")) {
                    // Everything should come out exact.
                    EXPECT_EQ(0, delta) << filename << ":(" << x << ", " << y
                                        << ") c = " << c << " wrote " << wrote
                                        << ", read " << rgb[c]
                                        << ", delta = " << delta;
                } else if (HasExtension(filename, "exr")) {
                    if (c == 2)
                        // -1.5 is exactly representable as a float.
                        EXPECT_EQ(0, delta) << "(" << x << ", " << y
                                            << ") c = " << c << " wrote "
                                            << wrote << ", read " << rgb[c]
                                            << ", delta = " << delta;
                    else
                        EXPECT_LT(std::abs(delta), .001)
                            << filename << ":(" << x << ", " << y << ") c = " << c
                            << " wrote " << wrote << ", read " << rgb[c]
                            << ", delta = " << delta;
                } else {
                    // 8 bit format...
                    if (c == 2)
                        // -1.5 should be clamped to zero.
                        EXPECT_EQ(0, rgb[c]) << "(" << x << ", " << y
                                             << ") c = " << c << " wrote "
                                             << wrote << ", read " << rgb[c]
                                             << " (expected 0 back)";
                    else
                        // Allow a fair amount of slop, since there's an sRGB
                        // conversion before quantization to 8-bits...
                        EXPECT_LT(std::abs(delta), .02)
                            << filename << ":(" << x << ", " << y << ") c = " << c
                            << " wrote " << wrote << ", read " << rgb[c]
                            << ", delta = " << delta;
                }
            }
        }

    // Clean up
    EXPECT_EQ(0, remove(filename.c_str()));
}

TEST(ImageIO, RoundTripEXR) { TestRoundTrip("out.exr"); }

TEST(ImageIO, RoundTripPFM) { TestRoundTrip("out.pfm"); }

TEST(ImageIO, RoundTripPNG) { TestRoundTrip("out.png"); }

TEST(SummedArea, Constant) {
    Array2D<Float> v(3, 3);

    for (int y = 0; y < v.ySize(); ++y)
        for (int x = 0; x < v.xSize(); ++x)
            v(x, y) = 1;

    SummedAreaTable sat(v);

    EXPECT_EQ(1, sat.Sum(Bounds2i({0, 0}, {1, 1})));
    EXPECT_EQ(9, sat.Sum(Bounds2i({0, 0}, {3, 3})));
    EXPECT_EQ(6, sat.Sum(Bounds2i({0, 1}, {3, 3})));
    EXPECT_EQ(6, sat.Sum(Bounds2i({1, 0}, {3, 3})));
    EXPECT_EQ(4, sat.Sum(Bounds2i({1, 1}, {3, 3})));
}

TEST(SummedArea, Randoms) {
    std::array<int, 2> dims[] = { { 1, 6 }, { 6, 1 }, { 12, 19 }, { 16, 16 }, { 100, 300 }, { 49, 2 } };
    RNG rng;
    for (const auto d : dims) {
        Array2D<Float> v(d[0], d[1]);

        for (int y = 0; y < v.ySize(); ++y)
            for (int x = 0; x < v.xSize(); ++x)
                v(x, y) = rng.Uniform<int>(32);

        SummedAreaTable sat(v);

        for (int i = 0; i < 100; ++i) {
            Bounds2i b({rng.Uniform<int>(v.xSize()), rng.Uniform<int>(v.ySize()) },
                       {rng.Uniform<int>(v.xSize()), rng.Uniform<int>(v.ySize()) });
            double ref = 0;
            for (Point2i p : b)
                ref += v[p];
            double s = sat.Sum(b);
            EXPECT_EQ(ref, s);
        }
    }
}

TEST(SummedArea, ConstantVariance) {
    Array2D<Float> v(3, 3);

    for (int y = 0; y < v.ySize(); ++y)
        for (int x = 0; x < v.xSize(); ++x)
            v(x, y) = 1;

    SummedAreaTable sat(v);

    EXPECT_EQ(0, sat.Variance(Bounds2i({0, 0}, {1, 1})));
    EXPECT_EQ(0, sat.Variance(Bounds2i({0, 0}, {3, 3})));
    EXPECT_EQ(0, sat.Variance(Bounds2i({0, 1}, {3, 3})));
    EXPECT_EQ(0, sat.Variance(Bounds2i({1, 0}, {3, 3})));
    EXPECT_EQ(0, sat.Variance(Bounds2i({1, 1}, {3, 3})));
}

TEST(SummedArea, RandomsVariance) {
    std::array<int, 2> dims[] = { { 1, 6 }, { 6, 1 }, { 12, 19 }, { 16, 16 }, { 100, 300 }, { 49, 2 } };
    RNG rng;
    for (const auto d : dims) {
        Array2D<Float> v(d[0], d[1]);

        for (int y = 0; y < v.ySize(); ++y)
            for (int x = 0; x < v.xSize(); ++x)
                v(x, y) = rng.Uniform<int>(16);

        SummedAreaTable sat(v);

        for (int i = 0; i < 100; ++i) {
            Bounds2i b({rng.Uniform<int>(v.xSize()), rng.Uniform<int>(v.ySize()) },
                       {rng.Uniform<int>(v.xSize()), rng.Uniform<int>(v.ySize()) });
            if (b.Area() == 0)
                continue;

            double s = sat.Sum(b);
            double avg = s / b.Area();
            double refVarSum = 0;
            for (Point2i p : b)
                refVarSum += Sqr(v[p] - avg);
            double refVar = refVarSum / b.Area();

            double satVar = sat.Variance(b);
            if (refVar == 0) {
                EXPECT_EQ(0, satVar);
            } else {
                double error = std::abs(refVar - sat.Variance(b)) / refVar;
                EXPECT_LT(error, 1e-4) << " reference variance: " << refVar <<
                    ", SummedAreaTable::Variance: " << sat.Variance(b);
            }
        }
    }
}

