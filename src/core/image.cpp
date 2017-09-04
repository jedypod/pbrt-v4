
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

// core/image.cpp*
#include "image.h"

#include "error.h"
#include "fileutil.h"
#include "fp16.h"
#include "geometry.h"
#include "parallel.h"
#include "spectrum.h"
#include "texture.h"

#include "ext/lodepng.h"

#include <ImfChannelList.h>
#include <ImfFloatAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfInputFile.h>
#include <ImfOutputFile.h>
#include <ImfFrameBuffer.h>

namespace pbrt {

bool RemapPixelCoords(Point2i *p, Point2i resolution, WrapMode wrapMode) {
    switch (wrapMode) {
    case WrapMode::Repeat:
        (*p)[0] = Mod((*p)[0], resolution[0]);
        (*p)[1] = Mod((*p)[1], resolution[1]);
        return true;
    case WrapMode::Clamp:
        (*p)[0] = Clamp((*p)[0], 0, resolution[0] - 1);
        (*p)[1] = Clamp((*p)[1], 0, resolution[1] - 1);
        return true;
    case WrapMode::Black:
        return ((*p)[0] >= 0 && (*p)[0] < resolution[0] && (*p)[1] >= 0 &&
                (*p)[1] < resolution[1]);
    default:
        LOG(ERROR) << "Unhandled WrapMode mode";
    }
}

Image::Image(PixelFormat format, Point2i resolution)
    : format(format), resolution(resolution) {
    if (Is8Bit(format))
        p8.resize(nChannels() * resolution[0] * resolution[1]);
    else if (Is16Bit(format))
        p16.resize(nChannels() * resolution[0] * resolution[1]);
    else if (Is32Bit(format))
        p32.resize(nChannels() * resolution[0] * resolution[1]);
    else
        LOG(FATAL) << "Unhandled format in Image::Image()";
}

Image::Image(std::vector<uint8_t> p8c, PixelFormat format, Point2i resolution)
    : format(format), resolution(resolution), p8(std::move(p8c)) {
    CHECK_EQ(p8.size(), nChannels() * resolution[0] * resolution[1]);
    CHECK(Is8Bit(format));
}

Image::Image(std::vector<uint16_t> p16c, PixelFormat format, Point2i resolution)
    : format(format), resolution(resolution), p16(std::move(p16c)) {
    CHECK_EQ(p16.size(), nChannels() * resolution[0] * resolution[1]);
    CHECK(Is16Bit(format));
}

Image::Image(std::vector<float> p32c, PixelFormat format, Point2i resolution)
    : format(format), resolution(resolution), p32(std::move(p32c)) {
    CHECK_EQ(p32.size(), nChannels() * resolution[0] * resolution[1]);
    CHECK(Is32Bit(format));
}

Image Image::ConvertToFormat(PixelFormat newFormat) const {
    if (newFormat == format) return *this;
    CHECK_EQ(pbrt::nChannels(newFormat), nChannels());

    Image newImage(newFormat, resolution);
    int nc = nChannels();
    for (int y = 0; y < resolution.y; ++y)
        for (int x = 0; x < resolution.x; ++x)
            for (int c = 0; c < nc; ++c)
                newImage.SetChannel({x, y}, c, GetChannel({x, y}, c));
    return newImage;
}

Float Image::GetChannel(Point2i p, int c, WrapMode wrapMode) const {
    if (!RemapPixelCoords(&p, resolution, wrapMode)) return 0;

    // Use convert()? Some rewrite/refactor?
    switch (format) {
    case PixelFormat::SY8:
    case PixelFormat::SRGB8:
        return SRGB8ToLinear(p8[PixelOffset(p, c)]);
    case PixelFormat::Y8:
    case PixelFormat::RGB8:
        return Float(p8[PixelOffset(p, c)]) * (1.f / 255.f);
    case PixelFormat::Y16:
    case PixelFormat::RGB16:
        return HalfToFloat(p16[PixelOffset(p, c)]);
    case PixelFormat::Y32:
    case PixelFormat::RGB32:
        return Float(p32[PixelOffset(p, c)]);
    default:
        LOG(FATAL) << "Unhandled PixelFormat";
    }
}

Float Image::GetY(Point2i p, WrapMode wrapMode) const {
    if (nChannels() == 1) return GetChannel(p, 0, wrapMode);
    CHECK_EQ(3, nChannels());
    std::array<Float, 3> rgb = GetRGB(p, wrapMode);
    // FIXME: um, this isn't luminance as we think of it...
    return (rgb[0] + rgb[1] + rgb[2]) / 3;
}

std::array<Float, 3> Image::GetRGB(Point2i p, WrapMode wrapMode) const {
    CHECK_EQ(3, nChannels());

    if (!RemapPixelCoords(&p, resolution, wrapMode))
        return {(Float)0, (Float)0, (Float)0};

    std::array<Float, 3> rgb;
    switch (format) {
    case PixelFormat::SRGB8:
        for (int c = 0; c < 3; ++c)
            rgb[c] = SRGB8ToLinear(p8[PixelOffset(p, c)]);
        break;
    case PixelFormat::RGB8:
        for (int c = 0; c < 3; ++c)
            rgb[c] = p8[PixelOffset(p, c)] * (1.f / 255.f);
        break;
    case PixelFormat::RGB16:
        for (int c = 0; c < 3; ++c)
            rgb[c] = HalfToFloat(p16[PixelOffset(p, c)]);
        break;
    case PixelFormat::RGB32:
        for (int c = 0; c < 3; ++c) rgb[c] = p32[PixelOffset(p, c)];
        break;
    default:
        LOG(FATAL) << "Unhandled PixelFormat";
    }

    return rgb;
}

Spectrum Image::GetSpectrum(Point2i p, SpectrumType spectrumType,
                            WrapMode wrapMode) const {
    if (nChannels() == 1) return GetChannel(p, 0, wrapMode);
    std::array<Float, 3> rgb = GetRGB(p, wrapMode);
    return Spectrum::FromRGB(rgb, spectrumType);
}

void Image::CopyRectOut(const Bounds2i &extent,
                        gtl::MutableArraySlice<Float> buf, WrapMode wrapMode) {
    CHECK_GE(buf.size(), extent.Area() * nChannels());

    auto bufIter = buf.begin();

    switch (format) {
    case PixelFormat::SY8:
        ForExtent1(extent, wrapMode, [&bufIter, this](int offset) {
            *bufIter++ = SRGB8ToLinear(p8[offset]);
        });
        break;
    case PixelFormat::SRGB8:
        ForExtent3(extent, wrapMode, [&bufIter, this](int offset) {
            *bufIter++ = SRGB8ToLinear(p8[offset]);
        });
        break;
    case PixelFormat::Y8:
        ForExtent1(extent, wrapMode, [&bufIter, this](int offset) {
            *bufIter++ = p8[offset] * (1.f / 255.f);
        });
        break;
    case PixelFormat::RGB8:
        ForExtent3(extent, wrapMode, [&bufIter, this](int offset) {
            *bufIter++ = p8[offset] * (1.f / 255.f);
        });
        break;
    case PixelFormat::Y16:
        ForExtent1(extent, wrapMode, [&bufIter, this](int offset) {
            *bufIter++ = HalfToFloat(p16[offset]);
        });
        break;
    case PixelFormat::RGB16:
        ForExtent3(extent, wrapMode, [&bufIter, this](int offset) {
            *bufIter++ = HalfToFloat(p16[offset]);
        });
        break;
    case PixelFormat::Y32:
        ForExtent1(extent, wrapMode, [&bufIter, this](int offset) {
            *bufIter++ = Float(p32[offset]);
        });
        break;
    case PixelFormat::RGB32:
        ForExtent3(extent, wrapMode, [&bufIter, this](int offset) {
            *bufIter++ = Float(p32[offset]);
        });
        break;
    default:
        LOG(FATAL) << "Unhandled PixelFormat";
    }
}

void Image::CopyRectIn(const Bounds2i &extent, gtl::ArraySlice<Float> buf) {
    CHECK_GE(buf.size(), extent.Area() * nChannels());

    int nu = extent.pMax[0] - extent.pMin[0];
    auto bufIter = buf.begin();

    switch (format) {
    case PixelFormat::SY8:
        ForExtent1(extent, WrapMode::Clamp, [&bufIter, this](int offset) {
            p8[offset] = LinearToSRGB8(*bufIter++);
        });
        break;
    case PixelFormat::SRGB8:
        ForExtent3(extent, WrapMode::Clamp, [&bufIter, this](int offset) {
            p8[offset] = LinearToSRGB8(*bufIter++);
        });
        break;
    case PixelFormat::Y8:
        ForExtent1(extent, WrapMode::Clamp, [&bufIter, this](int offset) {
            p8[offset] = Clamp(255.f * *bufIter++ + 0.5f, 0, 255);
        });
        break;
    case PixelFormat::RGB8:
        ForExtent3(extent, WrapMode::Clamp, [&bufIter, this](int offset) {
            p8[offset] = Clamp(255.f * *bufIter++ + 0.5f, 0, 255);
        });
        break;
    case PixelFormat::Y16:
        ForExtent1(extent, WrapMode::Clamp, [&bufIter, this](int offset) {
            p16[offset] = FloatToHalf(*bufIter++);
        });
        break;
    case PixelFormat::RGB16:
        ForExtent3(extent, WrapMode::Clamp, [&bufIter, this](int offset) {
            p16[offset] = FloatToHalf(*bufIter++);
        });
        break;
    case PixelFormat::Y32:
        ForExtent1(extent, WrapMode::Clamp,
                   [&bufIter, this](int offset) { p32[offset] = *bufIter++; });
        break;
    case PixelFormat::RGB32:
        ForExtent3(extent, WrapMode::Clamp,
                   [&bufIter, this](int offset) { p32[offset] = *bufIter++; });
        break;
    default:
        LOG(FATAL) << "Unhandled PixelFormat";
    }
}

Float Image::BilerpChannel(Point2f p, int c, WrapMode wrapMode) const {
    Float s = p[0] * resolution.x - 0.5f;
    Float t = p[1] * resolution.y - 0.5f;
    int si = std::floor(s), ti = std::floor(t);
    Float ds = s - si, dt = t - ti;
    return ((1 - ds) * (1 - dt) * GetChannel({si, ti}, c, wrapMode) +
            (1 - ds) * dt * GetChannel({si, ti + 1}, c, wrapMode) +
            ds * (1 - dt) * GetChannel({si + 1, ti}, c, wrapMode) +
            ds * dt * GetChannel({si + 1, ti + 1}, c, wrapMode));
}

Float Image::BilerpY(Point2f p, WrapMode wrapMode) const {
    if (nChannels() == 1) return BilerpChannel(p, 0, wrapMode);
    CHECK_EQ(3, nChannels());
    return (BilerpChannel(p, 0, wrapMode) + BilerpChannel(p, 1, wrapMode) +
            BilerpChannel(p, 2, wrapMode)) /
           3;
}

Spectrum Image::BilerpSpectrum(Point2f p, SpectrumType spectrumType,
                               WrapMode wrapMode) const {
    if (nChannels() == 1) return Spectrum(BilerpChannel(p, 0, wrapMode));
    std::array<Float, 3> rgb = {BilerpChannel(p, 0, wrapMode),
                                BilerpChannel(p, 1, wrapMode),
                                BilerpChannel(p, 2, wrapMode)};
    return Spectrum::FromRGB(rgb, spectrumType);
}

void Image::SetChannel(Point2i p, int c, Float value) {
    CHECK(!std::isnan(value));

    switch (format) {
    case PixelFormat::SY8:
    case PixelFormat::SRGB8:
        p8[PixelOffset(p, c)] = LinearToSRGB8(value);
        break;
    case PixelFormat::Y8:
    case PixelFormat::RGB8:
        value = Clamp((value * 255.f) + 0.5f, 0, 255);
        p8[PixelOffset(p, c)] = uint8_t(value);
        break;
    case PixelFormat::Y16:
    case PixelFormat::RGB16:
        p16[PixelOffset(p, c)] = FloatToHalf(value);
        break;
    case PixelFormat::Y32:
    case PixelFormat::RGB32:
        p32[PixelOffset(p, c)] = value;
        break;
    default:
        LOG(FATAL) << "Unhandled PixelFormat in Image::SetChannel()";
    }
}

void Image::SetY(Point2i p, Float value) {
    for (int c = 0; c < nChannels(); ++c) SetChannel(p, c, value);
}

void Image::SetSpectrum(Point2i p, const Spectrum &s) {
    if (nChannels() == 1)
        SetChannel(p, 0, s.Average());
    else {
        CHECK_EQ(3, nChannels());
        std::array<Float, 3> rgb = s.ToRGB();
        for (int c = 0; c < 3; ++c) SetChannel(p, c, rgb[c]);
    }
}

struct ResampleWeight {
    int firstTexel;
    Float weight[4];
};

static std::vector<ResampleWeight> resampleWeights(int oldRes, int newRes) {
    CHECK_GE(newRes, oldRes);
    std::vector<ResampleWeight> wt(newRes);
    Float filterwidth = 2.f;
    for (int i = 0; i < newRes; ++i) {
        // Compute image resampling weights for _i_th texel
        Float center = (i + .5f) * oldRes / newRes;
        wt[i].firstTexel = std::floor((center - filterwidth) + 0.5f);
        for (int j = 0; j < 4; ++j) {
            Float pos = wt[i].firstTexel + j + .5f;
            wt[i].weight[j] = Lanczos((pos - center) / filterwidth);
        }

        // Normalize filter weights for texel resampling
        Float invSumWts = 1 / (wt[i].weight[0] + wt[i].weight[1] +
                               wt[i].weight[2] + wt[i].weight[3]);
        for (int j = 0; j < 4; ++j) wt[i].weight[j] *= invSumWts;
    }
    return wt;
}

Image Image::FloatResize(Point2i newResolution, WrapMode wrapMode) const {
    CHECK_GE(newResolution.x, resolution.x);
    CHECK_GE(newResolution.y, resolution.y);

    std::vector<ResampleWeight> sWeights =
        resampleWeights(resolution[0], newResolution[0]);
    std::vector<ResampleWeight> tWeights =
        resampleWeights(resolution[1], newResolution[1]);
    const int nc = nChannels();
    CHECK(nc == 1 || nc == 3);
    Image resampledImage(nc == 1 ? PixelFormat::Y32 : PixelFormat::RGB32,
                         newResolution);

    // Note: these aren't freed until the corresponding worker thread
    // exits, but that's probably ok...
    thread_local std::vector<Float> inBuf, sBuf, outBuf;

    ParallelFor2D(Bounds2i({0, 0}, newResolution), 64, [&](Bounds2i outExtent) {
        Bounds2i inExtent({sWeights[outExtent[0][0]].firstTexel,
                           tWeights[outExtent[0][1]].firstTexel},
                          {sWeights[outExtent[1][0] - 1].firstTexel + 4,
                           tWeights[outExtent[1][1] - 1].firstTexel + 4});

        if (inBuf.size() < nc * inExtent.Area())
            inBuf.resize(nc * inExtent.Area());

        // Copy the tile of the input image into inBuf. (The
        // main motivation for this copy is to convert it
        // into floats all at once, rather than repeatedly
        // and pixel-by-pixel during the first resampling
        // step.)
        // FIXME CAST
        ((Image *)this)->CopyRectOut(inExtent, &inBuf, wrapMode);

        // Zoom in s. We need to do this across all scanlines
        // in inExtent's t dimension so we have the border
        // pixels available for the zoom in t.
        int nsOut = outExtent[1][0] - outExtent[0][0];
        int ntOut = outExtent[1][1] - outExtent[0][1];
        int nsIn = inExtent[1][0] - inExtent[0][0];
        int ntIn = inExtent[1][1] - inExtent[0][1];

        if (sBuf.size() < nc * ntIn * nsOut) sBuf.resize(nc * ntIn * nsOut);

        int sBufOffset = 0;
        for (int t = 0; t < ntIn; ++t) {
            for (int s = 0; s < nsOut; ++s) {
                int sOut = s + outExtent[0][0];
                DCHECK(sOut >= 0 && sOut < sWeights.size());
                const ResampleWeight &rsw = sWeights[sOut];

                // w.r.t. inBuf
                int sIn = rsw.firstTexel - inExtent[0][0];
                DCHECK_GE(sIn, 0);
                DCHECK_LT(sIn + 3, nsIn);

                int inOffset = nc * (sIn + t * nsIn);
                DCHECK_GE(inOffset, 0);
                DCHECK_LT(inOffset + 3 * nc, inBuf.size());
                for (int c = 0; c < nc; ++c, ++sBufOffset, ++inOffset) {
                    sBuf[sBufOffset] =
                        (rsw.weight[0] * inBuf[inOffset] +
                         rsw.weight[1] * inBuf[inOffset + nc] +
                         rsw.weight[2] * inBuf[inOffset + 2 * nc] +
                         rsw.weight[3] * inBuf[inOffset + 3 * nc]);
                }
            }
        }

        if (outBuf.size() < nc * nsOut * ntOut)
            outBuf.resize(nc * nsOut * ntOut);

        // Zoom in t from sBuf to outBuf
        for (int s = 0; s < nsOut; ++s) {
            for (int t = 0; t < ntOut; ++t) {
                int tOut = t + outExtent[0][1];
                DCHECK(tOut >= 0 && tOut < tWeights.size());
                const ResampleWeight &rsw = tWeights[tOut];

                DCHECK_GE(rsw.firstTexel - inExtent[0][1], 0);
                int sBufOffset =
                    nc * (s + nsOut * (rsw.firstTexel - inExtent[0][1]));
                DCHECK_GE(sBufOffset, 0);
                int step = nc * nsOut;
                DCHECK_LT(sBufOffset + 3 * step, sBuf.size());

                int outOffset = nc * (s + t * nsOut);
                ;
                for (int c = 0; c < nc; ++c, ++outOffset, ++sBufOffset)
                    outBuf[outOffset] =
                        (rsw.weight[0] * sBuf[sBufOffset] +
                         rsw.weight[1] * sBuf[sBufOffset + step] +
                         rsw.weight[2] * sBuf[sBufOffset + 2 * step] +
                         rsw.weight[3] * sBuf[sBufOffset + 3 * step]);
            }
        }
        // Copy out...
        resampledImage.CopyRectIn(outExtent, outBuf);
    });

    return resampledImage;
}

void Image::FlipY() {
    const int nc = nChannels();
    for (int y = 0; y < resolution.y / 2; ++y) {
        for (int x = 0; x < resolution.x; ++x) {
            size_t o1 = PixelOffset({x, y}),
                   o2 = PixelOffset({x, resolution.y - 1 - y});
            for (int c = 0; c < nc; ++c) {
                if (Is8Bit(format))
                    std::swap(p8[o1 + c], p8[o2 + c]);
                else if (Is16Bit(format))
                    std::swap(p16[o1 + c], p16[o2 + c]);
                else if (Is32Bit(format))
                    std::swap(p32[o1 + c], p32[o2 + c]);
                else
                    LOG(FATAL) << "unexpected format";
            }
        }
    }
}

std::vector<Image> Image::GenerateMIPMap(Image image, WrapMode wrapMode) {
    PixelFormat origFormat = image.format;
    // Set things up so we have a power-of-two sized image stored with
    // floats.
    if (!IsPowerOf2(image.resolution[0]) || !IsPowerOf2(image.resolution[1])) {
        // Resample image to power-of-two resolution
        image = image.FloatResize({RoundUpPow2(image.resolution[0]),
                                   RoundUpPow2(image.resolution[1])},
                                  wrapMode);
    } else if (!Is32Bit(image.format))
        image = image.ConvertToFormat(
            image.nChannels() == 1 ? PixelFormat::Y32 : PixelFormat::RGB32);
    CHECK(Is32Bit(image.format));

    // Initialize levels of MIPMap from image
    int nLevels =
        1 + Log2Int(std::max(image.resolution[0], image.resolution[1]));
    std::vector<Image> pyramid(nLevels);

    Point2i levelResolution = image.resolution;
    const int nc = image.nChannels();
    for (int i = 0; i < nLevels - 1; ++i) {
        // Initialize $i+1$st MIPMap level from $i$th level and also convert
        // i'th level to the internal format
        pyramid[i] = Image(origFormat, levelResolution);

        Point2i nextResolution(std::max(1, levelResolution[0] / 2),
                               std::max(1, levelResolution[1] / 2));
        Image nextImage(image.format, nextResolution);

        // Offsets from the base pixel to the four neighbors that we'll
        // downfilter.
        int srcDeltas[4] = {0, nc, nc * levelResolution[0],
                            nc * levelResolution[0] + nc};
        // Clamp offsets once a dimension has a single texel.
        if (levelResolution[0] == 1) {
            srcDeltas[1] = 0;
            srcDeltas[3] -= nc;
        }
        if (levelResolution[1] == 1) {
            srcDeltas[2] = 0;
            srcDeltas[3] -= nc * levelResolution[0];
        }

        // Work in scanlines for best cache coherence (vs 2d tiles).
        ParallelFor(0, nextResolution[1], 8, [&](int64_t start, int64_t end) {
            for (int t = start; t < end; ++t) {
                // Downfilter with a box filter for the next MIP level
                int srcOffset = image.PixelOffset({0, 2 * t}, 0);
                int nextOffset = nextImage.PixelOffset({0, t}, 0);
                for (int s = 0; s < nextResolution[0]; ++s) {
                    for (int c = 0; c < nc; ++c) {
                        nextImage.p32[nextOffset] =
                            .25f * (image.p32[srcOffset] +
                                    image.p32[srcOffset + srcDeltas[1]] +
                                    image.p32[srcOffset + srcDeltas[2]] +
                                    image.p32[srcOffset + srcDeltas[3]]);
                        ++srcOffset;
                        ++nextOffset;
                    }
                    srcOffset += nc;
                }

                // Copy the current level out to the current pyramid level
                int tStart = 2 * t;
                int tEnd = std::min(2 * t + 2, levelResolution[1]);
                int offset = image.PixelOffset({0, tStart}, 0);
                size_t count = (tEnd - tStart) * nc * levelResolution[0];
                pyramid[i].CopyRectIn(
                    Bounds2i({0, tStart}, {levelResolution[0], tEnd}),
                    {image.p32.data() + offset, count});
            }
        });

        image = std::move(nextImage);
        levelResolution = nextResolution;
    }

    // Top level
    CHECK(levelResolution[0] == 1 && levelResolution[1] == 1);
    pyramid[nLevels - 1] = Image(origFormat, levelResolution);
    pyramid[nLevels - 1].CopyRectIn({{0, 0}, {1, 1}},
                                    {image.p32.data(), size_t(nc)});

    return pyramid;
}

// ImageIO Local Declarations
static std::experimental::optional<Image> ReadEXR(const std::string &name,
                                                  ImageMetadata *metadata);
static std::experimental::optional<Image> ReadPNG(const std::string &name,
                                                  bool gamma,
                                                  ImageMetadata *metadata);
static std::experimental::optional<Image> ReadPFM(const std::string &filename,
                                                  ImageMetadata *metadata);

// ImageIO Function Definitions
std::experimental::optional<Image> Image::Read(const std::string &name,
                                               ImageMetadata *metadata,
                                               bool gamma) {
    if (HasExtension(name, ".exr"))
        return ReadEXR(name, metadata);
    else if (HasExtension(name, ".png"))
        return ReadPNG(name, gamma, metadata);
    else if (HasExtension(name, ".pfm"))
        return ReadPFM(name, metadata);
    else {
        Error("%s: no support for reading images with this extension",
              name.c_str());
        return {};
    }
}

bool Image::Write(const std::string &name, const ImageMetadata *metadata) const {
    if (metadata && metadata->pixelBounds)
        CHECK_EQ(metadata->pixelBounds->Area(), resolution.x * resolution.y);

    if (HasExtension(name, ".exr"))
        return WriteEXR(name, metadata);
    else if (HasExtension(name, ".pfm"))
        return WritePFM(name, metadata);
    else if (HasExtension(name, ".png"))
        return WritePNG(name, metadata);
    else {
        Error("%s: no support for writing images with this extension",
              name.c_str());
        return false;
    }
}

///////////////////////////////////////////////////////////////////////////
// OpenEXR

static Imf::FrameBuffer imageToFrameBuffer(const Image &image,
                                           const Imath::Box2i &dataWindow) {
    size_t xStride = TexelBytes(image.format);
    size_t yStride = image.resolution.x * xStride;
    // Would be nice to use PixelOffset(-dw.min.x, -dw.min.y) but
    // it checks to make sure the coordiantes are >= 0 (which
    // usually makes sense...)
    char *originPtr = (((char *)image.RawPointer({0, 0})) -
                       dataWindow.min.x * xStride - dataWindow.min.y * yStride);

    Imf::FrameBuffer fb;
    switch (image.format) {
    case PixelFormat::Y16:
        fb.insert("Y", Imf::Slice(Imf::HALF, originPtr, xStride, yStride));
        break;
    case PixelFormat::RGB16:
        fb.insert("R", Imf::Slice(Imf::HALF, originPtr, xStride, yStride));
        fb.insert("G", Imf::Slice(Imf::HALF, originPtr + sizeof(uint16_t),
                                  xStride, yStride));
        fb.insert("B", Imf::Slice(Imf::HALF, originPtr + 2 * sizeof(uint16_t),
                                  xStride, yStride));
        break;
    case PixelFormat::Y32:
        fb.insert("Y", Imf::Slice(Imf::FLOAT, originPtr, xStride, yStride));
        break;
    case PixelFormat::RGB32:
        fb.insert("R", Imf::Slice(Imf::FLOAT, originPtr, xStride, yStride));
        fb.insert("G", Imf::Slice(Imf::FLOAT, originPtr + sizeof(float),
                                  xStride, yStride));
        fb.insert("B", Imf::Slice(Imf::FLOAT, originPtr + 2 * sizeof(float),
                                  xStride, yStride));
        break;
    default:
        LOG(FATAL) << "Unexpected image format";
    }
    return fb;
}

static std::experimental::optional<Image> ReadEXR(const std::string &name,
                                                  ImageMetadata *metadata) {
    try {
        Imf::InputFile file(name.c_str());
        Imath::Box2i dw = file.header().dataWindow();

        if (metadata) {
            const Imf::FloatAttribute *renderTimeAttrib =
                file.header().findTypedAttribute<Imf::FloatAttribute>("renderTimeSeconds");
            if (renderTimeAttrib)
                metadata->renderTimeSeconds = renderTimeAttrib->value();

            const Imf::M44fAttribute *worldToCameraAttrib =
                file.header().findTypedAttribute<Imf::M44fAttribute>("worldToCamera");
            if (worldToCameraAttrib) {
                Matrix4x4 m;
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        // Can't memcpy since Float may be a double...
                        m.m[i][j] = worldToCameraAttrib->value().getValue()[4*i+j];
                metadata->worldToCamera = m;
            }

            const Imf::M44fAttribute *worldToNDCAttrib =
                file.header().findTypedAttribute<Imf::M44fAttribute>("worldToNDC");
            if (worldToNDCAttrib) {
                Matrix4x4 m;
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        m.m[i][j] = worldToNDCAttrib->value().getValue()[4*i+j];
                metadata->worldToNDC = m;
            }

            // OpenEXR uses inclusive pixel bounds; adjust to non-inclusive
            // (the convention pbrt uses) in the values returned.
            *metadata->pixelBounds = {{dw.min.x, dw.min.y}, {dw.max.x + 1, dw.max.y + 1}};

            Imath::Box2i dispw = file.header().displayWindow();
            metadata->fullResolution->x = dispw.max.x - dispw.min.x + 1;
            metadata->fullResolution->y = dispw.max.y - dispw.min.y + 1;
        }

        int width = dw.max.x - dw.min.x + 1;
        int height = dw.max.y - dw.min.y + 1;

        const Imf::ChannelList &channels = file.header().channels();
        const Imf::Channel *rc = channels.findChannel("R");
        const Imf::Channel *gc = channels.findChannel("G");
        const Imf::Channel *bc = channels.findChannel("B");
        Image image;
        if (rc && gc && bc) {
            if (rc->type == Imf::HALF && gc->type == Imf::HALF &&
                bc->type == Imf::HALF)
                image = Image(PixelFormat::RGB16, {width, height});
            else
                image = Image(PixelFormat::RGB32, {width, height});
        } else if (channels.findChannel("Y")) {
            if (channels.findChannel("Y")->type == Imf::HALF)
                image = Image(PixelFormat::Y16, {width, height});
            else
                image = Image(PixelFormat::Y32, {width, height});
        } else {
            std::string channelNames;
            for (auto iter = channels.begin(); iter != channels.end(); ++iter) {
                channelNames += iter.name();
                channelNames += ' ';
            }
            Error("%s: didn't find RGB or Y stored in image. Channels: %s",
                  name.c_str(), channelNames.c_str());
            return {};
        }
        file.setFrameBuffer(imageToFrameBuffer(image, dw));
        file.readPixels(dw.min.y, dw.max.y);

        LOG(INFO) << StringPrintf("Read EXR image %s (%d x %d)", name.c_str(),
                                  width, height);
        return image;
    } catch (const std::exception &e) {
        Error("Unable to read image file \"%s\": %s", name.c_str(), e.what());
    }

    return {};
}

bool Image::WriteEXR(const std::string &name, const ImageMetadata *metadata) const {
    if (Is8Bit(format)) {
        if (nChannels() == 1)
            return ConvertToFormat(PixelFormat::Y16).WriteEXR(name, metadata);
        else
            return ConvertToFormat(PixelFormat::RGB16).WriteEXR(name, metadata);
    }
    CHECK(Is16Bit(format) || Is32Bit(format));

    try {
        Imath::Box2i displayWindow, dataWindow;
        if (metadata && metadata->fullResolution)
            // Agan, -1 offsets to handle inclusive indexing in OpenEXR...
            displayWindow = {Imath::V2i(0, 0),
                             Imath::V2i(metadata->fullResolution->x - 1,
                                        metadata->fullResolution->y - 1)};
        else
            displayWindow =
                {Imath::V2i(0, 0), Imath::V2i(resolution.x - 1, resolution.y - 1)};

        if (metadata && metadata->pixelBounds)
            dataWindow = {Imath::V2i(metadata->pixelBounds->pMin.x,
                                     metadata->pixelBounds->pMin.y),
                          Imath::V2i(metadata->pixelBounds->pMax.x - 1,
                                     metadata->pixelBounds->pMax.y - 1)};
        else
            dataWindow =
                {Imath::V2i(0, 0), Imath::V2i(resolution.x - 1, resolution.y - 1)};

        Imf::FrameBuffer fb = imageToFrameBuffer(*this, dataWindow);

        Imf::Header header(displayWindow, dataWindow);
        for (auto iter = fb.begin(); iter != fb.end(); ++iter)
            header.channels().insert(iter.name(), iter.slice().type);

        if (metadata) {
            if (metadata->renderTimeSeconds)
                header.insert("renderTimeSeconds", Imf::FloatAttribute(*metadata->renderTimeSeconds));
            // TODO: fix this for Float = double builds.
            if (metadata->worldToCamera)
                header.insert("worldToCamera", Imf::M44fAttribute(metadata->worldToCamera->m));
            if (metadata->worldToNDC)
                header.insert("worldToNDC", Imf::M44fAttribute(metadata->worldToNDC->m));
        }

        Imf::OutputFile file(name.c_str(), header);
        file.setFrameBuffer(fb);
        file.writePixels(resolution.y);
    } catch (const std::exception &exc) {
        Error("Error writing \"%s\": %s", name.c_str(), exc.what());
        return false;
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////
// PNG Function Definitions

static inline uint8_t FloatToSRGB(Float v) {
    return uint8_t(Clamp(255.f * LinearToSRGB(v) + 0.5f, 0.f, 255.f));
}

static std::experimental::optional<Image> ReadPNG(const std::string &name,
                                                  bool gamma,
                                                  ImageMetadata *metadata) {
    auto contents = ReadFileContents(name);
    if (!contents)
        return {};

    unsigned width, height;
    LodePNGState state;
    lodepng_state_init(&state);
    unsigned int error = lodepng_inspect(&width, &height, &state,
                                         (const unsigned char *)contents->data(),
                                         contents->size());
    if (error != 0) {
        Error("%s: %s", name.c_str(), lodepng_error_text(error));
        return {};
    }

    Image image;
    switch (state.info_png.color.colortype) {
    case LCT_GREY:
    case LCT_GREY_ALPHA: {
        std::vector<unsigned char> buf;
        int bpp = state.info_png.color.bitdepth == 16 ? 16 : 8;
        error = lodepng::decode(buf, width, height,
                                (const unsigned char *)contents->data(),
                                contents->size(), LCT_GREY, bpp);
        if (error != 0) {
            Error("%s: %s", name.c_str(), lodepng_error_text(error));
            return {};
        }

        if (state.info_png.color.bitdepth == 16) {
            image = Image(PixelFormat::Y16, Point2i(width, height));
            auto bufIter = buf.begin();
            for (unsigned int y = 0; y < height; ++y)
                for (unsigned int x = 0; x < width; ++x, bufIter += 2) {
                    // Convert from little endian.
                    Float v = (((int)bufIter[0] << 8) + (int)bufIter[1]) / 65535.f;
                    if (gamma) v = SRGBToLinear(v);
                    image.SetChannel(Point2i(x, y), 0, v);
                }
            CHECK(bufIter == buf.end());
        } else {
            image = Image(gamma ? PixelFormat::SY8 : PixelFormat::Y8,
                           Point2i(width, height));
            std::copy(buf.begin(), buf.end(), (uint8_t *)image.RawPointer({0, 0}));
        }
        break;
    }
    default: {
        std::vector<unsigned char> buf;
        int bpp = state.info_png.color.bitdepth == 16 ? 16 : 8;
        error = lodepng::decode(buf, width, height,
                                (const unsigned char *)contents->data(),
                                contents->size(), LCT_RGB, bpp);
        if (error != 0) {
            Error("%s: %s", name.c_str(), lodepng_error_text(error));
            return {};
        }

        if (state.info_png.color.bitdepth == 16) {
            image = Image(PixelFormat::RGB16, Point2i(width, height));
            auto bufIter = buf.begin();
            for (unsigned int y = 0; y < height; ++y)
                for (unsigned int x = 0; x < width; ++x, bufIter += 6) {
                    CHECK(bufIter < buf.end()) ;
                    // Convert from little endian.
                    Float rgb[3] = {
                        (((int)bufIter[0] << 8) + (int)bufIter[1]) / 65535.f,
                        (((int)bufIter[2] << 8) + (int)bufIter[3]) / 65535.f,
                        (((int)bufIter[4] << 8) + (int)bufIter[5]) / 65535.f };
                    if (gamma)
                        for (int c = 0; c < 3; ++c)
                            // TODO: this is slow; could replace with a LUT
                            rgb[c] = SRGBToLinear(rgb[c]);
                    image.SetSpectrum(Point2i(x, y), Spectrum::FromRGB(rgb));
                }
            CHECK(bufIter == buf.end());
        } else {
            image = Image(gamma ? PixelFormat::SRGB8 : PixelFormat::RGB8,
                          Point2i(width, height));
            std::copy(buf.begin(), buf.end(), (uint8_t *)image.RawPointer({0, 0}));
        }
    }
    }

    return image;
}

bool Image::WritePNG(const std::string &name, const ImageMetadata *metadata) const {
    unsigned int error = 0;
    switch (format) {
    case PixelFormat::SRGB8:
        error = lodepng_encode24_file(name.c_str(), &p8[0], resolution.x,
                                      resolution.y);
        break;
    case PixelFormat::SY8:
        error = lodepng_encode_file(name.c_str(), &p8[0], resolution.x,
                                    resolution.y, LCT_GREY, 8 /* bitdepth */);
        break;
    case PixelFormat::RGB8:
    case PixelFormat::RGB16:
    case PixelFormat::RGB32: {
        std::unique_ptr<uint8_t[]> rgb8 =
            std::make_unique<uint8_t[]>(3 * resolution.x * resolution.y);
        for (int y = 0; y < resolution.y; ++y)
            for (int x = 0; x < resolution.x; ++x)
                for (int c = 0; c < 3; ++c)
                    rgb8[3 * (y * resolution.x + x) + c] =
                        FloatToSRGB(GetChannel({x, y}, c));

        error = lodepng_encode24_file(name.c_str(), rgb8.get(), resolution.x,
                                      resolution.y);
        break;
    }
    case PixelFormat::Y8:
    case PixelFormat::Y16:
    case PixelFormat::Y32: {
        std::unique_ptr<uint8_t[]> y8 =
            std::make_unique<uint8_t[]>(resolution.x * resolution.y);
        for (int y = 0; y < resolution.y; ++y)
            for (int x = 0; x < resolution.x; ++x)
                y8[y * resolution.x + x] = FloatToSRGB(GetChannel({x, y}, 0));

        error = lodepng_encode_file(name.c_str(), y8.get(), resolution.x,
                                    resolution.y, LCT_GREY, 8 /* bitdepth */);
        break;
    }
    }

    if (error != 0) {
        Error("Error writing PNG \"%s\": %s", name.c_str(),
              lodepng_error_text(error));
        return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////
// PFM Function Definitions

/*
 * PFM reader/writer code courtesy Jiawen "Kevin" Chen
 * (http://people.csail.mit.edu/jiawen/)
 */

static constexpr bool hostLittleEndian =
#if defined(__BYTE_ORDER__)
  #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    true
  #elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    false
  #else
    #error "__BYTE_ORDER__ defined but has unexpected value"
  #endif
#else
  #if defined(__LITTLE_ENDIAN__) || defined(__i386__) || defined(__x86_64__) || \
      defined(WIN32)
    true
  #elif defined(__BIG_ENDIAN__)
    false
  #elif defined(__sparc) || defined(__sparc__)
    false
  #else
    #error "Can't detect machine endian-ness at compile-time."
  #endif
#endif
    ;

#define BUFFER_SIZE 80

static inline int isWhitespace(char c) {
    return c == ' ' || c == '\n' || c == '\t';
}

// Reads a "word" from the fp and puts it into buffer and adds a null
// terminator.  i.e. it keeps reading until whitespace is reached.  Returns
// the number of characters read *not* including the whitespace, and
// returns -1 on an error.
static int readWord(FILE *fp, char *buffer, int bufferLength) {
    int n;
    int c;

    if (bufferLength < 1) return -1;

    n = 0;
    c = fgetc(fp);
    while (c != EOF && !isWhitespace(c) && n < bufferLength) {
        buffer[n] = c;
        ++n;
        c = fgetc(fp);
    }

    if (n < bufferLength) {
        buffer[n] = '\0';
        return n;
    }

    return -1;
}

static std::experimental::optional<Image> ReadPFM(const std::string &filename,
                                                  ImageMetadata *metadata) {
    std::vector<float> rgb32;
    char buffer[BUFFER_SIZE];
    unsigned int nFloats;
    int nChannels, width, height;
    float scale;
    bool fileLittleEndian;

    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp) goto fail;

    // read either "Pf" or "PF"
    if (readWord(fp, buffer, BUFFER_SIZE) == -1) goto fail;

    if (strcmp(buffer, "Pf") == 0)
        nChannels = 1;
    else if (strcmp(buffer, "PF") == 0)
        nChannels = 3;
    else
        goto fail;

    // read the rest of the header
    // read width
    if (readWord(fp, buffer, BUFFER_SIZE) == -1) goto fail;
    width = atoi(buffer);

    // read height
    if (readWord(fp, buffer, BUFFER_SIZE) == -1) goto fail;
    height = atoi(buffer);

    // read scale
    if (readWord(fp, buffer, BUFFER_SIZE) == -1) goto fail;
    sscanf(buffer, "%f", &scale);

    // read the data
    nFloats = nChannels * width * height;
    rgb32.resize(nFloats);
    for (int y = height - 1; y >= 0; --y)
      if (fread(&rgb32[nChannels * y * width], sizeof(float),
                nChannels * width, fp) != nChannels * width) goto fail;

    // apply endian conversian and scale if appropriate
    fileLittleEndian = (scale < 0.f);
    if (hostLittleEndian ^ fileLittleEndian) {
        uint8_t bytes[4];
        for (unsigned int i = 0; i < nFloats; ++i) {
            memcpy(bytes, &rgb32[i], 4);
            std::swap(bytes[0], bytes[3]);
            std::swap(bytes[1], bytes[2]);
            memcpy(&rgb32[i], bytes, 4);
        }
    }
    if (std::abs(scale) != 1.f)
        for (unsigned int i = 0; i < nFloats; ++i) rgb32[i] *= std::abs(scale);

    // create RGBs...
    fclose(fp);
    LOG(INFO) << StringPrintf("Read PFM image %s (%d x %d)",
                              filename.c_str(), width, height);
    return Image(std::move(rgb32),
                 nChannels == 1 ? PixelFormat::Y32 : PixelFormat::RGB32,
                 Point2i(width, height));

fail:
    Error("Error reading PFM file \"%s\"", filename.c_str());
    if (fp) fclose(fp);
    return {};
}

bool Image::WritePFM(const std::string &filename, const ImageMetadata *metadata) const {
    FILE *fp;
    float scale;

    fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        Error("Unable to open output PFM file \"%s\"", filename.c_str());
        return false;
    }

    std::unique_ptr<float[]> scanline = std::make_unique<float[]>(3 * resolution.x);

    // only write 3 channel PFMs here...
    if (fprintf(fp, "PF\n") < 0) goto fail;

    // write the width and height, which must be positive
    if (fprintf(fp, "%d %d\n", resolution.x, resolution.y) < 0) goto fail;

    // write the scale, which encodes endianness
    scale = hostLittleEndian ? -1.f : 1.f;
    if (fprintf(fp, "%f\n", scale) < 0) goto fail;

    // write the data from bottom left to upper right as specified by
    // http://netpbm.sourceforge.net/doc/pfm.html
    // The raster is a sequence of pixels, packed one after another, with no
    // delimiters of any kind. They are grouped by row, with the pixels in each
    // row ordered left to right and the rows ordered bottom to top.
    for (int y = resolution.y - 1; y >= 0; y--) {
        for (int x = 0; x < resolution.x; ++x) {
            Spectrum s = GetSpectrum({x, y});
            std::array<Float, 3> rgb = s.ToRGB();
            for (int c = 0; c < 3; ++c)
                // Assign element-wise in case Float is typedefed as 'double'.
                scanline[3 * x + c] = rgb[c];
        }
        if (fwrite(&scanline[0], sizeof(float), 3 * resolution.x, fp) <
            (size_t)(3 * resolution.x))
            goto fail;
    }

    fclose(fp);
    return true;

fail:
    Error("Error writing PFM file \"%s\"", filename.c_str());
    fclose(fp);
    return false;
}

}  // namespace pbrt
