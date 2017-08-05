
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
#include "parallel.h"
#include "texture.h"

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
    return Spectrum::FromRGB(&rgb[0], spectrumType);
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
    return Spectrum::FromRGB(&rgb[0], spectrumType);
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
        Float rgb[3];
        s.ToRGB(rgb);
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

    // Note: these aren't freed until the corresponding worker thread exits, but
    // that's
    // probably ok...
    thread_local std::vector<Float> inBuf, sBuf, outBuf;

    ParallelFor2D(Bounds2i({0, 0}, newResolution), 16, [&](Bounds2i outExtent) {
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
        ParallelFor(0, nextResolution[1], 16, [&](int t) {
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

}  // namespace pbrt
