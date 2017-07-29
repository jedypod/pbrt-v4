
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
                        gtl::MutableArraySlice<Float> buf) {
    CHECK_GE(buf.size(), extent.Area() * nChannels());
    CHECK_LT(extent.pMin.x, extent.pMax.x);
    CHECK_LT(extent.pMin.y, extent.pMax.y);

    int nu = extent.pMax[0] - extent.pMin[0];
    auto bufIter = buf.begin();

    switch (format) {
    case PixelFormat::SY8:
        for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
            int offset = PixelOffset({extent.pMin[0], v}, 0);
            for (int u = 0; u < nu; ++u)
                *bufIter++ = SRGB8ToLinear(p8[offset++]);
        }
        break;
    case PixelFormat::SRGB8:
        for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
            int offset = PixelOffset({extent.pMin[0], v}, 0);
            for (int u = 0; u < nu; ++u)
                for (int c = 0; c < 3; ++c)
                    *bufIter++ = SRGB8ToLinear(p8[offset++]);
        }
        break;
    case PixelFormat::Y8:
        for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
            int offset = PixelOffset({extent.pMin[0], v}, 0);
            for (int u = 0; u < nu; ++u)
                *bufIter++ = p8[offset++] * (1.f / 255.f);
        }
        break;
    case PixelFormat::RGB8:
        for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
            int offset = PixelOffset({extent.pMin[0], v}, 0);
            for (int u = 0; u < nu; ++u)
                for (int c = 0; c < 3; ++c)
                    *bufIter++ = p8[offset++] * (1.f / 255.f);
        }
        break;
    case PixelFormat::Y16:
        for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
            int offset = PixelOffset({extent.pMin[0], v}, 0);
            for (int u = 0; u < nu; ++u)
                *bufIter++ = HalfToFloat(p16[offset++]);
        }
        break;
    case PixelFormat::RGB16:
        for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
            int offset = PixelOffset({extent.pMin[0], v}, 0);
            for (int u = 0; u < nu; ++u)
                for (int c = 0; c < 3; ++c)
                    *bufIter++ = HalfToFloat(p16[offset++]);
        }
        break;
    case PixelFormat::Y32:
        for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
            int offset = PixelOffset({extent.pMin[0], v}, 0);
            for (int u = 0; u < nu; ++u)
                *bufIter++ = Float(p32[offset++]);
        }
        break;
    case PixelFormat::RGB32:
        for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
            int offset = PixelOffset({extent.pMin[0], v}, 0);
            for (int u = 0; u < nu; ++u)
                for (int c = 0; c < 3; ++c)
                    *bufIter++ = Float(p32[offset++]);
        }
        break;
    default:
        LOG(FATAL) << "Unhandled PixelFormat";
    }
}

void Image::CopyRectIn(const Bounds2i &extent,
                       gtl::ArraySlice<Float> buf) {
    CHECK_GE(buf.size(), extent.Area() * nChannels());
    CHECK_LT(extent.pMin.x, extent.pMax.x);
    CHECK_LT(extent.pMin.y, extent.pMax.y);

    int nu = extent.pMax[0] - extent.pMin[0];
    auto bufIter = buf.begin();

    switch (format) {
    case PixelFormat::SY8:
        for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
            int offset = PixelOffset({extent.pMin[0], v}, 0);
            for (int u = 0; u < nu; ++u)
                p8[offset++] = LinearToSRGB8(*bufIter++);
        }
        break;
    case PixelFormat::SRGB8:
        for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
            int offset = PixelOffset({extent.pMin[0], v}, 0);
            for (int u = 0; u < nu; ++u)
                for (int c = 0; c < 3; ++c)
                    p8[offset++] = LinearToSRGB8(*bufIter++);
        }
        break;
    case PixelFormat::Y8:
        for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
            int offset = PixelOffset({extent.pMin[0], v}, 0);
            for (int u = 0; u < nu; ++u)
                p8[offset++] = Clamp(255.f * *bufIter++ + 0.5f, 0, 255);
        }
        break;
    case PixelFormat::RGB8:
        for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
            int offset = PixelOffset({extent.pMin[0], v}, 0);
            for (int u = 0; u < nu; ++u)
                for (int c = 0; c < 3; ++c)
                    p8[offset++] = Clamp(255.f * *bufIter++ + 0.5f, 0, 255);
        }
        break;
    case PixelFormat::Y16:
        for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
            int offset = PixelOffset({extent.pMin[0], v}, 0);
            for (int u = 0; u < nu; ++u)
                p16[offset++] = FloatToHalf(*bufIter++);
        }
        break;
    case PixelFormat::RGB16:
        for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
            int offset = PixelOffset({extent.pMin[0], v}, 0);
            for (int u = 0; u < nu; ++u)
                for (int c = 0; c < 3; ++c)
                    p16[offset++] = FloatToHalf(*bufIter++);
        }
        break;
    case PixelFormat::Y32:
        for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
            int offset = PixelOffset({extent.pMin[0], v}, 0);
            for (int u = 0; u < nu; ++u)
                p32[offset++] = *bufIter++;
        }
        break;
    case PixelFormat::RGB32:
        for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
            int offset = PixelOffset({extent.pMin[0], v}, 0);
            for (int u = 0; u < nu; ++u)
                for (int c = 0; c < 3; ++c)
                    p32[offset++] = *bufIter++;
        }
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

static std::unique_ptr<ResampleWeight[]> resampleWeights(int oldRes,
                                                         int newRes) {
    CHECK_GE(newRes, oldRes);
    std::unique_ptr<ResampleWeight[]> wt = std::make_unique<ResampleWeight[]>(newRes);
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

void Image::Resize(Point2i newResolution, WrapMode wrapMode) {
    CHECK_GE(newResolution.x, resolution.x);
    CHECK_GE(newResolution.y, resolution.y);

    // Resample image in $s$ direction
    std::unique_ptr<ResampleWeight[]> sWeights =
        resampleWeights(resolution[0], newResolution[0]);
    const int nc = nChannels();
    CHECK(nc == 1 || nc == 3);
    Image resampledImage(nc == 1 ? PixelFormat::Y32 : PixelFormat::RGB32,
                         newResolution);

    // Apply _sWeights_ to zoom in $s$ direction
    ParallelFor(
        [&](int t) {
            for (int s = 0; s < newResolution[0]; ++s) {
                // Compute texel $(s,t)$ in $s$-zoomed image
                for (int c = 0; c < nc; ++c) {
                    Float value = 0;
                    for (int j = 0; j < 4; ++j) {
                        int origS = sWeights[s].firstTexel + j;
                        value += sWeights[s].weight[j] *
                                 GetChannel({origS, t}, c, wrapMode);
                    }
                    resampledImage.SetChannel({s, t}, c, value);
                }
            }
        },
        resolution[1], 16);

    // Resample image in $t$ direction
    std::unique_ptr<ResampleWeight[]> tWeights =
        resampleWeights(resolution[1], newResolution[1]);
    std::vector<Float *> resampleBufs;
    int nThreads = MaxThreadIndex();
    for (int i = 0; i < nThreads; ++i)
        resampleBufs.push_back(new Float[nc * newResolution[1]]);
    ParallelFor(
        [&](int s) {
            Float *workData = resampleBufs[ThreadIndex];
            memset(workData, 0, sizeof(Float) * nc * newResolution[1]);

            for (int t = 0; t < newResolution[1]; ++t) {
                for (int j = 0; j < 4; ++j) {
                    int tSrc = tWeights[t].firstTexel + j;
                    for (int c = 0; c < nc; ++c)
                        workData[t * nc + c] +=
                            tWeights[t].weight[j] *
                            resampledImage.GetChannel({s, tSrc}, c);
                }
            }
            for (int t = 0; t < newResolution[1]; ++t)
                for (int c = 0; c < nc; ++c) {
                    Float v = Clamp(workData[nc * t + c], 0, Infinity);
                    resampledImage.SetChannel({s, t}, c, v);
                }
        },
        newResolution[0], 32);

    resolution = newResolution;
    if (Is8Bit(format))
        p8.resize(nc * newResolution[0] * newResolution[1]);
    else if (Is16Bit(format))
        p16.resize(nc * newResolution[0] * newResolution[1]);
    else if (Is32Bit(format))
        p32.resize(nc * newResolution[0] * newResolution[1]);
    else
        LOG(FATAL) << "unexpected PixelFormat";

    for (int t = 0; t < resolution[1]; ++t)
        for (int s = 0; s < resolution[0]; ++s)
            for (int c = 0; c < nc; ++c)
                SetChannel(Point2i(s, t), c,
                           resampledImage.GetChannel({s, t}, c));
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

std::vector<Image> Image::GenerateMIPMap(WrapMode wrapMode) const {
    // Make a copy for level 0.
    Image image = *this;

    if (!IsPowerOf2(resolution[0]) || !IsPowerOf2(resolution[1])) {
        // Resample image to power-of-two resolution
        image.Resize({RoundUpPow2(resolution[0]), RoundUpPow2(resolution[1])},
                     wrapMode);
    }

    // Initialize levels of MIPMap from image
    int nLevels =
        1 + Log2Int(std::max(image.resolution[0], image.resolution[1]));
    std::vector<Image> pyramid(nLevels);

    // Initialize most detailed level of MIPMap
    pyramid[0] = std::move(image);

    Point2i levelResolution = pyramid[0].resolution;
    const int nc = nChannels();
    for (int i = 1; i < nLevels; ++i) {
        // Initialize $i$th MIPMap level from $i-1$st level
        levelResolution[0] = std::max(1, levelResolution[0] / 2);
        levelResolution[1] = std::max(1, levelResolution[1] / 2);
        pyramid[i] = Image(pyramid[0].format, levelResolution);

        // Filter four texels from finer level of pyramid
        ParallelFor(
            [&](int t) {
                for (int s = 0; s < levelResolution[0]; ++s) {
                    for (int c = 0; c < nc; ++c) {
                        Float texel =
                            .25f *
                            (pyramid[i - 1].GetChannel(Point2i(2 * s, 2 * t), c,
                                                       wrapMode) +
                             pyramid[i - 1].GetChannel(
                                 Point2i(2 * s + 1, 2 * t), c, wrapMode) +
                             pyramid[i - 1].GetChannel(
                                 Point2i(2 * s, 2 * t + 1), c, wrapMode) +
                             pyramid[i - 1].GetChannel(
                                 Point2i(2 * s + 1, 2 * t + 1), c, wrapMode));
                        pyramid[i].SetChannel(Point2i(s, t), c, texel);
                    }
                }
            },
            levelResolution[1], 16);
    }
    return pyramid;
}

}  // namespace pbrt
