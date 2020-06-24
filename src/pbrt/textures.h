
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

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_TEXTURES_CONSTANT_H
#define PBRT_TEXTURES_CONSTANT_H

// textures.h*
#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/interaction.h>
#include <pbrt/mipmap.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/math.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

#include <initializer_list>
#include <map>
#include <mutex>
#include <string>

namespace pbrt {

PBRT_HOST_DEVICE
Float Noise(Float x, Float y = .5f, Float z = .5f);
PBRT_HOST_DEVICE
Float Noise(const Point3f &p);
PBRT_HOST_DEVICE
Float FBm(const Point3f &p, const Vector3f &dpdx, const Vector3f &dpdy,
          Float omega, int octaves);
PBRT_HOST_DEVICE
Float Turbulence(const Point3f &p, const Vector3f &dpdx, const Vector3f &dpdy,
                 Float omega, int octaves);

class TextureEvalContext {
public:
    TextureEvalContext() = default;
    PBRT_HOST_DEVICE
    TextureEvalContext(const Point3f &p, const Vector3f &dpdx, const Vector3f &dpdy,
                       const Point2f &uv, Float dudx, Float dvdx, Float dudy, Float dvdy)
        : uv(uv), dudx(dudx), dvdx(dvdx), dudy(dudy), dvdy(dvdy),
          p(p), dpdx(dpdx), dpdy(dpdy) { }
    PBRT_HOST_DEVICE
    TextureEvalContext(const SurfaceInteraction &si)
        : uv(si.uv), dudx(si.dudx), dvdx(si.dvdx), dudy(si.dudy), dvdy(si.dvdy),
          p(si.p()), dpdx(si.dpdx), dpdy(si.dpdy), faceIndex(si.faceIndex) { }

    Point2f uv;
    Float dudx = 0, dvdx = 0, dudy = 0, dvdy = 0;

    Point3f p;
    Vector3f dpdx, dpdy;

    int faceIndex = 0;
};

class alignas(8) UVMapping2D {
  public:
    // UVMapping2D Public Methods
    UVMapping2D(Float su = 1, Float sv = 1, Float du = 0, Float dv = 0)
        : su(su), sv(sv), du(du), dv(dv) {}

    PBRT_HOST_DEVICE_INLINE
    Point2f Map(const TextureEvalContext &ctx, Vector2f *dstdx,
                Vector2f *dstdy) const {
        // Compute texture differentials for 2D identity mapping
        *dstdx = Vector2f(su * ctx.dudx, sv * ctx.dvdx);
        *dstdy = Vector2f(su * ctx.dudy, sv * ctx.dvdy);
        return {su * ctx.uv[0] + du, sv * ctx.uv[1] + dv};
    }

    std::string ToString() const;

  private:
    Float su, sv, du, dv;
};

class alignas(8) SphericalMapping2D {
  public:
    // SphericalMapping2D Public Methods
    SphericalMapping2D(const Transform &textureFromWorld)
        : textureFromWorld(textureFromWorld) {}

    PBRT_HOST_DEVICE_INLINE
    Point2f Map(const TextureEvalContext &ctx, Vector2f *dstdx,
                Vector2f *dstdy) const {
        Point2f st = sphere(ctx.p);
        // Compute texture coordinate differentials for sphere $(u,v)$ mapping
        const Float delta = .1f;
        Point2f stDeltaX = sphere(ctx.p + delta * ctx.dpdx);
        *dstdx = (stDeltaX - st) / delta;
        Point2f stDeltaY = sphere(ctx.p + delta * ctx.dpdy);
        *dstdy = (stDeltaY - st) / delta;

        // Handle sphere mapping discontinuity for coordinate differentials
        if ((*dstdx)[1] > .5)
            (*dstdx)[1] = 1 - (*dstdx)[1];
        else if ((*dstdx)[1] < -.5f)
            (*dstdx)[1] = -((*dstdx)[1] + 1);
        if ((*dstdy)[1] > .5)
            (*dstdy)[1] = 1 - (*dstdy)[1];
        else if ((*dstdy)[1] < -.5f)
            (*dstdy)[1] = -((*dstdy)[1] + 1);
        return st;
    }

    std::string ToString() const;

  private:
    PBRT_HOST_DEVICE_INLINE
    Point2f sphere(const Point3f &p) const {
        Vector3f vec = Normalize(textureFromWorld(p) - Point3f(0, 0, 0));
        Float theta = SphericalTheta(vec), phi = SphericalPhi(vec);
        return {theta * InvPi, phi * Inv2Pi};
    }

    Transform textureFromWorld;

};

class alignas(8) CylindricalMapping2D {
  public:
    // CylindricalMapping2D Public Methods
    CylindricalMapping2D(const Transform &textureFromWorld)
        : textureFromWorld(textureFromWorld) {}

    PBRT_HOST_DEVICE_INLINE
    Point2f Map(const TextureEvalContext &ctx, Vector2f *dstdx,
                Vector2f *dstdy) const {
        Point2f st = cylinder(ctx.p);
        // Compute texture coordinate differentials for cylinder $(u,v)$ mapping
        const Float delta = .01f;
        Point2f stDeltaX = cylinder(ctx.p + delta * ctx.dpdx);
        *dstdx = (stDeltaX - st) / delta;
        if ((*dstdx)[1] > .5)
            (*dstdx)[1] = 1.f - (*dstdx)[1];
        else if ((*dstdx)[1] < -.5f)
            (*dstdx)[1] = -((*dstdx)[1] + 1);
        Point2f stDeltaY = cylinder(ctx.p + delta * ctx.dpdy);
        *dstdy = (stDeltaY - st) / delta;
        if ((*dstdy)[1] > .5)
            (*dstdy)[1] = 1.f - (*dstdy)[1];
        else if ((*dstdy)[1] < -.5f)
            (*dstdy)[1] = -((*dstdy)[1] + 1);
        return st;
    }

    std::string ToString() const;

  private:
    // CylindricalMapping2D Private Methods
    PBRT_HOST_DEVICE_INLINE
    Point2f cylinder(const Point3f &p) const {
        Vector3f vec = Normalize(textureFromWorld(p) - Point3f(0, 0, 0));
        return Point2f((Pi + std::atan2(vec.y, vec.x)) * Inv2Pi, vec.z);
    }
    Transform textureFromWorld;
};

class alignas(8) PlanarMapping2D {
  public:
    // PlanarMapping2D Public Methods
    PlanarMapping2D(const Vector3f &vs, const Vector3f &vt, Float ds = 0,
                    Float dt = 0)
        : vs(vs), vt(vt), ds(ds), dt(dt) {}

    PBRT_HOST_DEVICE_INLINE
    Point2f Map(const TextureEvalContext &ctx, Vector2f *dstdx,
                Vector2f *dstdy) const {
        Vector3f vec(ctx.p);
        *dstdx = Vector2f(Dot(ctx.dpdx, vs), Dot(ctx.dpdx, vt));
        *dstdy = Vector2f(Dot(ctx.dpdy, vs), Dot(ctx.dpdy, vt));
        return {ds + Dot(vec, vs), dt + Dot(vec, vt)};
    }

    std::string ToString() const;

  private:
    Vector3f vs, vt;
    Float ds, dt;
};

class TextureMapping2DHandle :
        public TaggedPointer<UVMapping2D, SphericalMapping2D, CylindricalMapping2D,
                             PlanarMapping2D> {
public:
    using TaggedPointer::TaggedPointer;
    PBRT_HOST_DEVICE
    TextureMapping2DHandle(TaggedPointer<UVMapping2D, SphericalMapping2D, CylindricalMapping2D,
                                         PlanarMapping2D> tp)
        : TaggedPointer(tp) { }

    static TextureMapping2DHandle Create(const ParameterDictionary &dict,
                                         const Transform &worldFromTexture,
                                         const FileLoc *loc, Allocator alloc);

    PBRT_HOST_DEVICE
    Point2f Map(const TextureEvalContext &ctx, Vector2f *dstdx,
                Vector2f *dstdy) const {
        if (Tag() == TypeIndex<UVMapping2D>())
            return Cast<UVMapping2D>()->Map(ctx, dstdx, dstdy);
        else if (Tag() == TypeIndex<SphericalMapping2D>())
            return Cast<SphericalMapping2D>()->Map(ctx, dstdx, dstdy);
        else if (Tag() == TypeIndex<CylindricalMapping2D>())
            return Cast<CylindricalMapping2D>()->Map(ctx, dstdx, dstdy);
        else {
            CHECK_EQ(Tag(), TypeIndex<PlanarMapping2D>());
            return Cast<PlanarMapping2D>()->Map(ctx, dstdx, dstdy);
        }
    }
};

class alignas(8) TransformMapping3D {
  public:
    // TransformMapping3D Public Methods
    TransformMapping3D(const Transform &textureFromWorld)
        : textureFromWorld(textureFromWorld) {}

    PBRT_HOST_DEVICE_INLINE
    Point3f Map(const TextureEvalContext &ctx, Vector3f *dpdx,
                Vector3f *dpdy) const {
        *dpdx = textureFromWorld(ctx.dpdx);
        *dpdy = textureFromWorld(ctx.dpdy);
        return textureFromWorld(ctx.p);
    }

    std::string ToString() const;

  private:
    Transform textureFromWorld;
};

class TextureMapping3DHandle : public TaggedPointer<TransformMapping3D> {
public:
    using TaggedPointer::TaggedPointer;
    PBRT_HOST_DEVICE
    TextureMapping3DHandle(TaggedPointer<TransformMapping3D> tp)
        : TaggedPointer(tp) { }

    static TextureMapping3DHandle Create(const ParameterDictionary &dict,
                                         const Transform &worldFromTexture,
                                         const FileLoc *loc, Allocator alloc);

    PBRT_HOST_DEVICE_INLINE
    Point3f Map(const TextureEvalContext &ctx, Vector3f *dpdx,
                Vector3f *dpdy) const {
        CHECK_EQ(Tag(), TypeIndex<TransformMapping3D>());
        return Cast<TransformMapping3D>()->Map(ctx, dpdx, dpdy);
    }
};

// ConstantTexture Declarations
class alignas(8) FloatConstantTexture {
  public:
    FloatConstantTexture(Float value) : value(value) {}

    PBRT_HOST_DEVICE_INLINE
    Float Evaluate(const TextureEvalContext &) const { return value; }

    static FloatConstantTexture *Create(const Transform &worldFromTexture,
                                        const TextureParameterDictionary &dict,
                                        const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    Float value;
};

class alignas(8) SpectrumConstantTexture {
  public:
    SpectrumConstantTexture(SpectrumHandle value) : value(value) {
        DCHECK(value);
    }

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum Evaluate(const TextureEvalContext &ctx,
                             const SampledWavelengths &lambda) const {
        return value.Sample(lambda);
    }

    static SpectrumConstantTexture *Create(const Transform &worldFromTexture,
                                           const TextureParameterDictionary &dict,
                                           const FileLoc *loc, Allocator alloc);
    std::string ToString() const;

  private:
    SpectrumHandle value;
};

// BilerpTexture Declarations
class alignas(8) FloatBilerpTexture {
  public:
    FloatBilerpTexture(TextureMapping2DHandle mapping, Float v00,
                       Float v01, Float v10, Float v11)
        : mapping(mapping), v00(v00), v01(v01), v10(v10), v11(v11) {}

    PBRT_HOST_DEVICE_INLINE
    Float Evaluate(const TextureEvalContext &ctx) const {
        Vector2f dstdx, dstdy;
        Point2f st = mapping.Map(ctx, &dstdx, &dstdy);
        return Bilerp({st[0], st[1]}, {v00, v10, v01, v11});
    }

    static FloatBilerpTexture *Create(const Transform &worldFromTexture,
                                      const TextureParameterDictionary &dict,
                                      const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // BilerpTexture Private Data
    TextureMapping2DHandle mapping;
    Float v00, v01, v10, v11;
};

class alignas(8) SpectrumBilerpTexture {
  public:
    SpectrumBilerpTexture(TextureMapping2DHandle mapping,
                          SpectrumHandle v00, SpectrumHandle v01,
                          SpectrumHandle v10, SpectrumHandle v11)
        : mapping(mapping),
          v00(v00), v01(v01), v10(v10), v11(v11) {}

    PBRT_HOST_DEVICE
    SampledSpectrum Evaluate(const TextureEvalContext &ctx,
                             const SampledWavelengths &lambda) const {
        Vector2f dstdx, dstdy;
        Point2f st = mapping.Map(ctx, &dstdx, &dstdy);
        return Bilerp({st[0], st[1]}, {v00.Sample(lambda), v10.Sample(lambda),
                                       v01.Sample(lambda), v11.Sample(lambda)});
    }

    static SpectrumBilerpTexture *Create(const Transform &worldFromTexture,
                                         const TextureParameterDictionary &dict,
                                         const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // BilerpTexture Private Data
    TextureMapping2DHandle mapping;
    SpectrumHandle v00, v01, v10, v11;
};

// AAMethod Declaration
enum class AAMethod { None, ClosedForm };

PBRT_HOST_DEVICE
pstd::array<Float, 2> Checkerboard(AAMethod aaMethod,
                                   const TextureEvalContext &ctx,
                                   const TextureMapping2DHandle map2D,
                                   const TextureMapping3DHandle map3D);

// CheckerboardTexture Declarations
class alignas(8) FloatCheckerboardTexture {
  public:
    FloatCheckerboardTexture(TextureMapping2DHandle map2D,
                             TextureMapping3DHandle map3D,
                             FloatTextureHandle tex1,
                             FloatTextureHandle tex2,
                             AAMethod aaMethod)
        : map2D(map2D), map3D(map3D), tex { tex1, tex2 }, aaMethod(aaMethod) {}

    PBRT_HOST_DEVICE_INLINE
    Float Evaluate(const TextureEvalContext &ctx) const {
        pstd::array<Float, 2> wt = Checkerboard(aaMethod, ctx, map2D, map3D);
        if (wt[0] == 0)
            return wt[1] * tex[1].Evaluate(ctx);
        else if (wt[1] == 0)
            return wt[0] * tex[0].Evaluate(ctx);
        else return wt[0] * tex[0].Evaluate(ctx) + wt[1] * tex[1].Evaluate(ctx);
    }

    static FloatCheckerboardTexture *Create(const Transform &worldFromTexture,
                                            const TextureParameterDictionary &dict,
                                            const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    TextureMapping2DHandle map2D;
    TextureMapping3DHandle map3D;
    FloatTextureHandle tex[2];
    AAMethod aaMethod;
};

class alignas(8) SpectrumCheckerboardTexture {
  public:
    SpectrumCheckerboardTexture(TextureMapping2DHandle map2D,
                                TextureMapping3DHandle map3D,
                                SpectrumTextureHandle tex1,
                                SpectrumTextureHandle tex2,
                                AAMethod aaMethod)
        : map2D(map2D), map3D(map3D), tex { tex1, tex2 }, aaMethod(aaMethod) {}

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum Evaluate(const TextureEvalContext &ctx,
                             const SampledWavelengths &lambda) const {
        pstd::array<Float, 2> wt = Checkerboard(aaMethod, ctx, map2D, map3D);
        if (wt[0] == 0)
            return wt[1] * tex[1].Evaluate(ctx, lambda);
        else if (wt[1] == 0)
            return wt[0] * tex[0].Evaluate(ctx, lambda);
        else return wt[0] * tex[0].Evaluate(ctx, lambda) +
                 wt[1] * tex[1].Evaluate(ctx, lambda);
    }

    static SpectrumCheckerboardTexture *Create(const Transform &worldFromTexture,
                                               const TextureParameterDictionary &dict,
                                               const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    TextureMapping2DHandle map2D;
    TextureMapping3DHandle map3D;
    SpectrumTextureHandle tex[2];
    AAMethod aaMethod;
};

// DotsTexture Declarations
class DotsBase {
 protected:
    PBRT_HOST_DEVICE_INLINE
    static bool Inside(const Point2f &st) {
        int sCell = std::floor(st[0] + .5f), tCell = std::floor(st[1] + .5f);

        // Return _insideDot_ result if point is inside dot
        if (Noise(sCell + .5f, tCell + .5f) > 0) {
            Float radius = .35f;
            Float maxShift = 0.5f - radius;
            Float sCenter =
                sCell + maxShift * Noise(sCell + 1.5f, tCell + 2.8f);
            Float tCenter =
                tCell + maxShift * Noise(sCell + 4.5f, tCell + 9.8f);
            Vector2f dst = st - Point2f(sCenter, tCenter);
            if (LengthSquared(dst) < radius * radius)
                return true;
        }
        return false;
    }
};

class alignas(8) FloatDotsTexture : public DotsBase {
  public:
    FloatDotsTexture(TextureMapping2DHandle mapping,
                     FloatTextureHandle outsideDot,
                     FloatTextureHandle insideDot)
        : mapping(mapping), outsideDot(outsideDot), insideDot(insideDot) {}

    PBRT_HOST_DEVICE_INLINE
    Float Evaluate(const TextureEvalContext &ctx) const {
        Vector2f dstdx, dstdy;
        Point2f st = mapping.Map(ctx, &dstdx, &dstdy);
        return Inside(st) ? insideDot.Evaluate(ctx) : outsideDot.Evaluate(ctx);
    }

    static FloatDotsTexture *Create(const Transform &worldFromTexture,
                                    const TextureParameterDictionary &dict,
                                    const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // DotsTexture Private Data
    TextureMapping2DHandle mapping;
    FloatTextureHandle outsideDot, insideDot;
};

class alignas(8) SpectrumDotsTexture : public DotsBase {
  public:
    SpectrumDotsTexture(TextureMapping2DHandle mapping,
                        SpectrumTextureHandle outsideDot,
                        SpectrumTextureHandle insideDot)
        : mapping(mapping), outsideDot(outsideDot), insideDot(insideDot) {}

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum Evaluate(const TextureEvalContext &ctx,
                             const SampledWavelengths &lambda) const {
        Vector2f dstdx, dstdy;
        Point2f st = mapping.Map(ctx, &dstdx, &dstdy);
        return Inside(st) ? insideDot.Evaluate(ctx, lambda) :
            outsideDot.Evaluate(ctx, lambda);
    }

    static SpectrumDotsTexture *Create(const Transform &worldFromTexture,
                                       const TextureParameterDictionary &dict,
                                       const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // DotsTexture Private Data
    TextureMapping2DHandle mapping;
    SpectrumTextureHandle outsideDot, insideDot;
};

// FBmTexture Declarations
class alignas(8) FBmTexture {
  public:
    // FBmTexture Public Methods
    FBmTexture(TextureMapping3DHandle mapping, int octaves, Float omega)
        : mapping(mapping), omega(omega), octaves(octaves) {}

    PBRT_HOST_DEVICE_INLINE
    Float Evaluate(const TextureEvalContext &ctx) const {
        Vector3f dpdx, dpdy;
        Point3f P = mapping.Map(ctx, &dpdx, &dpdy);
        return FBm(P, dpdx, dpdy, omega, octaves);
    }

    static FBmTexture *Create(const Transform &worldFromTexture,
                              const TextureParameterDictionary &dict,
                              const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    TextureMapping3DHandle mapping;
    Float omega;
    int octaves;
};

// TexInfo Declarations
struct TexInfo {
    TexInfo(const std::string &f, const std::string &filt, Float ma,
            WrapMode wm, const ColorEncoding *encoding)
      : filename(f), filter(filt), maxAniso(ma), wrapMode(wm), encoding(encoding) {}
    std::string filename;
    std::string filter;
    Float maxAniso;
    WrapMode wrapMode;
    const ColorEncoding *encoding;
    bool operator<(const TexInfo &t2) const {
        return std::tie(filename, filter, maxAniso, encoding, wrapMode) <
               std::tie(t2.filename, t2.filter, t2.maxAniso, t2.encoding,
                        t2.wrapMode);
    }

    std::string ToString() const;
};

// ImageTexture Declarations
class ImageTextureBase {
  public:
    ImageTextureBase(TextureMapping2DHandle m,
                     const std::string &filename, const std::string &filter,
                     Float maxAniso, WrapMode wm, Float scale,
                     const ColorEncoding *encoding, Allocator alloc);

    static void ClearCache() {
        textureCache.clear();
    }

    TextureMapping2DHandle mapping;
    Float scale;
    MIPMap *mipmap;

 private:
    // ImageTexture Private Methods
    static MIPMap *GetTexture(const std::string &filename,
                              const std::string &filter, Float maxAniso,
                              WrapMode wm, const ColorEncoding *encoding,
                              Allocator alloc);

    // ImageTexture Private Data
    static std::mutex textureCacheMutex;
    static std::map<TexInfo, std::unique_ptr<MIPMap>> textureCache;
};

class alignas(8) FloatImageTexture : public ImageTextureBase {
  public:
    FloatImageTexture(TextureMapping2DHandle m,
                      const std::string &filename, const std::string &filter,
                      Float maxAniso, WrapMode wm, Float scale,
                      const ColorEncoding *encoding, Allocator alloc)
        : ImageTextureBase(m, filename, filter, maxAniso, wm, scale,
                           encoding, alloc) {}
    PBRT_HOST_DEVICE_INLINE
    Float Evaluate(const TextureEvalContext &ctx) const {
#ifdef __CUDA_ARCH__
        assert(!"BAH FIXME");
#endif
        if (!mipmap) return scale;
        Vector2f dstdx, dstdy;
        Point2f st = mapping.Map(ctx, &dstdx, &dstdy);
        // Texture coordinates are (0,0) in the lower left corner, but
        // image coordinates are (0,0) in the upper left.
        st[1] = 1 - st[1];
        return scale * mipmap->Lookup<Float>(st, dstdx, dstdy);
    }

    static FloatImageTexture *Create(const Transform &worldFromTexture,
                                     const TextureParameterDictionary &dict,
                                     const FileLoc *loc, Allocator alloc);

    std::string ToString() const;
};

class alignas(8) SpectrumImageTexture : public ImageTextureBase {
  public:
    SpectrumImageTexture(TextureMapping2DHandle m,
                         const std::string &filename, const std::string &filter,
                         Float maxAniso, WrapMode wm, Float scale,
                         const ColorEncoding *encoding, Allocator alloc)
        : ImageTextureBase(m, filename, filter, maxAniso, wm, scale,
                           encoding, alloc) {}

    PBRT_HOST_DEVICE
    SampledSpectrum Evaluate(const TextureEvalContext &ctx,
                             const SampledWavelengths &lambda) const;

    static SpectrumImageTexture *Create(const Transform &worldFromTexture,
                                        const TextureParameterDictionary &dict,
                                        const FileLoc *loc, Allocator alloc);

    std::string ToString() const;
};

#if defined(PBRT_HAVE_OPTIX) && defined(__NVCC__)
class alignas(8) GPUSpectrumImageTexture {
public:
    GPUSpectrumImageTexture(TextureMapping2DHandle mapping,
                            cudaTextureObject_t texObj, Float scale,
                            bool isSingleChannel, const RGBColorSpace *colorSpace)
        : mapping(mapping), texObj(texObj), scale(scale), isSingleChannel(isSingleChannel),
          colorSpace(colorSpace) { }

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum Evaluate(const TextureEvalContext &ctx,
                             const SampledWavelengths &lambda) const {
#ifndef __CUDA_ARCH__
        LOG_FATAL("GPUSpectrumImageTexture::Evaluate called from CPU");
        return SampledSpectrum(0);
#else
        // flip y coord since image has (0,0) at upper left, texture at lower left
        Vector2f dstdx, dstdy;
        Point2f st = mapping.Map(ctx, &dstdx, &dstdy);
        RGB rgb;
        if (isSingleChannel) {
            float tex = scale * tex2D<float>(texObj, st[0], 1 - st[1]);
            rgb = RGB(tex, tex, tex);
        } else {
            float4 tex = tex2D<float4>(texObj, st[0], 1 - st[1]);
            rgb = scale * RGB(tex.x, tex.y, tex.z);
        }
        if (std::max({rgb.r, rgb.g, rgb.b}) > 1)
            return RGBSpectrum(*colorSpace, rgb).Sample(lambda);
        return RGBReflectanceSpectrum(*colorSpace, rgb).Sample(lambda);
#endif
    }

    static GPUSpectrumImageTexture *Create(const Transform &worldFromTexture,
                                           const TextureParameterDictionary &dict,
                                           const FileLoc *loc, Allocator alloc);

    std::string ToString() const { return "GPUSpectrumImageTexture"; }

    TextureMapping2DHandle mapping;
    cudaTextureObject_t texObj;
    Float scale;
    bool isSingleChannel;
    const RGBColorSpace *colorSpace;
};

class alignas(8) GPUFloatImageTexture {
public:
    GPUFloatImageTexture(TextureMapping2DHandle mapping,
                         cudaTextureObject_t texObj, Float scale)
        : mapping(mapping), texObj(texObj), scale(scale) { }

    PBRT_HOST_DEVICE_INLINE
    Float Evaluate(const TextureEvalContext &ctx) const {
#ifndef __CUDA_ARCH__
        LOG_FATAL("GPUSpectrumImageTexture::Evaluate called from CPU");
        return 0;
#else
        Vector2f dstdx, dstdy;
        Point2f st = mapping.Map(ctx, &dstdx, &dstdy);
        // flip y coord since image has (0,0) at upper left, texture at lower left
        return scale * tex2D<float>(texObj, st[0], 1 - st[1]);
#endif
    }

    static GPUFloatImageTexture *Create(const Transform &worldFromTexture,
                                        const TextureParameterDictionary &dict,
                                        const FileLoc *loc, Allocator alloc);

    std::string ToString() const { return "GPUFloatImageTexture"; }

    TextureMapping2DHandle mapping;
    cudaTextureObject_t texObj;
    Float scale;
};

#else // Optix && NVCC

class alignas(8) GPUSpectrumImageTexture {
public:
    SampledSpectrum Evaluate(const TextureEvalContext &ctx,
                             const SampledWavelengths &lambda) const {
        LOG_FATAL("GPUSpectrumImageTexture::Evaluate called from CPU");
        return SampledSpectrum(0);
    }

    static GPUSpectrumImageTexture *Create(const Transform &worldFromTexture,
                                           const TextureParameterDictionary &dict,
                                           const FileLoc *loc, Allocator alloc) {
        LOG_FATAL("GPUSpectrumImageTexture::Create called in non-GPU configuration.");
        return nullptr;
    }

    std::string ToString() const { return "GPUSpectrumImageTexture"; }
};

class alignas(8) GPUFloatImageTexture {
public:
    Float Evaluate(const TextureEvalContext &) const {
        LOG_FATAL("GPUFloatImageTexture::Evaluate called from CPU");
        return 0;
    }

    static GPUFloatImageTexture *Create(const Transform &worldFromTexture,
                                        const TextureParameterDictionary &dict,
                                        const FileLoc *loc, Allocator alloc) {
        LOG_FATAL("GPUFloatImageTexture::Create called in non-GPU configuration.");
        return nullptr;
    }

    std::string ToString() const { return "GPUFloatImageTexture"; }
};

#endif // Optix && NVCC


// MarbleTexture Declarations
class alignas(8) MarbleTexture {
  public:
    // MarbleTexture Public Methods
    MarbleTexture(TextureMapping3DHandle mapping, int octaves,
                  Float omega, Float scale, Float variation)
        : mapping(mapping), octaves(octaves), omega(omega),
          scale(scale), variation(variation) {}

    PBRT_HOST_DEVICE
    SampledSpectrum Evaluate(const TextureEvalContext &ctx,
                             const SampledWavelengths &lambda) const;

    static MarbleTexture *Create(const Transform &worldFromTexture,
                                 const TextureParameterDictionary &dict,
                                 const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // MarbleTexture Private Data
    TextureMapping3DHandle mapping;
    int octaves;
    Float omega, scale, variation;
};

// MixTexture Declarations
class alignas(8) FloatMixTexture {
  public:
    // MixTexture Public Methods
    FloatMixTexture(FloatTextureHandle tex1, FloatTextureHandle tex2,
                    FloatTextureHandle amount)
        :  tex1(tex1), tex2(tex2), amount(amount) {}

    PBRT_HOST_DEVICE_INLINE
    Float Evaluate(const TextureEvalContext &ctx) const {
        Float t1 = tex1.Evaluate(ctx), t2 = tex2.Evaluate(ctx);
        Float amt = amount.Evaluate(ctx);
        return (1 - amt) * t1 + amt * t2;
    }

    static FloatMixTexture *Create(const Transform &worldFromTexture,
                                   const TextureParameterDictionary &dict,
                                   const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    FloatTextureHandle tex1, tex2;
    FloatTextureHandle amount;
};

class alignas(8) SpectrumMixTexture {
  public:
    // MixTexture Public Methods
   SpectrumMixTexture(SpectrumTextureHandle tex1,
                      SpectrumTextureHandle tex2,
                      FloatTextureHandle amount)
       : tex1(tex1), tex2(tex2), amount(amount) {}

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum Evaluate(const TextureEvalContext &ctx,
                             const SampledWavelengths &lambda) const {
        SampledSpectrum t1 = tex1.Evaluate(ctx, lambda);
        SampledSpectrum t2 = tex2.Evaluate(ctx, lambda);
        Float amt = amount.Evaluate(ctx);
        return (1 - amt) * t1 + amt * t2;
    }

    static SpectrumMixTexture *Create(const Transform &worldFromTexture,
                                      const TextureParameterDictionary &dict,
                                      const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    SpectrumTextureHandle tex1, tex2;
    FloatTextureHandle amount;
};


// PtexTexture Declarations
class PtexTextureBase {
  public:
    PtexTextureBase(const std::string &filename,
                    const ColorEncoding *encoding);
    ~PtexTextureBase();

  protected:
    int SampleTexture(const TextureEvalContext &, float *result) const;
    std::string BaseToString() const;

  private:
    bool valid;
    std::string filename;
    const ColorEncoding *encoding;
};

class alignas(8) FloatPtexTexture : public PtexTextureBase {
 public:
    FloatPtexTexture(const std::string &filename, const ColorEncoding *encoding)
        : PtexTextureBase(filename, encoding) {}

    PBRT_HOST_DEVICE
    Float Evaluate(const TextureEvalContext &ctx) const;
    static FloatPtexTexture *Create(const Transform &worldFromTexture,
                                    const TextureParameterDictionary &dict,
                                    const FileLoc *loc, Allocator alloc);
    std::string ToString() const;
};

class alignas(8) SpectrumPtexTexture : public PtexTextureBase {
 public:
    SpectrumPtexTexture(const std::string &filename,
                        const ColorEncoding *encoding)
        : PtexTextureBase(filename, encoding) {}

    PBRT_HOST_DEVICE
    SampledSpectrum Evaluate(const TextureEvalContext &ctx,
                             const SampledWavelengths &lambda) const;

    static SpectrumPtexTexture *Create(const Transform &worldFromTexture,
                                       const TextureParameterDictionary &dict,
                                       const FileLoc *loc, Allocator alloc);

    std::string ToString() const;
};

class alignas(8) FloatScaledTexture {
public:
    FloatScaledTexture(FloatTextureHandle tex, FloatTextureHandle scale)
        : tex(tex), scale(scale) { }

    static FloatTextureHandle Create(const Transform &worldFromTexture,
                                     const TextureParameterDictionary &dict,
                                     const FileLoc *loc, Allocator alloc);

    PBRT_HOST_DEVICE_INLINE
    Float Evaluate(const TextureEvalContext &ctx) const {
        return tex.Evaluate(ctx) * scale.Evaluate(ctx);
    }

    std::string ToString() const;

private:
    FloatTextureHandle tex, scale;
};

class alignas(8) SpectrumScaledTexture {
public:
    SpectrumScaledTexture(SpectrumTextureHandle tex,
                          FloatTextureHandle scale)
        : tex(tex), scale(scale) { }

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum Evaluate(const TextureEvalContext &ctx,
                             const SampledWavelengths &lambda) const {
        return tex.Evaluate(ctx, lambda) * scale.Evaluate(ctx);
    }

    static SpectrumTextureHandle Create(const Transform &worldFromTexture,
                                        const TextureParameterDictionary &dict,
                                        const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

private:
    SpectrumTextureHandle tex;
    FloatTextureHandle scale;
};


// UVTexture Declarations
class alignas(8) UVTexture {
  public:
    // UVTexture Public Methods
    UVTexture(TextureMapping2DHandle mapping)
        : mapping(mapping) {}

    PBRT_HOST_DEVICE
    SampledSpectrum Evaluate(const TextureEvalContext &ctx,
                             const SampledWavelengths &lambda) const;

    static UVTexture *Create(const Transform &worldFromTexture,
                             const TextureParameterDictionary &dict,
                             const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    TextureMapping2DHandle mapping;
};

// WindyTexture Declarations
class alignas(8) WindyTexture {
  public:
    // WindyTexture Public Methods
    WindyTexture(TextureMapping3DHandle mapping)
        : mapping(mapping) {}

    PBRT_HOST_DEVICE_INLINE
    Float Evaluate(const TextureEvalContext &ctx) const {
        Vector3f dpdx, dpdy;
        Point3f P = mapping.Map(ctx, &dpdx, &dpdy);
        Float windStrength = FBm(.1f * P, .1f * dpdx, .1f * dpdy, .5, 3);
        Float waveHeight = FBm(P, dpdx, dpdy, .5, 6);
        return std::abs(windStrength) * waveHeight;
    }

    static WindyTexture *Create(const Transform &worldFromTexture,
                                const TextureParameterDictionary &dict,
                                const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    TextureMapping3DHandle mapping;
};

// WrinkledTexture Declarations
class alignas(8) WrinkledTexture {
  public:
    // WrinkledTexture Public Methods
    WrinkledTexture(TextureMapping3DHandle mapping, int octaves, Float omega)
        : mapping(mapping), octaves(octaves), omega(omega) {}

    PBRT_HOST_DEVICE_INLINE
    Float Evaluate(const TextureEvalContext &ctx) const {
        Vector3f dpdx, dpdy;
        Point3f p = mapping.Map(ctx, &dpdx, &dpdy);
        return Turbulence(p, dpdx, dpdy, omega, octaves);
    }

    static WrinkledTexture *Create(const Transform &worldFromTexture,
                                   const TextureParameterDictionary &dict,
                                   const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // WrinkledTexture Private Data
    TextureMapping3DHandle mapping;
    int octaves;
    Float omega;
};

inline Float
FloatTextureHandle::Evaluate(const TextureEvalContext &ctx) const {
    if (Tag() == TypeIndex<FloatImageTexture>())
        return Cast<FloatImageTexture>()->Evaluate(ctx);
    else if (Tag() == TypeIndex<GPUFloatImageTexture>())
        return Cast<GPUFloatImageTexture>()->Evaluate(ctx);
    else if (Tag() == TypeIndex<FloatMixTexture>())
        return Cast<FloatMixTexture>()->Evaluate(ctx);
    else if (Tag() == TypeIndex<FloatScaledTexture>())
        return Cast<FloatScaledTexture>()->Evaluate(ctx);
    else if (Tag() == TypeIndex<FloatConstantTexture>())
        return Cast<FloatConstantTexture>()->Evaluate(ctx);
    else
        return EvaluateRare(ctx);
}

inline SampledSpectrum
SpectrumTextureHandle::Evaluate(const TextureEvalContext &ctx,
                                const SampledWavelengths &lambda) const {
    if (Tag() == TypeIndex<GPUSpectrumImageTexture>())
        return Cast<GPUSpectrumImageTexture>()->Evaluate(ctx, lambda);
    else if (Tag() == TypeIndex<SpectrumMixTexture>())
        return Cast<SpectrumMixTexture>()->Evaluate(ctx, lambda);
    else if (Tag() == TypeIndex<SpectrumScaledTexture>())
        return Cast<SpectrumScaledTexture>()->Evaluate(ctx, lambda);
    else if (Tag() == TypeIndex<SpectrumImageTexture>())
        return Cast<SpectrumImageTexture>()->Evaluate(ctx, lambda);
    else if (Tag() == TypeIndex<SpectrumConstantTexture>())
        return Cast<SpectrumConstantTexture>()->Evaluate(ctx, lambda);
    else
        return EvaluateRare(ctx, lambda);
}

class UniversalTextureEvaluator {
public:
    PBRT_HOST_DEVICE
    bool Matches(std::initializer_list<FloatTextureHandle>,
                 std::initializer_list<SpectrumTextureHandle>) {
        return true;
    }
    PBRT_HOST_DEVICE
    Float operator()(FloatTextureHandle tex, const TextureEvalContext &ctx) {
        return tex.Evaluate(ctx);
    }
    PBRT_HOST_DEVICE
    SampledSpectrum operator()(SpectrumTextureHandle tex,
                               const TextureEvalContext &ctx,
                               const SampledWavelengths &lambda) {
        return tex.Evaluate(ctx, lambda);
    }
};

class BasicTextureEvaluator {
public:
    PBRT_HOST_DEVICE
    bool Matches(std::initializer_list<FloatTextureHandle> ftex,
                 std::initializer_list<SpectrumTextureHandle> stex) {
        for (auto f : ftex)
            if (f && (!f.Is<FloatConstantTexture>() &&
                      !f.Is<GPUFloatImageTexture>())) {
                return false;
            }
        for (auto s : stex)
            if (s && (!s.Is<SpectrumConstantTexture>() &&
                      !s.Is<GPUSpectrumImageTexture>())) {
                return false;
            }
        return true;
    }

    PBRT_HOST_DEVICE
    Float operator()(FloatTextureHandle tex, const TextureEvalContext &ctx) {
        if (FloatConstantTexture *fc = tex.CastOrNullptr<FloatConstantTexture>())
            return fc->Evaluate(ctx);
        else {
            GPUFloatImageTexture *fg = tex.CastOrNullptr<GPUFloatImageTexture>();
            DCHECK(fg != nullptr);
            return fg->Evaluate(ctx);
        }
    }

    PBRT_HOST_DEVICE
    SampledSpectrum operator()(SpectrumTextureHandle tex,
                               const TextureEvalContext &ctx,
                               const SampledWavelengths &lambda) {
        if (SpectrumConstantTexture *sc = tex.CastOrNullptr<SpectrumConstantTexture>())
            return sc->Evaluate(ctx, lambda);
        else {
            GPUSpectrumImageTexture *sg = tex.CastOrNullptr<GPUSpectrumImageTexture>();
            DCHECK(sg != nullptr);
            return sg->Evaluate(ctx, lambda);
        }
    }
};

}  // namespace pbrt

#endif  // PBRT_TEXTURES_WRINKLED_H
