
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

// textures/checkerboard.cpp*
#include <pbrt/textures/checkerboard.h>

#include <pbrt/core/error.h>

namespace pbrt {

// CheckerboardTexture Method Definitions
std::shared_ptr<Texture<Float>> CreateCheckerboardFloatTexture(
    const Transform &tex2world, const TextureParams &tp) {
    int dim = tp.GetOneInt("dimension", 2);
    if (dim != 2 && dim != 3) {
        Error("%d dimensional checkerboard texture not supported", dim);
        return nullptr;
    }
    std::shared_ptr<Texture<Float>> tex1 = tp.GetFloatTexture("tex1", 1.f);
    std::shared_ptr<Texture<Float>> tex2 = tp.GetFloatTexture("tex2", 0.f);
    if (dim == 2) {
        // Initialize 2D texture mapping _map_ from _tp_
        std::unique_ptr<TextureMapping2D> map;
        std::string type = tp.GetOneString("mapping", "uv");
        if (type == "uv") {
            Float su = tp.GetOneFloat("uscale", 1.);
            Float sv = tp.GetOneFloat("vscale", 1.);
            Float du = tp.GetOneFloat("udelta", 0.);
            Float dv = tp.GetOneFloat("vdelta", 0.);
            map = std::make_unique<UVMapping2D>(su, sv, du, dv);
        } else if (type == "spherical")
            map = std::make_unique<SphericalMapping2D>(Inverse(tex2world));
        else if (type == "cylindrical")
            map = std::make_unique<CylindricalMapping2D>(Inverse(tex2world));
        else if (type == "planar")
            map = std::make_unique<PlanarMapping2D>(
                tp.GetOneVector3f("v1", Vector3f(1, 0, 0)),
                tp.GetOneVector3f("v2", Vector3f(0, 1, 0)),
                tp.GetOneFloat("udelta", 0.f), tp.GetOneFloat("vdelta", 0.f));
        else {
            Error("2D texture mapping \"%s\" unknown", type.c_str());
            map = std::make_unique<UVMapping2D>();
        }

        // Compute _aaMethod_ for _CheckerboardTexture_
        std::string aa = tp.GetOneString("aamode", "closedform");
        AAMethod aaMethod;
        if (aa == "none")
            aaMethod = AAMethod::None;
        else if (aa == "closedform")
            aaMethod = AAMethod::ClosedForm;
        else {
            Warning(
                "Antialiasing mode \"%s\" not understood by "
                "Checkerboard2DTexture; using \"closedform\"",
                aa.c_str());
            aaMethod = AAMethod::ClosedForm;
        }
        return std::make_shared<Checkerboard2DTexture<Float>>(
            std::move(map), tex1, tex2, aaMethod);
    } else {
        // Initialize 3D texture mapping _map_ from _tp_
        std::unique_ptr<TextureMapping3D> map(new TransformMapping3D(tex2world));
        return std::make_shared<Checkerboard3DTexture<Float>>(std::move(map),
                                                              tex1, tex2);
    }
}

std::shared_ptr<Texture<Spectrum>> CreateCheckerboardSpectrumTexture(
    const Transform &tex2world, const TextureParams &tp) {
    int dim = tp.GetOneInt("dimension", 2);
    if (dim != 2 && dim != 3) {
        Error("%d dimensional checkerboard texture not supported", dim);
        return nullptr;
    }
    std::shared_ptr<Texture<Spectrum>> tex1 =
        tp.GetSpectrumTexture("tex1", 1.f);
    std::shared_ptr<Texture<Spectrum>> tex2 =
        tp.GetSpectrumTexture("tex2", 0.f);
    if (dim == 2) {
        // Initialize 2D texture mapping _map_ from _tp_
        std::unique_ptr<TextureMapping2D> map;
        std::string type = tp.GetOneString("mapping", "uv");
        if (type == "uv") {
            Float su = tp.GetOneFloat("uscale", 1.);
            Float sv = tp.GetOneFloat("vscale", 1.);
            Float du = tp.GetOneFloat("udelta", 0.);
            Float dv = tp.GetOneFloat("vdelta", 0.);
            map = std::make_unique<UVMapping2D>(su, sv, du, dv);
        } else if (type == "spherical")
            map = std::make_unique<SphericalMapping2D>(Inverse(tex2world));
        else if (type == "cylindrical")
            map = std::make_unique<CylindricalMapping2D>(Inverse(tex2world));
        else if (type == "planar")
            map = std::make_unique<PlanarMapping2D>(
                tp.GetOneVector3f("v1", Vector3f(1, 0, 0)),
                tp.GetOneVector3f("v2", Vector3f(0, 1, 0)),
                tp.GetOneFloat("udelta", 0.f), tp.GetOneFloat("vdelta", 0.f));
        else {
            Error("2D texture mapping \"%s\" unknown", type.c_str());
            map = std::make_unique<UVMapping2D>();
        }

        // Compute _aaMethod_ for _CheckerboardTexture_
        std::string aa = tp.GetOneString("aamode", "closedform");
        AAMethod aaMethod;
        if (aa == "none")
            aaMethod = AAMethod::None;
        else if (aa == "closedform")
            aaMethod = AAMethod::ClosedForm;
        else {
            Warning(
                "Antialiasing mode \"%s\" not understood by "
                "Checkerboard2DTexture; using \"closedform\"",
                aa.c_str());
            aaMethod = AAMethod::ClosedForm;
        }
        return std::make_shared<Checkerboard2DTexture<Spectrum>>(
            std::move(map), tex1, tex2, aaMethod);
    } else {
        // Initialize 3D texture mapping _map_ from _tp_
        std::unique_ptr<TextureMapping3D> map(new TransformMapping3D(tex2world));
        return std::make_shared<Checkerboard3DTexture<Spectrum>>(
            std::make_unique<TransformMapping3D>(tex2world), tex1, tex2);
    }
}

}  // namespace pbrt