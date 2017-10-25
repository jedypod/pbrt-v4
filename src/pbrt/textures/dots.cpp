
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

// textures/dots.cpp*
#include <pbrt/textures/dots.h>

#include <pbrt/core/error.h>
#include <pbrt/core/paramset.h>

namespace pbrt {

// DotsTexture Method Definitions
std::shared_ptr<DotsTexture<Float>> CreateDotsFloatTexture(
    const Transform &tex2world, const TextureParams &tp) {
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
    return std::make_shared<DotsTexture<Float>>(
        std::move(map), tp.GetFloatTexture("inside", 1.f),
        tp.GetFloatTexture("outside", 0.f));
}

std::shared_ptr<DotsTexture<Spectrum>> CreateDotsSpectrumTexture(
    const Transform &tex2world, const TextureParams &tp) {
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
    return std::make_shared<DotsTexture<Spectrum>>(
        std::move(map), tp.GetSpectrumTexture("inside", 1.f),
        tp.GetSpectrumTexture("outside", 0.f));
}

}  // namespace pbrt
