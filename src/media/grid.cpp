
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


// media/grid.cpp*
#include "media/grid.h"

#include "paramset.h"
#include "sampler.h"
#include "stats.h"
#include "interaction.h"

namespace pbrt {

STAT_RATIO("Media/Grid steps per Tr() call", nTrSteps, nTrCalls);

// GridDensityMedium Method Definitions
std::shared_ptr<GridDensityMedium> GridDensityMedium::Create(
        const ParamSet &ps, const Transform &mediumToWorld,
        const std::shared_ptr<const ParamSet> &attributes) {
    Float sig_a_rgb[3] = {.0011f, .0024f, .014f},
          sig_s_rgb[3] = {2.55f, 3.21f, 3.77f};
    Spectrum sig_a = Spectrum::FromRGB(sig_a_rgb),
             sig_s = Spectrum::FromRGB(sig_s_rgb);
    std::string preset = ps.GetOneString("preset", "");
    bool found = GetMediumScatteringProperties(preset, &sig_a, &sig_s);
    if (preset != "" && !found)
        Warning("Material preset \"%s\" not found.  Using defaults.",
                preset.c_str());
    Float scale = ps.GetOneFloat("scale", 1.f);
    Float g = ps.GetOneFloat("g", 0.0f);
    sig_a = ps.GetOneSpectrum("sigma_a", sig_a) * scale;
    sig_s = ps.GetOneSpectrum("sigma_s", sig_s) * scale;

    gtl::ArraySlice<Float> data = ps.GetFloatArray("density");
    if (data.empty()) {
        Error("No \"density\" values provided for heterogeneous medium?");
        return nullptr;
    }
    int nx = ps.GetOneInt("nx", 1);
    int ny = ps.GetOneInt("ny", 1);
    int nz = ps.GetOneInt("nz", 1);
    Point3f p0 = ps.GetOnePoint3f("p0", Point3f(0.f, 0.f, 0.f));
    Point3f p1 = ps.GetOnePoint3f("p1", Point3f(1.f, 1.f, 1.f));
    if (data.size() != nx * ny * nz) {
        Error(
              "GridDensityMedium has %d density values; expected nx*ny*nz = "
              "%d",
              (int)data.size(), nx * ny * nz);
        return nullptr;
    }

    Transform dataToMedium = Translate(Vector3f(p0)) *
        Scale(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
    return std::make_shared<GridDensityMedium>(
        sig_a, sig_s, g, nx, ny, nz, mediumToWorld * dataToMedium, data,
        attributes);
}


Float GridDensityMedium::Density(const Point3f &p) const {
    // Compute voxel coordinates and offsets for _p_
    Point3f pSamples(p.x * nx - .5f, p.y * ny - .5f, p.z * nz - .5f);
    Point3i pi = (Point3i)Floor(pSamples);
    Vector3f d = pSamples - (Point3f)pi;

    // Trilinearly interpolate density values to compute local density
    Float d00 = Lerp(d.x, D(pi), D(pi + Vector3i(1, 0, 0)));
    Float d10 = Lerp(d.x, D(pi + Vector3i(0, 1, 0)), D(pi + Vector3i(1, 1, 0)));
    Float d01 = Lerp(d.x, D(pi + Vector3i(0, 0, 1)), D(pi + Vector3i(1, 0, 1)));
    Float d11 = Lerp(d.x, D(pi + Vector3i(0, 1, 1)), D(pi + Vector3i(1, 1, 1)));
    Float d0 = Lerp(d.y, d00, d10);
    Float d1 = Lerp(d.y, d01, d11);
    return Lerp(d.z, d0, d1);
}

Spectrum GridDensityMedium::Sample(const Ray &rWorld, Sampler &sampler,
                                   MemoryArena &arena,
                                   MediumInteraction *mi) const {
    ProfilePhase _(Prof::MediumSample);
    Ray ray = WorldToMedium(
        Ray(rWorld.o, Normalize(rWorld.d), rWorld.tMax * Length(rWorld.d)));
    // Compute $[\tmin, \tmax]$ interval of _ray_'s overlap with medium bounds
    const Bounds3f b(Point3f(0, 0, 0), Point3f(1, 1, 1));
    Float tMin, tMax;
    if (!b.IntersectP(ray, &tMin, &tMax)) return Spectrum(1.f);

    // Run delta-tracking iterations to sample a medium interaction
    Float t = tMin;
    while (true) {
        t -= std::log(1 - sampler.Get1D()) * invMaxDensity / sigma_t;
        if (t >= tMax) break;
        if (Density(ray(t)) * invMaxDensity > sampler.Get1D()) {
            // Populate _mi_ with medium interaction information and return
            PhaseFunction *phase = arena.Alloc<HenyeyGreenstein>(g);
            *mi = MediumInteraction(rWorld(t), -rWorld.d, rWorld.time, this,
                                    phase);
            return sigma_s / sigma_t;
        }
    }
    return Spectrum(1.f);
}

Spectrum GridDensityMedium::Tr(const Ray &rWorld, Sampler &sampler) const {
    ProfilePhase _(Prof::MediumTr);
    ++nTrCalls;

    Ray ray = WorldToMedium(
        Ray(rWorld.o, Normalize(rWorld.d), rWorld.tMax * Length(rWorld.d)));
    // Compute $[\tmin, \tmax]$ interval of _ray_'s overlap with medium bounds
    const Bounds3f b(Point3f(0, 0, 0), Point3f(1, 1, 1));
    Float tMin, tMax;
    if (!b.IntersectP(ray, &tMin, &tMax)) return Spectrum(1.f);

    // Perform ratio tracking to estimate the transmittance value
    Float Tr = 1, t = tMin;
    while (true) {
        ++nTrSteps;
        t -= std::log(1 - sampler.Get1D()) * invMaxDensity / sigma_t;
        if (t >= tMax) break;
        Float density = Density(ray(t));
        Tr *= 1 - std::max((Float)0, density * invMaxDensity);
        // Added after book publication: when transmittance gets low,
        // start applying Russian roulette to terminate sampling.
        const Float rrThreshold = .1;
        if (Tr < rrThreshold) {
            Float q = std::max((Float).05, 1 - Tr);
            if (sampler.Get1D() < q) return 0;
            Tr /= 1 - q;
        }
    }
    return Spectrum(Tr);
}

std::string GridDensityMedium::ToString() const {
    return StringPrintf("[ GridDensityMedium sigma_a: %s sigma_s: %s sigma_t: %f "
                        " nx: %d ny: %d nz: %d WorldToMedium: %s invMaxDensity: %f ]",
                        sigma_a.ToString().c_str(), sigma_s.ToString().c_str(), sigma_t,
                        nx, ny, nz, WorldToMedium.ToString().c_str(), invMaxDensity);
}

}  // namespace pbrt
