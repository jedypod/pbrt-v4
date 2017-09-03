
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


// media/homogeneous.cpp*
#include "media/homogeneous.h"

#include "error.h"
#include "sampler.h"
#include "interaction.h"
#include "paramset.h"
#include "stats.h"

#include <cmath>

namespace pbrt {

// HomogeneousMedium Method Definitions
std::shared_ptr<HomogeneousMedium> HomogeneousMedium::Create(
        const ParamSet &ps, const std::shared_ptr<const ParamSet> &attributes) {
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

    return std::make_shared<HomogeneousMedium>(sig_a, sig_s, g, attributes);
}

Spectrum HomogeneousMedium::Tr(const Ray &ray, Sampler &sampler) const {
    ProfilePhase _(Prof::MediumTr);
    return Exp(-sigma_t * std::min(ray.tMax * Length(ray.d), MaxFloat));
}

Spectrum HomogeneousMedium::Sample(const Ray &ray, Sampler &sampler,
                                   MemoryArena &arena,
                                   MediumInteraction *mi) const {
    ProfilePhase _(Prof::MediumSample);
    // Sample a channel and distance along the ray
    int channel = sampler.GetDiscrete1D(Spectrum::nSamples);
    Float dist = -std::log(1 - sampler.Get1D()) / sigma_t[channel];
    Float t = std::min(dist * Length(ray.d), ray.tMax);
    bool sampledMedium = t < ray.tMax;
    if (sampledMedium)
        *mi = MediumInteraction(ray(t), -ray.d, ray.time, this,
                                arena.Alloc<HenyeyGreenstein>(g));

    // Compute the transmittance and sampling density
    Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) * Length(ray.d));

    // Return weighting factor for scattering from homogeneous medium
    Spectrum density = sampledMedium ? (sigma_t * Tr) : Tr;
    Float pdf = 0;
    for (int i = 0; i < Spectrum::nSamples; ++i) pdf += density[i];
    pdf *= 1 / (Float)Spectrum::nSamples;
    if (pdf == 0) {
        CHECK(Tr.IsBlack());
        pdf = 1;
    }
    return sampledMedium ? (Tr * sigma_s / pdf) : (Tr / pdf);
}

}  // namespace pbrt
