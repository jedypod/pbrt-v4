
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


// media.cpp*
#include <pbrt/media.h>

#include <pbrt/base.h>
#include <pbrt/interaction.h>
#include <pbrt/paramdict.h>
#include <pbrt/primitive.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/octree.h>
#include <pbrt/util/print.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/scattering.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>

#include <algorithm>
#include <cmath>

namespace pbrt {

// HenyeyGreenstein Method Definitions
std::string HenyeyGreenstein::ToString() const {
    return StringPrintf("[ HenyeyGreenstein g: %f ]", g);
}

// Media Local Definitions
struct MeasuredSS {
    const char *name;
    RGB sigma_prime_s, sigma_a;  // mm^-1
};

bool GetMediumScatteringProperties(const std::string &name,
                                   SpectrumHandle *sigma_a,
                                   SpectrumHandle *sigma_s,
                                   Allocator alloc) {
    static MeasuredSS SubsurfaceParameterTable[] = {
        // From "A Practical Model for Subsurface Light Transport"
        // Jensen, Marschner, Levoy, Hanrahan
        // Proc SIGGRAPH 2001
        {"Apple", RGB(2.29, 2.39, 1.97), RGB(0.0030, 0.0034, 0.046)},
        {"Chicken1", RGB(0.15, 0.21, 0.38), RGB(0.015, 0.077, 0.19)},
        {"Chicken2", RGB(0.19, 0.25, 0.32), RGB(0.018, 0.088, 0.20)},
        {"Cream", RGB(7.38, 5.47, 3.15), RGB(0.0002, 0.0028, 0.0163)},
        {"Ketchup", RGB(0.18, 0.07, 0.03), RGB(0.061, 0.97, 1.45)},
        {"Marble", RGB(2.19, 2.62, 3.00), RGB(0.0021, 0.0041, 0.0071)},
        {"Potato", RGB(0.68, 0.70, 0.55), RGB(0.0024, 0.0090, 0.12)},
        {"Skimmilk", RGB(0.70, 1.22, 1.90), RGB(0.0014, 0.0025, 0.0142)},
        {"Skin1", RGB(0.74, 0.88, 1.01), RGB(0.032, 0.17, 0.48)},
        {"Skin2", RGB(1.09, 1.59, 1.79), RGB(0.013, 0.070, 0.145)},
        {"Spectralon", RGB(11.6, 20.4, 14.9), RGB(0.00, 0.00, 0.00)},
        {"Wholemilk", RGB(2.55, 3.21, 3.77), RGB(0.0011, 0.0024, 0.014)},

        // From "Acquiring Scattering Properties of Participating Media by
        // Dilution",
        // Narasimhan, Gupta, Donner, Ramamoorthi, Nayar, Jensen
        // Proc SIGGRAPH 2006
        {"Lowfat Milk", RGB(0.89187, 1.5136, 2.532), RGB(0.002875, 0.00575, 0.0115)},
        {"Reduced Milk", RGB(2.4858, 3.1669, 4.5214), RGB(0.0025556, 0.0051111, 0.012778)},
        {"Regular Milk", RGB(4.5513, 5.8294, 7.136), RGB(0.0015333, 0.0046, 0.019933)},
        {"Espresso", RGB(0.72378, 0.84557, 1.0247), RGB(4.7984, 6.5751, 8.8493)},
        {"Mint Mocha Coffee", RGB(0.31602, 0.38538, 0.48131), RGB(3.772, 5.8228, 7.82)},
        {"Lowfat Soy Milk", RGB(0.30576, 0.34233, 0.61664), RGB(0.0014375, 0.0071875, 0.035937)},
        {"Regular Soy Milk", RGB(0.59223, 0.73866, 1.4693), RGB(0.0019167, 0.0095833, 0.065167)},
        {"Lowfat Chocolate Milk", RGB(0.64925, 0.83916, 1.1057), RGB(0.0115, 0.0368, 0.1564)},
        {"Regular Chocolate Milk", RGB(1.4585, 2.1289, 2.9527), RGB(0.010063, 0.043125, 0.14375)},
        {"Coke", RGB(8.9053e-05, 8.372e-05, 0), RGB(0.10014, 0.16503, 0.2468)},
        {"Pepsi", RGB(6.1697e-05, 4.2564e-05, 0), RGB(0.091641, 0.14158, 0.20729)},
        {"Sprite", RGB(6.0306e-06, 6.4139e-06, 6.5504e-06), RGB(0.001886, 0.0018308, 0.0020025)},
        {"Gatorade", RGB(0.0024574, 0.003007, 0.0037325), RGB(0.024794, 0.019289, 0.008878)},
        {"Chardonnay", RGB(1.7982e-05, 1.3758e-05, 1.2023e-05), RGB(0.010782, 0.011855, 0.023997)},
        {"White Zinfandel", RGB(1.7501e-05, 1.9069e-05, 1.288e-05), RGB(0.012072, 0.016184, 0.019843)},
        {"Merlot", RGB(2.1129e-05, 0, 0), RGB(0.11632, 0.25191, 0.29434)},
        {"Budweiser Beer", RGB(2.4356e-05, 2.4079e-05, 1.0564e-05), RGB(0.011492, 0.024911, 0.057786)},
        {"Coors Light Beer", RGB(5.0922e-05, 4.301e-05, 0), RGB(0.006164, 0.013984, 0.034983)},
        {"Clorox", RGB(0.0024035, 0.0031373, 0.003991), RGB(0.0033542, 0.014892, 0.026297)},
        {"Apple Juice", RGB(0.00013612, 0.00015836, 0.000227), RGB(0.012957, 0.023741, 0.052184)},
        {"Cranberry Juice", RGB(0.00010402, 0.00011646, 7.8139e-05), RGB(0.039437, 0.094223, 0.12426)},
        {"Grape Juice", RGB(5.382e-05, 0, 0), RGB(0.10404, 0.23958, 0.29325)},
        {"Ruby Grapefruit Juice", RGB(0.011002, 0.010927, 0.011036), RGB(0.085867, 0.18314, 0.25262)},
        {"White Grapefruit Juice", RGB(0.22826, 0.23998, 0.32748), RGB(0.0138, 0.018831, 0.056781)},
        {"Shampoo", RGB(0.0007176, 0.0008303, 0.0009016), RGB(0.014107, 0.045693, 0.061717)},
        {"Strawberry Shampoo", RGB(0.00015671, 0.00015947, 1.518e-05), RGB(0.01449, 0.05796, 0.075823)},
        {"Head & Shoulders Shampoo", RGB(0.023805, 0.028804, 0.034306), RGB(0.084621, 0.15688, 0.20365)},
        {"Lemon Tea Powder", RGB(0.040224, 0.045264, 0.051081), RGB(2.4288, 4.5757, 7.2127)},
        {"Orange Powder", RGB(0.00015617, 0.00017482, 0.0001762), RGB(0.001449, 0.003441, 0.007863)},
        {"Pink Lemonade Powder", RGB(0.00012103, 0.00013073, 0.00012528), RGB(0.001165, 0.002366, 0.003195)},
        {"Cappuccino Powder", RGB(1.8436, 2.5851, 2.1662), RGB(35.844, 49.547, 61.084)},
        {"Salt Powder", RGB(0.027333, 0.032451, 0.031979), RGB(0.28415, 0.3257, 0.34148)},
        {"Sugar Powder", RGB(0.00022272, 0.00025513, 0.000271), RGB(0.012638, 0.031051, 0.050124)},
        {"Suisse Mocha Powder", RGB(2.7979, 3.5452, 4.3365), RGB(17.502, 27.004, 35.433)},
        {"Pacific Ocean Surface Water", RGB(0.0001764, 0.00032095, 0.00019617), RGB(0.031845, 0.031324, 0.030147)}};

    for (MeasuredSS &mss : SubsurfaceParameterTable) {
        if (name == mss.name) {
            *sigma_a = alloc.new_object<RGBSpectrum>(*RGBColorSpace::sRGB, mss.sigma_a);
            *sigma_s = alloc.new_object<RGBSpectrum>(*RGBColorSpace::sRGB, mss.sigma_prime_s);
            return true;
        }
    }
    return false;
}

// HomogeneousMedium Method Definitions
std::unique_ptr<HomogeneousMedium> HomogeneousMedium::Create(
    const ParameterDictionary &dict, Allocator alloc) {
    SpectrumHandle sig_a = nullptr, sig_s = nullptr;
    std::string preset = dict.GetOneString("preset", "");
    if (!preset.empty()) {
        if (!GetMediumScatteringProperties(preset, &sig_a, &sig_s, alloc))
            Warning("Material preset \"%s\" not found.", preset);
    }
    if (sig_a == nullptr) {
        sig_a = dict.GetOneSpectrum("sigma_a", nullptr, SpectrumType::General,
                                    alloc);
        if (sig_a == nullptr)
            sig_a =
                alloc.new_object<RGBSpectrum>(*RGBColorSpace::sRGB,
                                              RGB(.0011f, .0024f, .014f));
    }
    if (sig_s == nullptr) {
        sig_s = dict.GetOneSpectrum("sigma_s", nullptr, SpectrumType::General,
                                    alloc);
        if (sig_s == nullptr)
            sig_s =
                alloc.new_object<RGBSpectrum>(*RGBColorSpace::sRGB,
                                              RGB(2.55f, 3.21f, 3.77f));
    }

    Float scale = dict.GetOneFloat("scale", 1.f);
    if (scale != 1) {
        sig_a = alloc.new_object<ScaledSpectrum>(scale, sig_a);
        sig_s = alloc.new_object<ScaledSpectrum>(scale, sig_s);
    }

    Float g = dict.GetOneFloat("g", 0.0f);

    return std::make_unique<HomogeneousMedium>(sig_a, sig_s, g);
}

SampledSpectrum HomogeneousMedium::Tr(const Ray &ray, Float tMax,
                                      const SampledWavelengths &lambda,
                                      Sampler &sampler) const {
    ProfilerScope _(ProfilePhase::MediumTr);
    SampledSpectrum sigma_t = sigma_a.Sample(lambda) + sigma_s.Sample(lambda);
    return Exp(-sigma_t * std::min(tMax * Length(ray.d), MaxFloat));
}

SampledSpectrum HomogeneousMedium::Sample(const Ray &ray, Float tMax, Sampler &sampler,
                                          const SampledWavelengths &lambda,
                                          MemoryArena &arena,
                                          MediumInteraction *mi) const {
    ProfilerScope _(ProfilePhase::MediumSample);

    // Sample a channel and distance along the ray
    SampledSpectrum sigma_t = sigma_a.Sample(lambda) + sigma_s.Sample(lambda);
    int channel = sampler.GetDiscrete1D(NSpectrumSamples);
    if (sigma_t[channel] == 0) return SampledSpectrum(1);
    Float dist = -std::log(1 - sampler.Get1D()) / sigma_t[channel];
    Float t = std::min(dist / Length(ray.d), tMax);
    bool sampledMedium = t < tMax;
    if (sampledMedium)
        *mi = MediumInteraction(ray(t), -ray.d, ray.time, this,
                                arena.Alloc<HenyeyGreenstein>(g));

    // Compute the transmittance and sampling density
    SampledSpectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) * Length(ray.d));

    // Return weighting factor for scattering from homogeneous medium
    SampledSpectrum density = sampledMedium ? (sigma_t * Tr) : Tr;
    Float pdf = density.Average();
    if (pdf == 0) {
        CHECK(!Tr);
        pdf = 1;
    }
    return sampledMedium ? (Tr * sigma_s.Sample(lambda) / pdf) : (Tr / pdf);
}

std::string HomogeneousMedium::ToString() const {
    return StringPrintf("[ Homogeneous medium sigma_a: %s sigma_s: %s g: %f ]",
                        sigma_a, sigma_s, g);
}


STAT_RATIO("Media/Grid steps per Tr() call", nTrSteps, nTrCalls);
STAT_RATIO("Media/Grid steps per Sample() call", nSampleSteps, nSampleCalls);
STAT_MEMORY_COUNTER("Memory/Volume density grid", densityBytes);
STAT_MEMORY_COUNTER("Memory/Volume density octree", densityOctreeBytes);

// GridDensityMedium Method Definitions
GridDensityMedium::GridDensityMedium(SpectrumHandle sigma_a, SpectrumHandle sigma_s, Float g,
                                     int nx, int ny, int nz, const Transform &worldFromMedium,
                                     std::vector<Float> d, Allocator alloc)
    : sigma_a_spec(sigma_a),
      sigma_s_spec(sigma_s),
      g(g),
      nx(nx),
      ny(ny),
      nz(nz),
      mediumFromWorld(Inverse(worldFromMedium)),
      worldFromMedium(worldFromMedium),
      density(d.begin(), d.end()) {
    CHECK_EQ(nx * ny * nz, density.size());
    densityBytes += density.size() * sizeof(Float);

    // Create densityOctree. For starters, make the full thing. (Probably
    // not the best approach).
    const int maxDepth = 6;
    Bounds3f bounds(Point3f(0, 0, 0), Point3f(1, 1, 1));
    buildOctree(&densityOctree, alloc, bounds, maxDepth);
//CO    return;///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Want world-space bounds, but not including the rotation, so that the bbox
    // doesn't expand
    Vector3f T;
    SquareMatrix<4> R, S;
    worldFromMedium.Decompose(&T, &R, &S);
    Bounds3f worldBounds = Transform(S)(bounds);
    Float SE = worldBounds.SurfaceArea();
    Float sigma_t = sigma_a_spec.MaxValue() + sigma_s_spec.MaxValue();
    simplifyOctree(&densityOctree, worldBounds, SE, sigma_t);

    // FIXME
    // densityOctreeBytes += octreeArena.BytesAllocated();
}

void GridDensityMedium::simplifyOctree(OctreeNode *node, const Bounds3f &bounds,
                                       Float SE, Float sigma_t) {
    if (node->children == nullptr)
        // leaf
        return;

    // Equation (14) from Yue et al: Toward Optimal Space Partitioning for
    // Unbiased, Adaptive Free Path Sampling.
    Float kParent = node->maxDensity * sigma_t;
    Float Nparent = 4 * kParent * bounds.Volume() / SE;

    Float kChildSum = 0;
    for (int i = 0; i < 8; ++i)
        kChildSum += node->child(i)->maxDensity * sigma_t;
    Float childVolume = bounds.Volume() / 8;
    // |P_k| = bounds.SurfaceArea() / 2, but then there is a factor of 2 in
    // equation (14)...
    Float Nsplit = (4 * childVolume * kChildSum + bounds.SurfaceArea()) / SE;

    //LOG(INFO) << bounds << ": Nparent " << Nparent << " (kparent) " << kParent << ", NSplit " << Nsplit;
    if (1.1 * Nparent < Nsplit) {
        //LOG(INFO) << " -> SIMPLIFYING";
        node->children = nullptr;
    } else {
        for (int i = 0; i < 8; ++i)
            simplifyOctree(node->child(i), OctreeChildBounds(bounds, i), SE,
                           sigma_t);
    }
}

void GridDensityMedium::buildOctree(OctreeNode *node, Allocator alloc,
                                    const Bounds3f &bounds, int depth) {
    node->bounds = bounds;
    if (depth == 0) {
        // leaf
        Point3f ps[2] = { Point3f(bounds.pMin.x * nx - .5f,
                                  bounds.pMin.y * ny - .5f,
                                  bounds.pMin.z * nz - .5f),
                          Point3f(bounds.pMax.x * nx - .5f,
                                  bounds.pMax.y * ny - .5f,
                                  bounds.pMax.z * nz - .5f) };
        Point3i pi[2] = { Max(Point3i(Floor(ps[0])), Point3i(0, 0, 0)),
                          Min(Point3i(Floor(ps[1])) + Vector3i(1, 1, 1),
                              Point3i(nx - 1, ny - 1, nz - 1)) };

        Float minDensity = Infinity, maxDensity = 0;
        for (int z = pi[0].z; z <= pi[1].z; ++z)
            for (int y = pi[0].y; y <= pi[1].y; ++y)
                for (int x = pi[0].x; x <= pi[1].x; ++x) {
                    Float d = D(Point3i(x, y, z));
                    minDensity = std::min(minDensity, d);
                    maxDensity = std::max(maxDensity, d);
                }

        node->minDensity = minDensity;
        node->maxDensity = maxDensity;
        return;
    }

    node->children = alloc.new_object<pstd::array<OctreeNode *, 8>>();
    for (int i = 0; i < 8; ++i) {
        node->child(i) = alloc.new_object<OctreeNode>();
        buildOctree(node->child(i), alloc, OctreeChildBounds(bounds, i), depth - 1);
    }

    node->minDensity = node->child(0)->minDensity;
    node->maxDensity = node->child(0)->maxDensity;
    for (int i = 1; i < 8; ++i) {
        node->minDensity = std::min(node->minDensity,
                                    node->child(i)->minDensity);
        node->maxDensity = std::max(node->maxDensity,
                                    node->child(i)->maxDensity);
    }
}

std::unique_ptr<GridDensityMedium> GridDensityMedium::Create(
        const ParameterDictionary &dict,
        const Transform &worldFromMedium, Allocator alloc) {
    SpectrumHandle sig_a = nullptr, sig_s = nullptr;
    std::string preset = dict.GetOneString("preset", "");
    if (!preset.empty()) {
        if (!GetMediumScatteringProperties(preset, &sig_a, &sig_s, alloc))
            Warning("Material preset \"%s\" not found.", preset);
    }

    if (sig_a == nullptr) {
        sig_a = dict.GetOneSpectrum("sigma_a", nullptr, SpectrumType::General, alloc);
        if (sig_a == nullptr)
            sig_a = alloc.new_object<RGBSpectrum>(*RGBColorSpace::sRGB,
                                                  RGB(.0011f, .0024f, .014f));
    }
    if (sig_s == nullptr) {
        sig_s = dict.GetOneSpectrum("sigma_s", nullptr, SpectrumType::General, alloc);
        if (sig_s == nullptr)
            sig_s = alloc.new_object<RGBSpectrum>(*RGBColorSpace::sRGB,
                                                  RGB(2.55f, 3.21f, 3.77f));
    }

    Float scale = dict.GetOneFloat("scale", 1.f);
    if (scale != 1) {
        sig_a = alloc.new_object<ScaledSpectrum>(scale, sig_a);
        sig_s = alloc.new_object<ScaledSpectrum>(scale, sig_s);
    }

    Float g = dict.GetOneFloat("g", 0.0f);

    std::vector<Float> density = dict.GetFloatArray("density");
    if (density.empty()) {
        Error("No \"density\" values provided for heterogeneous medium?");
        return nullptr;
    }
    int nx = dict.GetOneInt("nx", 1);
    int ny = dict.GetOneInt("ny", 1);
    int nz = dict.GetOneInt("nz", 1);
    Point3f p0 = dict.GetOnePoint3f("p0", Point3f(0.f, 0.f, 0.f));
    Point3f p1 = dict.GetOnePoint3f("p1", Point3f(1.f, 1.f, 1.f));
    if (density.size() != nx * ny * nz) {
        Error(
              "GridDensityMedium has %d density values; expected nx*ny*nz = "
              "%d",
              (int)density.size(), nx * ny * nz);
        return nullptr;
    }

    Transform MediumFromData = Translate(Vector3f(p0)) *
        Scale(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
    return std::make_unique<GridDensityMedium>(
        sig_a, sig_s, g, nx, ny, nz, worldFromMedium * MediumFromData, std::move(density),
        alloc);
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

SampledSpectrum GridDensityMedium::Sample(const Ray &rWorld, Float raytMax, Sampler &sampler,
                                          const SampledWavelengths &lambda,
                                          MemoryArena &arena,
                                          MediumInteraction *mi) const {
    ProfilerScope _(ProfilePhase::MediumSample);
    ++nSampleCalls;

    raytMax *= Length(rWorld.d);
    Ray ray = mediumFromWorld(Ray(rWorld.o, Normalize(rWorld.d)), &raytMax);
    // Compute $[\tmin, \tmax]$ interval of _ray_'s overlap with medium bounds
    const Bounds3f b(Point3f(0, 0, 0), Point3f(1, 1, 1));
    Float tMin, tMax;
    if (!b.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax)) return SampledSpectrum(1.f);

    // Run delta-tracking iterations to sample a medium interaction
    bool foundInteraction = false;
#if 0
    // WHY IS THIS SO MUCH NOISIER? SHOULDNT IT BASICALLY BE EQUIVALENT?
    SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
    SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);
    SampledSpectrum sigma_t = sigma_a + sigma_s;

    SampledSpectrum w(1.f);
    TraverseOctree(&densityOctree, ray.o, ray.d, [&](const OctreeNode &node, Float t, Float t1) {
//CO            w = SampledSpectrum(1.f);
            if (node.maxDensity == 0)
                // Empty--skip it!
                return OctreeTraversal::Continue;

            CHECK(!std::isnan(t));
            CHECK(!std::isnan(t1));
            Float sigma_bar = node.maxDensity * sigma_t.MaxComponentValue();

            CHECK_LT(t0, tMax); // this should be the case now

            while (true) {
                ++nSampleSteps;
                t += -std::log(1 - sampler.Get1D()) / sigma_bar;

                if (t >= t1)
                    // exited this cell w/o a scattering event
                    return OctreeTraversal::Continue;

                if (t >= tMax)
                    // Nothing before the geom intersection; get out of here
                    return OctreeTraversal::Abort;

//CO                CHECK(Inside(ray(t), node.bounds)) << t << ", pos " << ray(t) << ", bounds " << node.bounds;
//CO                if (!Inside(ray(t), node.bounds))
//CO                    LOG(WARNING) << "Outside node bounds, t " << t << ", pos " << ray(t) << ", bounds " << node.bounds;
                Float d = Density(ray(t));
                CHECK_LE(d, 1.0001 * node.maxDensity) << t << ", pos " << ray(t) <<
                    ", t0 " << t0 << ", t1 " << t1 << ", tmax " << tMax;
                SampledSpectrum sigma_n = sigma_bar - d * sigma_t;
//CO                CHECK_GE(sigma_n.MinComponentValue(), -1e-5) << sigma_n << ", d " << d <<
//CO                    ", max density " << node.maxDensity << ", sigma_a " << sigma_a <<
//CO                    ", sigma_s " << sigma_s;

                // Slightly confusing: sigma_{a,s} aren't scaled by d, but
                // sigma_n is.
                Float Ps = (sigma_s * d * w).Average();
                Float Pn = (sigma_n * w).Average();
                Float Psum = Ps + Pn;
                // TODO: multiply the sample value by this instead, or use SampleDiscrete for that matter...
                Ps /= Psum;
                Pn /= Psum;

                if (sampler.Get1D() < Ps) {
                    // Scatter
                    w *= sigma_s * (d / (sigma_bar * Ps));
                    CHECK(w.MaxComponentValue() != Infinity);
                    CHECK(!w.HasNaNs());
                    // Populate _mi_ with medium interaction information and return
                    PhaseFunction *phase = arena.Alloc<HenyeyGreenstein>(g);
                    *mi = MediumInteraction(rWorld(t), -rWorld.d, rWorld.time, this,
                                            phase);
                    foundInteraction = true;
                    return OctreeTraversal::Abort;
                } else {
                    // null collision; keep going
                    w *= sigma_n / (sigma_bar * Pn);
                    CHECK(w.MaxComponentValue() != Infinity);
                    CHECK(!w.HasNaNs());
                }
            }
        });
                    CHECK(w.MaxComponentValue() != Infinity);
                    CHECK(!w.HasNaNs());
    return w; // TODO: can nuke foundInteractino???
    return foundInteraction ? w : SampledSpectrum(1.f); // TODO: safe to always return w?
#else
    // For now...
    Float sigma_t = sigma_a_spec.MaxValue() + sigma_s_spec.MaxValue();

    TraverseOctree(&densityOctree, ray.o, ray.d, raytMax, [&](const OctreeNode &node, Float t, Float t1) {
            if (node.maxDensity == 0)
                // Empty--skip it!
                return OctreeTraversal::Continue;

            DCHECK_RARE(1e-5, Density(ray((t + t1)/2)) > node.maxDensity);
            while (true) {
                ++nSampleSteps;
                t += -std::log(1 - sampler.Get1D()) / (sigma_t * node.maxDensity);

                if (t >= t1)
                    // exited this cell w/o a scattering event
                    return OctreeTraversal::Continue;

                if (t >= tMax)
                    // Nothing before the geom intersection; get out of here
                    return OctreeTraversal::Abort;

                if (Density(ray(t)) > sampler.Get1D() * node.maxDensity) {
                    // Populate _mi_ with medium interaction information and return
                    PhaseFunction *phase = arena.Alloc<HenyeyGreenstein>(g);
                    *mi = MediumInteraction(rWorld(t), -rWorld.d, rWorld.time, this,
                                            phase);
                    foundInteraction = true;
                    return OctreeTraversal::Abort;
                }
            }
        });
    return foundInteraction ? sigma_s_spec.Sample(lambda) / sigma_t :
        SampledSpectrum(1.f);
#endif
}

SampledSpectrum GridDensityMedium::Tr(const Ray &rWorld, Float raytMax,
                                      const SampledWavelengths &lambda,
                                      Sampler &sampler) const {
    ProfilerScope _(ProfilePhase::MediumTr);
    ++nTrCalls;

    raytMax *= Length(rWorld.d);
    Ray ray = mediumFromWorld(Ray(rWorld.o, Normalize(rWorld.d)), &raytMax);
    // Compute $[\tmin, \tmax]$ interval of _ray_'s overlap with medium bounds
    const Bounds3f b(Point3f(0, 0, 0), Point3f(1, 1, 1));
    Float tMin, tMax;
    if (!b.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax)) return SampledSpectrum(1.f);

    SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
    SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);
    SampledSpectrum sigma_t = sigma_a + sigma_s;

    // Perform ratio tracking to estimate the transmittance value
    SampledSpectrum Tr(1.f);
    TraverseOctree(&densityOctree, ray.o, ray.d, raytMax, [&](const OctreeNode &node, Float t, Float t1) {
            if (node.maxDensity == 0)
                // Empty--skip it!
                return OctreeTraversal::Continue;

            DCHECK_GE(t1, 0);

            CHECK_GE(t, .999 * tMin);

            // Residual tracking. First, account for the constant part.
            if (node.minDensity > 0) {
                Float dt = std::min(t1, tMax) - t;
                Tr *= Exp(-dt * node.minDensity * sigma_t);
            CHECK(Tr.MaxComponentValue() != Infinity);
            CHECK(!Tr.HasNaNs());
            }

            // Now do ratio tracking through the residual volume.
            Float sigma_bar = (node.maxDensity - node.minDensity) *
                sigma_t.MaxComponentValue();
            DCHECK_GE(sigma_bar, 0);
            if (sigma_bar == 0)
                // There's no residual; go on to the next octree node.
                return OctreeTraversal::Continue;

            while (true) {
                ++nTrSteps;
                t += -std::log(1 - sampler.Get1D()) / sigma_bar;
                if (t >= t1)
                    // exited node; keep going
                    return OctreeTraversal::Continue;

                if (t >= tMax)
                    // past hit point. stop
                    return OctreeTraversal::Abort;

                Float density = Density(ray(t)) - node.minDensity;
                CHECK_RARE(1e-9, density < 0);
                density = std::max<Float>(density, 0);

                // FIXME: if sigma_bar isn't a majorant, then is this clamp wrong???
                Tr *= 1 - Clamp(density * sigma_t / sigma_bar, 0, 1);

            CHECK(Tr.MaxComponentValue() != Infinity);
            CHECK(!Tr.HasNaNs());
                Float Tr_max = Tr.MaxComponentValue();
                if (Tr_max < 1) {
                    Float q = 1 - Tr_max;
                    if (sampler.Get1D() < q) {
                        Tr = SampledSpectrum(0.f);
                        return OctreeTraversal::Abort;
                    }
                    Tr /= 1 - q;
            CHECK(Tr.MaxComponentValue() != Infinity);
            CHECK(!Tr.HasNaNs());
                }
            }
        });

            CHECK(Tr.MaxComponentValue() != Infinity);
            CHECK(!Tr.HasNaNs());
    return Tr;
}

std::string GridDensityMedium::ToString() const {
    Float maxDensity = densityOctree.maxDensity;
    return StringPrintf("[ GridDensityMedium sigma_a: %s sigma_s: %s "
                        " nx: %d ny: %d nz: %d mediumFromWorld: %s root maxDensity: %f ]",
                        sigma_a_spec, sigma_s_spec, nx, ny, nz, mediumFromWorld,
                        maxDensity);
}

}  // namespace pbrt
