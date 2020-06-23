// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

// media.cpp*
#include <pbrt/media.h>

#include <pbrt/interaction.h>
#include <pbrt/paramdict.h>
#include <pbrt/samplers.h>
#include <pbrt/textures.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/scattering.h>
#include <pbrt/util/stats.h>

#include <algorithm>
#include <cmath>

namespace pbrt {

std::string MediumInterface::ToString() const {
    return StringPrintf("[ MediumInterface inside: %s outside: %s ]",
                        inside ? inside.ToString().c_str() : "(nullptr)",
                        outside ? outside.ToString().c_str() : "(nullptr)");
}

std::string PhaseFunctionHandle::ToString() const {
    if (ptr() == nullptr)
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return ApplyCPU<std::string>(ts);
}

std::string MediumSample::ToString() const {
    return StringPrintf("[ MediumSample intr: %s t: %f Tmaj: %s ]",
                        intr, t, Tmaj);
}

// Media Definitions

// HenyeyGreenstein Method Definitions
std::string HenyeyGreensteinPhaseFunction::ToString() const {
    return StringPrintf("[ HenyeyGreensteinPhaseFunction g: %f ]", g);
}

// Media Local Definitions
struct MeasuredSS {
    const char *name;
    RGB sigma_prime_s, sigma_a;  // mm^-1
};

bool GetMediumScatteringProperties(const std::string &name, SpectrumHandle *sigma_a,
                                   SpectrumHandle *sigma_s, Allocator alloc) {
    static MeasuredSS SubsurfaceParameterTable[] = {
        // From "A Practical Model for Subsurface Light Transport"
        // Jensen, Marschner, Levoy, Hanrahan
        // Proc SIGGRAPH 2001
        // clang-format off
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
        {"Pacific Ocean Surface Water", RGB(0.0001764, 0.00032095, 0.00019617), RGB(0.031845, 0.031324, 0.030147)}
        // clang-format on
    };

    for (MeasuredSS &mss : SubsurfaceParameterTable) {
        if (name == mss.name) {
            *sigma_a = alloc.new_object<RGBSpectrum>(*RGBColorSpace::sRGB, mss.sigma_a);
            *sigma_s =
                alloc.new_object<RGBSpectrum>(*RGBColorSpace::sRGB, mss.sigma_prime_s);
            return true;
        }
    }
    return false;
}

bool MediumHandle::IsEmissive() const {
    auto is = [&](auto ptr) { return ptr->IsEmissive(); };
    return ApplyCPU<bool>(is);
}

std::string MediumHandle::ToString() const {
    if (ptr() == nullptr)
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return ApplyCPU<std::string>(ts);
}

// HomogeneousMedium Method Definitions
MediumSample HomogeneousMedium::Sample_Tmaj(const Ray &ray, Float tMax, Float u,
                                              const SampledWavelengths &lambda,
                                              ScratchBuffer *scratchBuffer) const {
    // So t corresponds to distance...
    tMax *= Length(ray.d);
    Ray rayp(ray.o, Normalize(ray.d));

    SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
    SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);
    SampledSpectrum sigma_t = sigma_a + sigma_s;
    SampledSpectrum sigma_maj = sigma_t;

    Float t = SampleExponential(u, sigma_maj[0]);
    if (t >= tMax)
        return MediumSample(FastExp(-tMax * sigma_maj));

    SampledSpectrum Tmaj = FastExp(-t * sigma_maj);

    SampledSpectrum Le = Le_spec.Sample(lambda);
    MediumInteraction intr(rayp(t), -rayp.d, ray.time, sigma_a, sigma_s, sigma_maj, Le,
                           this, &phase);
    return MediumSample(intr, t, Tmaj);
}

HomogeneousMedium *HomogeneousMedium::Create(const ParameterDictionary &parameters,
                                             const FileLoc *loc, Allocator alloc) {
    SpectrumHandle sig_a = nullptr, sig_s = nullptr;
    std::string preset = parameters.GetOneString("preset", "");
    if (!preset.empty()) {
        if (!GetMediumScatteringProperties(preset, &sig_a, &sig_s, alloc))
            Warning(loc, "Material preset \"%s\" not found.", preset);
    }
    if (sig_a == nullptr) {
        sig_a = parameters.GetOneSpectrum("sigma_a", nullptr, SpectrumType::General, alloc);
        if (sig_a == nullptr)
            sig_a = alloc.new_object<RGBSpectrum>(*RGBColorSpace::sRGB,
                                                  RGB(.0011f, .0024f, .014f));
    }
    if (sig_s == nullptr) {
        sig_s = parameters.GetOneSpectrum("sigma_s", nullptr, SpectrumType::General, alloc);
        if (sig_s == nullptr)
            sig_s = alloc.new_object<RGBSpectrum>(*RGBColorSpace::sRGB,
                                                  RGB(2.55f, 3.21f, 3.77f));
    }

    SpectrumHandle Le = parameters.GetOneSpectrum("Le", nullptr, SpectrumType::General,
                                            alloc);
    if (Le == nullptr)
        Le = alloc.new_object<ConstantSpectrum>(0.f);

    Float scale = parameters.GetOneFloat("scale", 1.f);
    if (scale != 1) {
        sig_a = alloc.new_object<ScaledSpectrum>(scale, sig_a);
        sig_s = alloc.new_object<ScaledSpectrum>(scale, sig_s);
    }

    Float g = parameters.GetOneFloat("g", 0.0f);

    return alloc.new_object<HomogeneousMedium>(sig_a, sig_s, Le, g, alloc);
}

std::string HomogeneousMedium::ToString() const {
    return StringPrintf("[ Homogeneous medium sigma_a: %s sigma_s: %s g: %f ]",
                        sigma_a_spec, sigma_s_spec, g);
}

STAT_MEMORY_COUNTER("Memory/Volume grids", volumeGridBytes);
STAT_MEMORY_COUNTER("Memory/Volume density octree", densityOctreeBytes);

// GridDensityMedium Method Definitions
GridDensityMedium::GridDensityMedium(SpectrumHandle sigma_a, SpectrumHandle sigma_s,
                                     SpectrumHandle Le, Float g,
                                     const Transform &worldFromMedium,
                                     pstd::optional<SampledGrid<Float>> dgrid,
                                     pstd::optional<SampledGrid<RGB>> rgbgrid,
                                     const RGBColorSpace *colorSpace,
                                     SampledGrid<Float> Legrid, Allocator alloc)
    : sigma_a_spec(sigma_a, alloc),
      sigma_s_spec(sigma_s, alloc),
      Le_spec(Le, alloc),
      phase(g),
      g(g),
      mediumFromWorld(Inverse(worldFromMedium)),
      worldFromMedium(worldFromMedium),
      densityGrid(std::move(dgrid)),
      rgbDensityGrid(std::move(rgbgrid)),
      colorSpace(colorSpace),
      LeScaleGrid(std::move(Legrid)),
      treeBufferResource(256 * 1024, alloc.resource()) {
    volumeGridBytes += LeScaleGrid.BytesAllocated();
    volumeGridBytes += densityGrid ? densityGrid->BytesAllocated() :
        rgbDensityGrid->BytesAllocated();

    // Create densityOctree. For starters, make the full thing. (Probably
    // not the best approach).
    const int maxDepth = 6;
    Bounds3f bounds(Point3f(0, 0, 0), Point3f(1, 1, 1));
    Allocator treeAllocator(&treeBufferResource);
    buildOctree(&densityOctree, treeAllocator, bounds, maxDepth);

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

void GridDensityMedium::simplifyOctree(OctreeNode *node, const Bounds3f &bounds, Float SE,
                                       Float sigma_t) {
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

    // LOG(INFO) << bounds << ": Nparent " << Nparent << " (kparent) " <<
    // kParent << ", NSplit " << Nsplit;
    if (1.1 * Nparent < Nsplit) {
        // LOG(INFO) << " -> SIMPLIFYING";
        node->children = nullptr;
    } else {
        for (int i = 0; i < 8; ++i)
            simplifyOctree(node->child(i), OctreeChildBounds(bounds, i), SE, sigma_t);
    }
}

void GridDensityMedium::buildOctree(OctreeNode *node, Allocator alloc,
                                    const Bounds3f &bounds, int depth) {
    if (depth == 0) {
        // leaf
        if (densityGrid)
            node->maxDensity = densityGrid->MaximumValue(bounds);
        else {
            int nx = rgbDensityGrid->xSize();
            int ny = rgbDensityGrid->ySize();
            int nz = rgbDensityGrid->zSize();
            Point3f ps[2] = {Point3f(bounds.pMin.x * nx - .5f, bounds.pMin.y * ny - .5f,
                                     bounds.pMin.z * nz - .5f),
                             Point3f(bounds.pMax.x * nx - .5f, bounds.pMax.y * ny - .5f,
                                     bounds.pMax.z * nz - .5f)};
            Point3i pi[2] = {Max(Point3i(Floor(ps[0])), Point3i(0, 0, 0)),
                             Min(Point3i(Floor(ps[1])) + Vector3i(1, 1, 1),
                                 Point3i(nx - 1, ny - 1, nz - 1))};

            std::vector<SampledWavelengths> lambdas;
            for (Float u : Stratified1D(16))
                lambdas.push_back(SampledWavelengths::SampleImportance(u));

            node->maxDensity = 0;
            for (int z = pi[0].z; z <= pi[1].z; ++z)
                for (int y = pi[0].y; y <= pi[1].y; ++y)
                    for (int x = pi[0].x; x <= pi[1].x; ++x) {
                        RGB rgb = rgbDensityGrid->Lookup(Point3i(x, y, z));

                        Float maxComponent = std::max({rgb.r, rgb.g, rgb.b});
                        if (maxComponent == 0)
                            continue;

                        RGBSpectrum spec(*colorSpace, rgb);
                        for (const SampledWavelengths &lambda : lambdas) {
                            SampledSpectrum s = spec.Sample(lambda);
                            node->maxDensity =
                                std::max(node->maxDensity, s.MaxComponentValue());
                        }
                    }

            node->maxDensity *= 1.025f;
        }

        CHECK_GE(node->maxDensity, 0);
        return;
    }

    node->children = alloc.new_object<pstd::array<OctreeNode *, 8>>();
    for (int i = 0; i < 8; ++i) {
        node->child(i) = alloc.new_object<OctreeNode>();
        buildOctree(node->child(i), alloc, OctreeChildBounds(bounds, i), depth - 1);
    }

    node->maxDensity = node->child(0)->maxDensity;
    for (int i = 1; i < 8; ++i)
        node->maxDensity = std::max(node->maxDensity, node->child(i)->maxDensity);
}

MediumSample GridDensityMedium::Sample_Tmaj(const Ray &rWorld, Float raytMax, Float u,
                                              const SampledWavelengths &lambda,
                                              ScratchBuffer *scratchBuffer) const {
    raytMax *= Length(rWorld.d);
    Ray ray = mediumFromWorld(Ray(rWorld.o, Normalize(rWorld.d)), &raytMax);
    // Compute $[\tmin, \tmax]$ interval of _ray_'s overlap with medium bounds
    const Bounds3f b(Point3f(0, 0, 0), Point3f(1, 1, 1));
    Float tMin, tMax;
    if (!b.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax))
        return {};

    CHECK_LE(tMax, raytMax);

    SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
    SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);
    SampledSpectrum sigma_t = sigma_a + sigma_s;

    MediumSample mediumSample;

    TraverseOctree(
        &densityOctree, ray.o, ray.d, raytMax,
        [&](const OctreeNode &node, Float t0, Float t1) {
            if (node.maxDensity == 0)
                // Empty--skip it!
                return OctreeTraversal::Continue;

            SampledSpectrum sigma_maj(sigma_t * node.maxDensity);

            // At what u value do we hit the the cell exit point?
            Float uEnd = InvertExponentialSample(t1 - t0, sigma_maj[0]);
            if (u >= uEnd) {
                // exit this cell w/o a scattering event
                u = (u - uEnd) / (1 - uEnd);  // remap to [0,1)
                return OctreeTraversal::Continue;
            }

            Float t = t0 + SampleExponential(u, sigma_maj[0]);
            CHECK_RARE(1e-5, t > t1);

            if (t >= tMax) {
                // Nothing before the geom intersection; get out of here
                mediumSample = MediumSample(FastExp(-sigma_maj * (tMax - t0)));
                return OctreeTraversal::Abort;
            }

            // Scattering event (of some sort)
            Point3f p = ray(t);
            SampledSpectrum Tmaj = FastExp(-sigma_maj * (t - t0));

            if (densityGrid) {
                Float density = densityGrid->Lookup(p);
                sigma_a *= density;
                sigma_s *= density;
            } else {
                RGB density = rgbDensityGrid->Lookup(p);
                CHECK_LE(density.r, node.maxDensity);
                CHECK_LE(density.g, node.maxDensity);
                CHECK_LE(density.b, node.maxDensity);
                SampledSpectrum spec =
                    RGBSpectrum(*RGBColorSpace::sRGB, density).Sample(lambda);
                sigma_a *= spec;
                sigma_s *= spec;
            }

            MediumInteraction intr(worldFromMedium(p), -Normalize(rWorld.d), rWorld.time,
                                   sigma_a, sigma_s, sigma_maj, Le(p, lambda), this,
                                   &phase);
            mediumSample = MediumSample(intr, t, Tmaj);
            return OctreeTraversal::Abort;
        });

    return mediumSample;
}

GridDensityMedium *GridDensityMedium::Create(const ParameterDictionary &parameters,
                                             const Transform &worldFromMedium,
                                             const FileLoc *loc, Allocator alloc) {
    SpectrumHandle sig_a = nullptr, sig_s = nullptr;
    std::string preset = parameters.GetOneString("preset", "");
    if (!preset.empty()) {
        if (!GetMediumScatteringProperties(preset, &sig_a, &sig_s, alloc))
            Warning(loc, "Material preset \"%s\" not found.", preset);
    }

    if (sig_a == nullptr) {
        sig_a = parameters.GetOneSpectrum("sigma_a", nullptr, SpectrumType::General, alloc);
        if (sig_a == nullptr)
            sig_a = alloc.new_object<RGBSpectrum>(*RGBColorSpace::sRGB,
                                                  RGB(.0011f, .0024f, .014f));
    }
    if (sig_s == nullptr) {
        sig_s = parameters.GetOneSpectrum("sigma_s", nullptr, SpectrumType::General, alloc);
        if (sig_s == nullptr)
            sig_s = alloc.new_object<RGBSpectrum>(*RGBColorSpace::sRGB,
                                                  RGB(2.55f, 3.21f, 3.77f));
    }

    Float scale = parameters.GetOneFloat("scale", 1.f);
    if (scale != 1) {
        sig_a = alloc.new_object<ScaledSpectrum>(scale, sig_a);
        sig_s = alloc.new_object<ScaledSpectrum>(scale, sig_s);
    }

    Float g = parameters.GetOneFloat("g", 0.0f);

    std::vector<Float> density = parameters.GetFloatArray("density");
    std::vector<RGB> rgbDensity = parameters.GetRGBArray("density");
    if (density.empty() && rgbDensity.empty())
        ErrorExit(loc, "No \"density\" values provided for heterogeneous medium.");
    if (!density.empty() && !rgbDensity.empty())
        ErrorExit(loc, "Both \"float\" and \"rgb\" \"density\" values were provided.");

    int nx = parameters.GetOneInt("nx", 1);
    int ny = parameters.GetOneInt("ny", 1);
    int nz = parameters.GetOneInt("nz", 1);
    Point3f p0 = parameters.GetOnePoint3f("p0", Point3f(0.f, 0.f, 0.f));
    Point3f p1 = parameters.GetOnePoint3f("p1", Point3f(1.f, 1.f, 1.f));
    size_t nDensity = !density.empty() ? density.size() : rgbDensity.size();
    if (nDensity != nx * ny * nz)
        ErrorExit(loc, "GridDensityMedium has %d density values; expected nx*ny*nz = %d",
                  nDensity, nx * ny * nz);

    const RGBColorSpace *colorSpace = parameters.ColorSpace();

    pstd::optional<SampledGrid<Float>> densityGrid;
    pstd::optional<SampledGrid<RGB>> rgbDensityGrid;
    if (density.size())
        densityGrid = SampledGrid<Float>(density, nx, ny, nz, alloc);
    else
        rgbDensityGrid = SampledGrid<RGB>(rgbDensity, nx, ny, nz, alloc);

    SpectrumHandle Le = parameters.GetOneSpectrum("Le", nullptr, SpectrumType::General,
                                            alloc);
    if (Le == nullptr)
        Le = alloc.new_object<ConstantSpectrum>(0.f);

    SampledGrid<Float> LeGrid(alloc);
    std::vector<Float> LeScale = parameters.GetFloatArray("Lescale");
    if (LeScale.empty())
        LeGrid = SampledGrid<Float>({1.f}, 1, 1, 1, alloc);
    else {
        if (LeScale.size() != nx * ny * nz)
            ErrorExit("Expected %d x %d %d = %d values for \"Lescale\" but were "
                      "given %d.", nx, ny, nz, nx * ny * nz, LeScale.size());
        LeGrid = SampledGrid<Float>(LeScale, nx, ny, nz, alloc);
    }

    Transform MediumFromData =
        Translate(Vector3f(p0)) * Scale(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
    return alloc.new_object<GridDensityMedium>(sig_a, sig_s, Le, g,
                                               worldFromMedium * MediumFromData,
                                               std::move(densityGrid),
                                               std::move(rgbDensityGrid), colorSpace,
                                               std::move(LeGrid), alloc);
}

std::string GridDensityMedium::ToString() const {
    Float maxDensity = densityOctree.maxDensity;
    return StringPrintf("[ GridDensityMedium sigma_a: %s sigma_s: %s mediumFromWorld: %s "
                        "root maxDensity: %f ]",
                        sigma_a_spec, sigma_s_spec, mediumFromWorld, maxDensity);
}

MediumHandle MediumHandle::Create(const std::string &name,
                                  const ParameterDictionary &parameters,
                                  const Transform &worldFromMedium, const FileLoc *loc,
                                  Allocator alloc) {
    MediumHandle m = nullptr;
    if (name == "homogeneous")
        m = HomogeneousMedium::Create(parameters, loc, alloc);
    else if (name == "heterogeneous")
        m = GridDensityMedium::Create(parameters, worldFromMedium, loc, alloc);
    else
        ErrorExit(loc, "%s: medium unknown.", name);

    if (!m)
        ErrorExit(loc, "%s: unable to create medium.", name);

    parameters.ReportUnused();
    return m;
}

}  // namespace pbrt
