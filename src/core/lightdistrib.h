
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

#ifndef PBRT_CORE_LIGHTDISTRIB_H
#define PBRT_CORE_LIGHTDISTRIB_H

#include "pbrt.h"

#include "util/geometry.h"
#include "sampling.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace pbrt {

// LightDistribution defines a general interface for classes that provide
// probability distributions for sampling light sources at a given point in
// space.
class LightDistribution {
  public:
    virtual ~LightDistribution();

    // Given a point |p| in space, this method returns a (hopefully
    // effective) sampling distribution for light sources at that point.
    virtual const Light *Sample(const Point3f &p, Float u, Float *pdf) const = 0;

    // Returns the PDF for sampling the light |light| at the point |p|.
    virtual Float Pdf(const Point3f &p, const Light *light) const = 0;

 protected:
    LightDistribution(const Scene &scene) : scene(scene) {}

    const Scene &scene;
};

// FixedLightDistribution is a LightDistribution specialization for
// implementations that use the sampling probabilities at all points in the
// scene. Those just need to initialize a corresponding Distribution1D in
// their constructor and the implementation here takes care of the rest.
class FixedLightDistribution : public LightDistribution {
 public:
    virtual ~FixedLightDistribution();

    // If the caller knows they have a FixedLightDistribution, they can
    // sample and compute the PDF without providing a point.
    const Light *Sample(Float u, Float *pdf) const;
    Float Pdf(const Light *light) const;

    const Light *Sample(const Point3f &p, Float u, Float *pdf) const final {
        return Sample(u, pdf);
    }
    Float Pdf(const Point3f &p, const Light *light) const final {
        return Pdf(light);
    }

 protected:
    FixedLightDistribution(const Scene &scene);

    std::unique_ptr<Distribution1D> distrib;

 private:
    // Inverse mapping from elements in scene.lights to entries
    // in the |distrib| array.
    std::unordered_map<const Light *, size_t> lightToIndex;
};

std::unique_ptr<LightDistribution> CreateLightSampleDistribution(
    const std::string &name, const Scene &scene);

// The simplest possible implementation of LightDistribution: this uses a
// uniform distribution over all light sources. This approach works well
// for very simple scenes, but is quite ineffective for scenes with more
// than a handful of light sources. (This was the sampling method
// originally used for the PathIntegrator and the VolPathIntegrator in the
// printed book, though without the UniformLightDistribution class.)
class UniformLightDistribution : public FixedLightDistribution {
  public:
    UniformLightDistribution(const Scene &scene);
};

// PowerLightDistribution returns a distribution with sampling probability
// proportional to the total emitted power for each light. (It also ignores
// the provided point |p|.)  This approach works well for scenes where
// there the most powerful lights are also the most important contributors
// to lighting in the scene, but doesn't do well if there are many lights
// and if different lights are relatively important in some areas of the
// scene and unimportant in others. (This was the default sampling method
// used for the BDPT integrator and MLT integrator in the printed book,
// though also without the PowerLightDistribution class.)
class PowerLightDistribution : public FixedLightDistribution {
  public:
    PowerLightDistribution(const Scene &scene);
};

// A spatially-varying light distribution that adjusts the probability of
// sampling a light source based on an estimate of its contribution to a
// region of space.  A fixed voxel grid is imposed over the scene bounds
// and a sampling distribution is computed as needed for each voxel.
class SpatialLightDistribution : public LightDistribution {
  public:
    SpatialLightDistribution(const Scene &scene, int maxVoxels = 64);
    ~SpatialLightDistribution();

    const Light *Sample(const Point3f &p, Float u, Float *pdf) const;
    Float Pdf(const Point3f &p, const Light *light) const;

  private:
    // Returns a sampling distribution to use at the given point |p|.
    const Distribution1D *GetDistribution(const Point3f &p) const;

    // Compute the sampling distribution for the voxel with integer
    // coordiantes given by |pi|.
    Distribution1D *ComputeDistribution(Point3i pi) const;

    int nVoxels[3];

    std::unordered_map<const Light *, size_t> lightToIndex;

    // The hash table is a fixed number of HashEntry structs (where we
    // allocate more than enough entries in the SpatialLightDistribution
    // constructor). During rendering, the table is allocated without
    // locks, using atomic operations. (See the Lookup() method
    // implementation for details.)
    struct HashEntry {
        std::atomic<uint64_t> packedPos;
        std::atomic<Distribution1D *> distribution;
    };
    mutable std::unique_ptr<HashEntry[]> hashTable;
    size_t hashTableSize;
};

}  // namespace pbrt

#endif  // PBRT_CORE_LIGHTDISTRIB_H
