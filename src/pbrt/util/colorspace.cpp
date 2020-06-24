
#include <pbrt/util/colorspace.h>

namespace pbrt {

// TODO: improve variable names now that this works
RGBColorSpace::RGBColorSpace(Point2f r, Point2f g, Point2f b, Point2f w,
                             SpectrumHandle illum,
                             const RGBToSpectrumTable *rgbToSpectrumTable,
                             Allocator alloc)
    : r(r), g(g), b(b), w(w), illuminant(illum, alloc),
      rgbToSpectrumTable(rgbToSpectrumTable) {

    /* Try n:
       http://www.babelcolor.com/index_htm_files/A%20review%20of%20RGB%20color%20spaces.pfd
       p 28:
       1. We know that rgb(1,1,1) goes to the whitepoint in XYZ.
       2. We have the whitepoint in xy coordinates; we have Y=1,
          so can compute the whitepoint's XYZ
       3. We want to find the matrix M s.t.
          [X Y Z]^t = [ M ] [R G B]^t
       4. We know that the X value is proportional to the sum of the Rx, Gx, Bx
          primaries. (More specifically, the sum because we're dotting with
          rgb = 1,1,1.) Similarly for Y and Z. We can express this as
          [ X ] = [ rx gx bx ] [Cr  0  0 ] [ 1 ]
          [ Y ] = [ ry gy by ] [ 0 Cg  0 ] [ 1 ]
          [ Z ] = [ rz gz bz ] [ 0  0 Cb ] [ 1 ]
       5. Now solve for Cr/Cg/Cg: multiply both sides by the inverse of the rx...
          matrix and we've got Cr, Cg, and Cb. Now multiply to get the
          RGB -> XYZ matrix. Invert that and we can go the other way.
     */

    // Compute XYZ coordinates from xyz assuming Y=1
    XYZ R = XYZ::FromxyY(r.x, r.y), G = XYZ::FromxyY(g.x, g.y), B = XYZ::FromxyY(b.x, b.y);
    XYZ W = XYZ::FromxyY(w.x, w.y);

    SquareMatrix<3> M(R.X, G.X, B.X,
                      R.Y, G.Y, B.Y,
                      R.Z, G.Z, B.Z);
    auto Minv = Inverse(M);
    CHECK(Minv);

    XYZ C = *Minv * W;

    // now construct an rgb to xyz that has the effect of scaling the
    // computed rgb values by the whitepoint
    XYZFromRGB = M * SquareMatrix<3>::Diag(C[0], C[1], C[2]);
    auto rgbInv = Inverse(XYZFromRGB);
    CHECK(rgbInv);
    RGBFromXYZ = *rgbInv;
}

RGBSigmoidPolynomial RGBColorSpace::ToRGBCoeffs(const RGB &rgb) const {
    CHECK_RARE(1e-6, rgb.r < 0 || rgb.g < 0 || rgb.b < 0);
    return (*rgbToSpectrumTable)(ClampZero(rgb));
}

// Bradford matrix
const SquareMatrix<3> LMSFromXYZ( 0.8951,  0.2664, -0.1614,
                                 -0.7502,  1.7135,  0.0367,
                                  0.0389, -0.0685,  1.0296);

const SquareMatrix<3> XYZFromLMS = *Inverse(LMSFromXYZ);

SquareMatrix<3> RGBColorSpace::ColorCorrectionMatrixForxy(Float x, Float y) const {
    XYZ srcXYZ = XYZ::FromxyY(x, y), dstXYZ = XYZ::FromxyY(w.x, w.y);

    SquareMatrix<3> LMSFromRGB = LMSFromXYZ * XYZFromRGB;

    // Not actually XYZ...
    XYZ srcLMS = LMSFromXYZ * srcXYZ, dstLMS = LMSFromXYZ * dstXYZ;
    SquareMatrix<3> LMScorrect = SquareMatrix<3>::Diag(dstLMS[0] / srcLMS[0],
                                                       dstLMS[1] / srcLMS[1],
                                                       dstLMS[2] / srcLMS[2]);

    SquareMatrix<3> RGBFromLMS = RGBFromXYZ * XYZFromLMS;

    return RGBFromLMS * LMScorrect * LMSFromRGB;
}

const RGBColorSpace *RGBColorSpace::GetNamed(const std::string &n) {
    std::string name;
    std::transform(n.begin(), n.end(), std::back_inserter(name), ::tolower);
    if (name == "aces2065-1")
        return ACES2065_1;
    else if (name == "rec2020")
        return Rec2020;
    else if (name == "srgb")
        return sRGB;
    else
        return nullptr;
}

const RGBColorSpace *RGBColorSpace::Lookup(Point2f r, Point2f g, Point2f b, Point2f w) {
    auto closeEnough = [](const Point2f &a, const Point2f &b) {
                           return ((a.x == b.x || std::abs((a.x - b.x) / b.x) < 1e-3) &&
                                   (a.y == b.y || std::abs((a.y - b.y) / b.y) < 1e-3));
                       };
    for (const RGBColorSpace *cs : { ACES2065_1, Rec2020, sRGB }) {
        if (closeEnough(r, cs->r) && closeEnough(g, cs->g) && closeEnough(b, cs->b) &&
            closeEnough(w, cs->w))
            return cs;
    }
    return nullptr;
}

const RGBColorSpace *RGBColorSpace::ACES2065_1;
const RGBColorSpace *RGBColorSpace::Rec2020;
const RGBColorSpace *RGBColorSpace::sRGB;

void RGBColorSpace::Init(Allocator alloc) {
    ACES2065_1 = alloc.new_object<RGBColorSpace>(Point2f(.7347, .2653), Point2f(0., 1.),
                                                 Point2f(.0001, -.077), Point2f(.32168, .33767),
                                                 SPDs::IllumACESD60(), RGBToSpectrumTable::ACES2065_1,
                                                 alloc);
    // ITU-R Rec BT.2020
    Rec2020 = alloc.new_object<RGBColorSpace>(Point2f(.708, .292), Point2f(.170, .797),
                                              Point2f(.131, .046), Point2f(.3127, .3290),
                                              SPDs::IllumD65(), RGBToSpectrumTable::Rec2020,
                                              alloc);
    // Rec. ITU-R BT.709.3
    sRGB = alloc.new_object<RGBColorSpace>(Point2f(.64, .33), Point2f(.3, .6),
                                           Point2f(.15, .06), Point2f(.3127, .3290),
                                           SPDs::IllumD65(), RGBToSpectrumTable::sRGB,
                                           alloc);
}

std::string RGBColorSpace::ToString() const {
    return StringPrintf("[ RGBColorSpace r: %s g: %s b: %s w: %s illuminant: %s RGBToXYZ: %s XYZToRGB: %s ]",
                        r, g, b, w, illuminant, XYZFromRGB, RGBFromXYZ);
}

SquareMatrix<3> ConvertRGBColorSpace(const RGBColorSpace &from, const RGBColorSpace &to) {
    if (from == to) return {};
    return to.RGBFromXYZ * from.XYZFromRGB;
}



} // namespace pbrt
