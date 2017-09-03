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

// core/imageio.cpp*
#include "imageio.h"

#include "error.h"
#include "fileutil.h"
#include "fp16.h"
#include "geometry.h"
#include "image.h"
#include "spectrum.h"

#include "ext/lodepng.h"

#include <ImfChannelList.h>
#include <ImfFloatAttribute.h>
#include <ImfMatrixAttribute.h>
#include <ImfInputFile.h>
#include <ImfOutputFile.h>
#include <ImfFrameBuffer.h>

namespace pbrt {

// ImageIO Local Declarations
static bool ReadEXR(const std::string &name, Image *image,
                    ImageMetadata *metadata);
static bool ReadPNG(const std::string &name, bool gamma, Image *image,
                    ImageMetadata *metadata);
static bool ReadPFM(const std::string &filename, Image *image,
                    ImageMetadata *metadata);

// ImageIO Function Definitions
bool Image::Read(const std::string &name, Image *image, ImageMetadata *metadata,
                 bool gamma) {
    if (HasExtension(name, ".exr"))
        return ReadEXR(name, image, metadata);
    else if (HasExtension(name, ".png"))
        return ReadPNG(name, gamma, image, metadata);
    else if (HasExtension(name, ".pfm"))
        return ReadPFM(name, image, metadata);
    else {
        Error("%s: no support for reading images with this extension",
              name.c_str());
        return false;
    }
}

bool Image::Write(const std::string &name, const ImageMetadata *metadata) const {
    if (metadata && !metadata->pixelBounds.Empty())
        CHECK_EQ(metadata->pixelBounds.Area(), resolution.x * resolution.y);

    if (HasExtension(name, ".exr"))
        return WriteEXR(name, metadata);
    else if (HasExtension(name, ".pfm"))
        return WritePFM(name, metadata);
    else if (HasExtension(name, ".png"))
        return WritePNG(name, metadata);
    else {
        Error("%s: no support for writing images with this extension",
              name.c_str());
        return false;
    }
}

///////////////////////////////////////////////////////////////////////////
// OpenEXR

static Imf::FrameBuffer imageToFrameBuffer(const Image &image,
                                           const Imath::Box2i &dataWindow) {
    size_t xStride = TexelBytes(image.format);
    size_t yStride = image.resolution.x * xStride;
    // Would be nice to use PixelOffset(-dw.min.x, -dw.min.y) but
    // it checks to make sure the coordiantes are >= 0 (which
    // usually makes sense...)
    char *originPtr = (((char *)image.RawPointer({0, 0})) -
                       dataWindow.min.x * xStride - dataWindow.min.y * yStride);

    Imf::FrameBuffer fb;
    switch (image.format) {
    case PixelFormat::Y16:
        fb.insert("Y", Imf::Slice(Imf::HALF, originPtr, xStride, yStride));
        break;
    case PixelFormat::RGB16:
        fb.insert("R", Imf::Slice(Imf::HALF, originPtr, xStride, yStride));
        fb.insert("G", Imf::Slice(Imf::HALF, originPtr + sizeof(uint16_t),
                                  xStride, yStride));
        fb.insert("B", Imf::Slice(Imf::HALF, originPtr + 2 * sizeof(uint16_t),
                                  xStride, yStride));
        break;
    case PixelFormat::Y32:
        fb.insert("Y", Imf::Slice(Imf::FLOAT, originPtr, xStride, yStride));
        break;
    case PixelFormat::RGB32:
        fb.insert("R", Imf::Slice(Imf::FLOAT, originPtr, xStride, yStride));
        fb.insert("G", Imf::Slice(Imf::FLOAT, originPtr + sizeof(float),
                                  xStride, yStride));
        fb.insert("B", Imf::Slice(Imf::FLOAT, originPtr + 2 * sizeof(float),
                                  xStride, yStride));
        break;
    default:
        LOG(FATAL) << "Unexpected image format";
    }
    return fb;
}

static bool ReadEXR(const std::string &name, Image *image,
                    ImageMetadata *metadata) {
    try {
        Imf::InputFile file(name.c_str());
        Imath::Box2i dw = file.header().dataWindow();

        if (metadata) {
            const Imf::FloatAttribute *renderTimeAttrib =
                file.header().findTypedAttribute<Imf::FloatAttribute>("renderTimeSeconds");
            if (renderTimeAttrib)
                metadata->renderTimeSeconds = renderTimeAttrib->value();

            const Imf::M44fAttribute *worldToCameraAttrib =
                file.header().findTypedAttribute<Imf::M44fAttribute>("worldToCamera");
            if (worldToCameraAttrib)
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        // Can't memcpy since Float may be a double...
                        metadata->worldToCamera.m[i][j] =
                            worldToCameraAttrib->value().getValue()[4*i+j];

            const Imf::M44fAttribute *worldToNDCAttrib =
                file.header().findTypedAttribute<Imf::M44fAttribute>("worldToNDC");
            if (worldToNDCAttrib)
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        metadata->worldToNDC.m[i][j] =
                            worldToNDCAttrib->value().getValue()[4*i+j];

            // OpenEXR uses inclusive pixel bounds; adjust to non-inclusive
            // (the convention pbrt uses) in the values returned.
            metadata->pixelBounds = {{dw.min.x, dw.min.y}, {dw.max.x + 1, dw.max.y + 1}};

            Imath::Box2i dispw = file.header().displayWindow();
            metadata->fullResolution.x = dispw.max.x - dispw.min.x + 1;
            metadata->fullResolution.y = dispw.max.y - dispw.min.y + 1;
        }

        int width = dw.max.x - dw.min.x + 1;
        int height = dw.max.y - dw.min.y + 1;

        const Imf::ChannelList &channels = file.header().channels();
        const Imf::Channel *rc = channels.findChannel("R");
        const Imf::Channel *gc = channels.findChannel("G");
        const Imf::Channel *bc = channels.findChannel("B");
        if (rc && gc && bc) {
            if (rc->type == Imf::HALF && gc->type == Imf::HALF &&
                bc->type == Imf::HALF)
                *image = Image(PixelFormat::RGB16, {width, height});
            else
                *image = Image(PixelFormat::RGB32, {width, height});
        } else if (channels.findChannel("Y")) {
            if (channels.findChannel("Y")->type == Imf::HALF)
                *image = Image(PixelFormat::Y16, {width, height});
            else
                *image = Image(PixelFormat::Y32, {width, height});
        } else {
            std::string channelNames;
            for (auto iter = channels.begin(); iter != channels.end(); ++iter) {
                channelNames += iter.name();
                channelNames += ' ';
            }
            Error("%s: didn't find RGB or Y stored in image. Channels: %s",
                  name.c_str(), channelNames.c_str());
            return false;
        }
        file.setFrameBuffer(imageToFrameBuffer(*image, dw));
        file.readPixels(dw.min.y, dw.max.y);

        LOG(INFO) << StringPrintf("Read EXR image %s (%d x %d)", name.c_str(),
                                  width, height);
        return true;
    } catch (const std::exception &e) {
        Error("Unable to read image file \"%s\": %s", name.c_str(), e.what());
    }

    return false;
}

bool Image::WriteEXR(const std::string &name, const ImageMetadata *metadata) const {
    if (Is8Bit(format)) {
        if (nChannels() == 1)
            return ConvertToFormat(PixelFormat::Y16).WriteEXR(name, metadata);
        else
            return ConvertToFormat(PixelFormat::RGB16).WriteEXR(name, metadata);
    }
    CHECK(Is16Bit(format) || Is32Bit(format));

    try {
        Imath::Box2i displayWindow, dataWindow;
        if (metadata) {
            // Agan, -1 offsets to handle inclusive indexing in OpenEXR...
            displayWindow = {Imath::V2i(0, 0),
                             Imath::V2i(metadata->fullResolution.x - 1,
                                        metadata->fullResolution.y - 1)};
            dataWindow = {Imath::V2i(metadata->pixelBounds.pMin.x,
                                     metadata->pixelBounds.pMin.y),
                          Imath::V2i(metadata->pixelBounds.pMax.x - 1,
                                     metadata->pixelBounds.pMax.y - 1)};
        } else
            displayWindow = dataWindow =
                {Imath::V2i(0, 0), Imath::V2i(resolution.x - 1, resolution.y - 1)};

        Imf::FrameBuffer fb = imageToFrameBuffer(*this, dataWindow);

        Imf::Header header(displayWindow, dataWindow);
        for (auto iter = fb.begin(); iter != fb.end(); ++iter)
            header.channels().insert(iter.name(), iter.slice().type);

        if (metadata) {
            if (metadata->renderTimeSeconds > 0)
                header.insert("renderTimeSeconds", Imf::FloatAttribute(metadata->renderTimeSeconds));
            // TODO: fix this for Float = double builds.
            if (!metadata->worldToCamera.IsZero())
                header.insert("worldToCamera", Imf::M44fAttribute(metadata->worldToCamera.m));
            if (!metadata->worldToNDC.IsZero())
                header.insert("worldToNDC", Imf::M44fAttribute(metadata->worldToNDC.m));
        }

        Imf::OutputFile file(name.c_str(), header);
        file.setFrameBuffer(fb);
        file.writePixels(resolution.y);
    } catch (const std::exception &exc) {
        Error("Error writing \"%s\": %s", name.c_str(), exc.what());
        return false;
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////
// PNG Function Definitions

static inline uint8_t FloatToSRGB(Float v) {
    return uint8_t(Clamp(255.f * LinearToSRGB(v) + 0.5f, 0.f, 255.f));
}

static bool ReadPNG(const std::string &name, bool gamma, Image *image,
                    ImageMetadata *metadata) {
    auto contents = ReadFileContents(name);
    if (!contents)
        return false;

    unsigned width, height;
    LodePNGState state;
    lodepng_state_init(&state);
    unsigned int error = lodepng_inspect(&width, &height, &state,
                                         (const unsigned char *)contents->data(),
                                         contents->size());
    if (error != 0) {
        Error("%s: %s", name.c_str(), lodepng_error_text(error));
        return false;
    }

    switch (state.info_png.color.colortype) {
    case LCT_GREY:
    case LCT_GREY_ALPHA: {
        std::vector<unsigned char> buf;
        int bpp = state.info_png.color.bitdepth == 16 ? 16 : 8;
        error = lodepng::decode(buf, width, height,
                                (const unsigned char *)contents->data(),
                                contents->size(), LCT_GREY, bpp);
        if (error != 0) {
            Error("%s: %s", name.c_str(), lodepng_error_text(error));
            return false;
        }

        if (state.info_png.color.bitdepth == 16) {
            *image = Image(PixelFormat::Y16, Point2i(width, height));
            auto bufIter = buf.begin();
            for (unsigned int y = 0; y < height; ++y)
                for (unsigned int x = 0; x < width; ++x, bufIter += 2) {
                    // Convert from little endian.
                    Float v = (((int)bufIter[0] << 8) + (int)bufIter[1]) / 65535.f;
                    if (gamma) v = SRGBToLinear(v);
                    image->SetChannel(Point2i(x, y), 0, v);
                }
            CHECK(bufIter == buf.end());
        } else {
            *image = Image(gamma ? PixelFormat::SY8 : PixelFormat::Y8,
                           Point2i(width, height));
            std::copy(buf.begin(), buf.end(), (uint8_t *)image->RawPointer({0, 0}));
        }
        break;
    }
    default: {
        std::vector<unsigned char> buf;
        int bpp = state.info_png.color.bitdepth == 16 ? 16 : 8;
        error = lodepng::decode(buf, width, height,
                                (const unsigned char *)contents->data(),
                                contents->size(), LCT_RGB, bpp);
        if (error != 0) {
            Error("%s: %s", name.c_str(), lodepng_error_text(error));
            return false;
        }

        if (state.info_png.color.bitdepth == 16) {
            *image = Image(PixelFormat::RGB16, Point2i(width, height));
            auto bufIter = buf.begin();
            for (unsigned int y = 0; y < height; ++y)
                for (unsigned int x = 0; x < width; ++x, bufIter += 6) {
                    CHECK(bufIter < buf.end()) ;
                    // Convert from little endian.
                    Float rgb[3] = {
                        (((int)bufIter[0] << 8) + (int)bufIter[1]) / 65535.f,
                        (((int)bufIter[2] << 8) + (int)bufIter[3]) / 65535.f,
                        (((int)bufIter[4] << 8) + (int)bufIter[5]) / 65535.f };
                    if (gamma)
                        for (int c = 0; c < 3; ++c)
                            // TODO: this is slow; could replace with a LUT
                            rgb[c] = SRGBToLinear(rgb[c]);
                    image->SetSpectrum(Point2i(x, y), Spectrum::FromRGB(rgb));
                }
            CHECK(bufIter == buf.end());
        } else {
            *image = Image(gamma ? PixelFormat::SRGB8 : PixelFormat::RGB8,
                           Point2i(width, height));
            std::copy(buf.begin(), buf.end(), (uint8_t *)image->RawPointer({0, 0}));
        }
    }
    }

    return true;
}

bool Image::WritePNG(const std::string &name, const ImageMetadata *metadata) const {
    unsigned int error = 0;
    switch (format) {
    case PixelFormat::SRGB8:
        error = lodepng_encode24_file(name.c_str(), &p8[0], resolution.x,
                                      resolution.y);
        break;
    case PixelFormat::SY8:
        error = lodepng_encode_file(name.c_str(), &p8[0], resolution.x,
                                    resolution.y, LCT_GREY, 8 /* bitdepth */);
        break;
    case PixelFormat::RGB8:
    case PixelFormat::RGB16:
    case PixelFormat::RGB32: {
        std::unique_ptr<uint8_t[]> rgb8 =
            std::make_unique<uint8_t[]>(3 * resolution.x * resolution.y);
        for (int y = 0; y < resolution.y; ++y)
            for (int x = 0; x < resolution.x; ++x)
                for (int c = 0; c < 3; ++c)
                    rgb8[3 * (y * resolution.x + x) + c] =
                        FloatToSRGB(GetChannel({x, y}, c));

        error = lodepng_encode24_file(name.c_str(), rgb8.get(), resolution.x,
                                      resolution.y);
        break;
    }
    case PixelFormat::Y8:
    case PixelFormat::Y16:
    case PixelFormat::Y32: {
        std::unique_ptr<uint8_t[]> y8 =
            std::make_unique<uint8_t[]>(resolution.x * resolution.y);
        for (int y = 0; y < resolution.y; ++y)
            for (int x = 0; x < resolution.x; ++x)
                y8[y * resolution.x + x] = FloatToSRGB(GetChannel({x, y}, 0));

        error = lodepng_encode_file(name.c_str(), y8.get(), resolution.x,
                                    resolution.y, LCT_GREY, 8 /* bitdepth */);
        break;
    }
    }

    if (error != 0) {
        Error("Error writing PNG \"%s\": %s", name.c_str(),
              lodepng_error_text(error));
        return false;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////
// PFM Function Definitions

/*
 * PFM reader/writer code courtesy Jiawen "Kevin" Chen
 * (http://people.csail.mit.edu/jiawen/)
 */

static constexpr bool hostLittleEndian =
#if defined(__BYTE_ORDER__)
  #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    true
  #elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    false
  #else
    #error "__BYTE_ORDER__ defined but has unexpected value"
  #endif
#else
  #if defined(__LITTLE_ENDIAN__) || defined(__i386__) || defined(__x86_64__) || \
      defined(WIN32)
    true
  #elif defined(__BIG_ENDIAN__)
    false
  #elif defined(__sparc) || defined(__sparc__)
    false
  #else
    #error "Can't detect machine endian-ness at compile-time."
  #endif
#endif
    ;

#define BUFFER_SIZE 80

static inline int isWhitespace(char c) {
    return c == ' ' || c == '\n' || c == '\t';
}

// Reads a "word" from the fp and puts it into buffer and adds a null
// terminator.  i.e. it keeps reading until whitespace is reached.  Returns
// the number of characters read *not* including the whitespace, and
// returns -1 on an error.
static int readWord(FILE *fp, char *buffer, int bufferLength) {
    int n;
    int c;

    if (bufferLength < 1) return -1;

    n = 0;
    c = fgetc(fp);
    while (c != EOF && !isWhitespace(c) && n < bufferLength) {
        buffer[n] = c;
        ++n;
        c = fgetc(fp);
    }

    if (n < bufferLength) {
        buffer[n] = '\0';
        return n;
    }

    return -1;
}

static bool ReadPFM(const std::string &filename, Image *image, ImageMetadata *metadata) {
    std::vector<float> rgb32;
    char buffer[BUFFER_SIZE];
    unsigned int nFloats;
    int nChannels, width, height;
    float scale;
    bool fileLittleEndian;

    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp) goto fail;

    // read either "Pf" or "PF"
    if (readWord(fp, buffer, BUFFER_SIZE) == -1) goto fail;

    if (strcmp(buffer, "Pf") == 0)
        nChannels = 1;
    else if (strcmp(buffer, "PF") == 0)
        nChannels = 3;
    else
        goto fail;

    // read the rest of the header
    // read width
    if (readWord(fp, buffer, BUFFER_SIZE) == -1) goto fail;
    width = atoi(buffer);

    // read height
    if (readWord(fp, buffer, BUFFER_SIZE) == -1) goto fail;
    height = atoi(buffer);

    // read scale
    if (readWord(fp, buffer, BUFFER_SIZE) == -1) goto fail;
    sscanf(buffer, "%f", &scale);

    // read the data
    nFloats = nChannels * width * height;
    rgb32.resize(nFloats);
    for (int y = height - 1; y >= 0; --y)
      if (fread(&rgb32[nChannels * y * width], sizeof(float),
                nChannels * width, fp) != nChannels * width) goto fail;

    // apply endian conversian and scale if appropriate
    fileLittleEndian = (scale < 0.f);
    if (hostLittleEndian ^ fileLittleEndian) {
        uint8_t bytes[4];
        for (unsigned int i = 0; i < nFloats; ++i) {
            memcpy(bytes, &rgb32[i], 4);
            std::swap(bytes[0], bytes[3]);
            std::swap(bytes[1], bytes[2]);
            memcpy(&rgb32[i], bytes, 4);
        }
    }
    if (std::abs(scale) != 1.f)
        for (unsigned int i = 0; i < nFloats; ++i) rgb32[i] *= std::abs(scale);

    // create RGBs...
    *image = Image(std::move(rgb32),
                   nChannels == 1 ? PixelFormat::Y32 : PixelFormat::RGB32,
                   Point2i(width, height));
    fclose(fp);

    LOG(INFO) << StringPrintf("Read PFM image %s (%d x %d)",
                              filename.c_str(), width, height);
    return true;

fail:
    Error("Error reading PFM file \"%s\"", filename.c_str());
    if (fp) fclose(fp);
    return false;
}

bool Image::WritePFM(const std::string &filename, const ImageMetadata *metadata) const {
    FILE *fp;
    float scale;

    fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        Error("Unable to open output PFM file \"%s\"", filename.c_str());
        return false;
    }

    std::unique_ptr<float[]> scanline = std::make_unique<float[]>(3 * resolution.x);

    // only write 3 channel PFMs here...
    if (fprintf(fp, "PF\n") < 0) goto fail;

    // write the width and height, which must be positive
    if (fprintf(fp, "%d %d\n", resolution.x, resolution.y) < 0) goto fail;

    // write the scale, which encodes endianness
    scale = hostLittleEndian ? -1.f : 1.f;
    if (fprintf(fp, "%f\n", scale) < 0) goto fail;

    // write the data from bottom left to upper right as specified by
    // http://netpbm.sourceforge.net/doc/pfm.html
    // The raster is a sequence of pixels, packed one after another, with no
    // delimiters of any kind. They are grouped by row, with the pixels in each
    // row ordered left to right and the rows ordered bottom to top.
    for (int y = resolution.y - 1; y >= 0; y--) {
        for (int x = 0; x < resolution.x; ++x) {
            Spectrum s = GetSpectrum({x, y});
            Float rgb[3];
            s.ToRGB(rgb);
            for (int c = 0; c < 3; ++c)
                // Assign element-wise in case Float is typedefed as 'double'.
                scanline[3 * x + c] = rgb[c];
        }
        if (fwrite(&scanline[0], sizeof(float), 3 * resolution.x, fp) <
            (size_t)(3 * resolution.x))
            goto fail;
    }

    fclose(fp);
    return true;

fail:
    Error("Error writing PFM file \"%s\"", filename.c_str());
    fclose(fp);
    return false;
}

}  // namespace pbrt
