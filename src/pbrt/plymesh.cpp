
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


// shapes/plymesh.cpp*
#include <pbrt/plymesh.h>

#include <pbrt/paramdict.h>
#include <pbrt/base.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/sampling.h>
#include <pbrt/shapes.h>

#include <rply/rply.h>
#include <iostream>
#include <memory>

namespace pbrt {
using namespace std;

struct FaceCallbackContext {
    int face[4];
    std::vector<int> triIndices, quadIndices;
};

void rply_message_callback(p_ply ply, const char *message) {
    Warning("rply: %s", message);
}

/* Callback to handle vertex data from RPly */
int rply_vertex_callback(p_ply_argument argument) {
    Float *buffer;
    long index, flags;

    ply_get_argument_user_data(argument, (void **)&buffer, &flags);
    ply_get_argument_element(argument, nullptr, &index);

    int stride = (flags & 0x0F0) >> 4;
    int offset = flags & 0x00F;

    buffer[index * stride + offset] = (float)ply_get_argument_value(argument);

    return 1;
}

/* Callback to handle face data from RPly */
int rply_face_callback(p_ply_argument argument) {
    FaceCallbackContext *context;
    long flags;
    ply_get_argument_user_data(argument, (void **)&context, &flags);

    long length, value_index;
    ply_get_argument_property(argument, nullptr, &length, &value_index);

    if (length != 3 && length != 4) {
        Warning("plymesh: Ignoring face with %i vertices (only triangles and quads "
                "are supported!)",
                (int)length);
        return 1;
    } else if (value_index < 0) {
        return 1;
    }

    if (value_index >= 0)
        context->face[value_index] = (int)ply_get_argument_value(argument);

    if (value_index == length - 1) {
        if (length == 3)
            for (int i = 0; i < 3; ++i)
                context->triIndices.push_back(context->face[i]);
        else {
            CHECK_EQ(length, 4);

            // Note: modify order since we're specifying it as a blp...
            context->quadIndices.push_back(context->face[0]);
            context->quadIndices.push_back(context->face[1]);
            context->quadIndices.push_back(context->face[3]);
            context->quadIndices.push_back(context->face[2]);
        }
    }

    return 1;
}

int rply_faceindex_callback(p_ply_argument argument) {
    std::vector<int> *faceIndices;
    long flags;
    ply_get_argument_user_data(argument, (void **)&faceIndices, &flags);

    faceIndices->push_back((int)ply_get_argument_value(argument));

    return 1;
}

pstd::optional<PLYMesh> ReadPLYMesh(const std::string &filename, Allocator alloc) {
    PLYMesh mesh;

    p_ply ply = ply_open(filename.c_str(), rply_message_callback, 0, nullptr);
    if (ply == nullptr) {
        Error("Couldn't open PLY file \"%s\"", filename);
        return {};
    }

    if (ply_read_header(ply) == 0) {
        Error("Unable to read the header of PLY file \"%s\"", filename);
        return {};
    }

    p_ply_element element = nullptr;
    size_t vertexCount = 0, faceCount = 0;

    /* Inspect the structure of the PLY file */
    while ((element = ply_get_next_element(ply, element)) != nullptr) {
        const char *name;
        long nInstances;

        ply_get_element_info(element, &name, &nInstances);
        if (strcmp(name, "vertex") == 0)
            vertexCount = nInstances;
        else if (strcmp(name, "face") == 0)
            faceCount = nInstances;
    }

    if (vertexCount == 0 || faceCount == 0) {
        Error("%s: PLY file is invalid! No face/vertex elements found!",
              filename);
        return {};
    }

    mesh.p.resize(vertexCount);
    if (ply_set_read_cb(ply, "vertex", "x", rply_vertex_callback, mesh.p.data(), 0x30) == 0 ||
        ply_set_read_cb(ply, "vertex", "y", rply_vertex_callback, mesh.p.data(), 0x31) == 0 ||
        ply_set_read_cb(ply, "vertex", "z", rply_vertex_callback, mesh.p.data(), 0x32) == 0) {
        Error("%s: Vertex coordinate property not found!",
              filename);
        return {};
    }

    mesh.n.resize(vertexCount);
    if (ply_set_read_cb(ply, "vertex", "nx", rply_vertex_callback, mesh.n.data(), 0x30) == 0 ||
        ply_set_read_cb(ply, "vertex", "ny", rply_vertex_callback, mesh.n.data(), 0x31) == 0 ||
        ply_set_read_cb(ply, "vertex", "nz", rply_vertex_callback, mesh.n.data(), 0x32) == 0)
        mesh.n.resize(0);

    /* There seem to be lots of different conventions regarding UV coordinate
     * names */
    mesh.uv.resize(vertexCount);
    if (((ply_set_read_cb(ply, "vertex", "u", rply_vertex_callback, mesh.uv.data(), 0x20) != 0) &&
         (ply_set_read_cb(ply, "vertex", "v", rply_vertex_callback, mesh.uv.data(), 0x21) != 0)) ||
        ((ply_set_read_cb(ply, "vertex", "s", rply_vertex_callback, mesh.uv.data(), 0x20) != 0) &&
         (ply_set_read_cb(ply, "vertex", "t", rply_vertex_callback, mesh.uv.data(), 0x21) != 0)) ||
        ((ply_set_read_cb(ply, "vertex", "texture_u", rply_vertex_callback, mesh.uv.data(), 0x20) != 0) &&
         (ply_set_read_cb(ply, "vertex", "texture_v", rply_vertex_callback, mesh.uv.data(), 0x21) != 0)) ||
        ((ply_set_read_cb(ply, "vertex", "texture_s", rply_vertex_callback, mesh.uv.data(), 0x20) != 0) &&
         (ply_set_read_cb(ply, "vertex", "texture_t", rply_vertex_callback, mesh.uv.data(), 0x21) != 0)))
        ;
    else
        mesh.uv.resize(0);

    FaceCallbackContext context;
    context.triIndices.reserve(faceCount * 3);
    context.quadIndices.reserve(faceCount * 4);
    if (ply_set_read_cb(ply, "face", "vertex_indices", rply_face_callback, &context, 0) == 0)
        ErrorExit("%s: vertex indices not found in PLY file", filename);

    if (ply_set_read_cb(ply, "face", "face_indices", rply_faceindex_callback,
                        &mesh.faceIndices, 0) != 0)
        mesh.faceIndices.reserve(faceCount);

    if (ply_read(ply) == 0)
        ErrorExit("%s: unable to read the contents of PLY file", filename);

    mesh.triIndices = std::move(context.triIndices);
    mesh.quadIndices = std::move(context.quadIndices);

    ply_close(ply);

    for (int idx : mesh.triIndices)
        if (idx < 0 || idx >= mesh.p.size())
            ErrorExit("plymesh: Vertex index %i is out of bounds! "
                  "Valid range is [0..%i)", idx, int(mesh.p.size()));
    for (int idx : mesh.quadIndices)
        if (idx < 0 || idx >= mesh.p.size())
            ErrorExit("plymesh: Vertex index %i is out of bounds! "
                  "Valid range is [0..%i)", idx, int(mesh.p.size()));

    return mesh;
}

pstd::vector<ShapeHandle> CreatePLYMesh(
    const Transform *worldFromObject, bool reverseOrientation,
    const ParameterDictionary &dict, Allocator alloc) {
    ProfilerScope _(ProfilePhase::PLYLoading);
    std::string filename = ResolveFilename(dict.GetOneString("plyfile", ""));
    pstd::optional<PLYMesh> plyMesh = ReadPLYMesh(filename);
    if (!plyMesh)
        return {};

    pstd::vector<ShapeHandle> shapes(alloc);
    if (!plyMesh->triIndices.empty()) {
        TriangleMesh *mesh =
            alloc.new_object<TriangleMesh>(*worldFromObject, reverseOrientation,
                                           plyMesh->triIndices, plyMesh->p,
                                           std::vector<Vector3f>(), plyMesh->n,
                                           plyMesh->uv, plyMesh->faceIndices);
        shapes = mesh->CreateTriangles(alloc);
    }

    if (!plyMesh->quadIndices.empty()) {
        pstd::vector<ShapeHandle> quadMesh =
            BilinearPatchMesh::Create(worldFromObject, reverseOrientation,
                                      plyMesh->quadIndices, plyMesh->p, plyMesh->n,
                                      plyMesh->uv, plyMesh->faceIndices,
                                      nullptr /* image dist */, alloc);
        shapes.insert(shapes.end(), quadMesh.begin(), quadMesh.end());
    }

    return shapes;
}

}  // namespace pbrt
