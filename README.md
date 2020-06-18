pbrt, Version 4 WIP (NVIDIA Internal)
=====================================

Getting started
---------------

In order to check out all of pbrt's dependencies, make sure to use the
`--recursive` flag when cloning the repository.

If you accidentally already cloned pbrt without this flag (or to update the
pbrt source tree after a new submodule has been added, run the following
command to also fetch the dependencies:
```bash
$ git submodule update --init --recursive
```

pbrt uses [cmake](http://www.cmake.org/) for its build system.  It will
automatically attempt to find the CUDA compiler, looking in the usual
places; the cmake output will indicate whether it was successful. It is
necessary to set the cmake `PBRT_OPTIX7_PATH` configuration option to point
at an Optix7 install in order for the GPU rendering path to be built.

pbrt builds with CUDA 10.2 and CUDA 11 on Linux. It's only been tested with
OptiX 7 so far.  It does not currently build on Windows (purely due to lack
of attention to that).

Using pbrt on the GPU
---------------------

Even with a GPU build, pbrt uses the CPU by default unless the `--gpu`
command-line option is given.  Note that when rendering with the GPU, the
`--spp` command-line flag can be helpful to easily crank up the number of
samples per pixel.

The pbrt scene file format has changed since pbrt-v3.  There is an
`--upgrade` command-line option that (usually) upgrades pbrt-v3 scenes to
the new format.  The environment map parameterization has also changed
(from equi-rect to an equi-area mapping); you can upgrade environment maps
using `imgtool makeenv old.exr --outfile new.exr`.

A number of preconverted scenes are [available for download](https://drive.google.com/file/d/1IMTk8isTNU1s3SDoaZqcXHjFglg2X7pR/view?usp=sharing).

To render with the OptiX denoiser, change the scene's "Film" type from
"rgb" to "gbuffer".  Then, the resulting EXR can be denoised using `imgtool
denoise-optix noisy.exr --outfile denoised.exr`.

If you'd like to watch rendering while it happens, try the
[tev](https://github.com/Tom94/tev) image viewer.  If you launch tev like
`tev --hostname <local-ip>:port`, then if you run pbrt as `pbrt --gpu
--display-server <local-ip>:port scene.pbrt`, then the image will be
progressively displayed as it renders (even on a different machine).
