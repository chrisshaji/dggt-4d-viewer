# DGGT 4D Gaussian Viewer

Interactive viewer for DGGT driving-scene reconstructions using `gsplat`, `viser`, and `nerfview`.

The viewer loads a precomputed DGGT scene `.pt` file containing camera poses, static Gaussians, dynamic Gaussian states, timing information, and source-camera images. DGGT and TAPIP3D inference are performed before launching the viewer; the viewer itself only loads the exported scene and rasterizes it interactively.

**Sample viewer scene (`dggt_viewer_scene_interp6.pt`):** [Download via SwissTransfer](https://www.swisstransfer.com/dl/01a102b0-cf9e-717e-bb6a-5b7e7f496c86)

## Installation

Use a CUDA-enabled PyTorch environment with `gsplat` installed, then install the viewer dependencies:

```bash
python3 -m pip install numpy viser nerfview
```

## Run

The way I run it is I ssh into my lab PC, and i run the below command. This command gives me a link that I can open on my web browser to view the viewer. Feel free to make modifications so you can view it easily.
```bash
python3 dggt_viewer.py \
    --scene data/dggt_viewer_scene_interp6.pt \
    --device cuda:0 \
    --port 8080
```


The viewer does **not** contain hardcoded dataset or checkpoint paths. The scene path is supplied with `--scene`; camera poses, intrinsics, Gaussian banks, timing, and source images are loaded from that `.pt` file.

## Scene image comparisons

`images_u8` in the scene file is used for the source-camera thumbnails shown with the camera rig. It is inside the.pt files so don't worry about it. 

## Current improvements

The main remaining work is improving dynamic-object motion quality and multi-camera dynamic consistency. Viewer navigation should also be made more fluid, especially panning and general camera controls, closer to the interaction style of this example viewer:

https://www.3dgsviewers.com/m/nanxiang-DMN3pE

Planned viewer additions include clearer rendered-vs-ground-truth inspection, improved free-camera navigation, and better visualization/debugging of dynamic motion.
