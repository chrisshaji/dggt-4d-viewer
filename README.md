# DGGT 4D Gaussian Viewer

Interactive viewer for visualizing 4D Gaussian scene reconstructions produced from DGGT driving-scene predictions.

This repository contains the viewer used to inspect DGGT scene exports with static and dynamic Gaussian content over time. The viewer is designed to load a precomputed `.pt` scene file and render it interactively with `gsplat`, `viser`, and `nerfview`.

> **Important:** DGGT and TAPIP3D inference are **not** run inside the viewer. The expensive reconstruction/interpolation step is performed beforehand. The viewer loads the already-generated scene tensors from a `.pt` file and rasterizes them interactively.

---

## Features

- Interactive 3D Gaussian rendering with `gsplat`
- Temporal playback of DGGT scene states
- Supports `DGGT_VIEWER_SCENE_V1` and `DGGT_VIEWER_SCENE_INTERP_V1`
- Shared static Gaussian scene
- Time-dependent static Gaussian opacity / lifespan weighting
- Dynamic Gaussian states over time
- Per-camera dynamic Gaussian banks
- Dynamic camera selection: nearest-view, manual, or all-six diagnostic mode
- Optional opacity crossfade between adjacent saved dynamic states
- Ego-follow camera mode and manual navigation
- +Z-up vehicle/world normalization
- Fast interactive preview mode for large scenes
- Optional learned-sky and dynamic-probability visualization when those fields are present

---

## Repository Structure

```text
.
├── dggt_viewer_4d_v6.py
├── README.md
└── .gitignore
```

Large scene files and model checkpoints should **not** be committed directly to this repository. Host them separately and place download links below.

---

## Sample Files

### Sample DGGT Viewer Scene

**Scene file:** `dggt_viewer_scene_interp6.pt`

**Download:**  
`<ADD_SAMPLE_SCENE_FILE_LINK_HERE>`

This is the file passed directly to the viewer with `--scene`.

### DGGT Checkpoint

If you also want to provide the DGGT checkpoint used to generate the scene:

**Checkpoint:** `model_latest_nuscenes.pt`

**Download:**  
`<ADD_DGGT_CHECKPOINT_LINK_HERE>`

### TAPIP3D Tracking Checkpoint

If you also release the interpolation/export pipeline:

**Checkpoint:** `tracking_model.pth`

**Download:**  
`<ADD_TAPIP3D_CHECKPOINT_LINK_HERE>`

> The viewer itself does not require the DGGT or TAPIP3D checkpoints. It only needs the exported viewer scene `.pt` file.

---

## Current Pipeline

```text
NuScenes / driving RGB images
        |
        v
      DGGT
        |
        +-- predicted camera parameters
        +-- predicted depth / 3D point maps
        +-- Gaussian attributes
        +-- dynamic confidence
        +-- Gaussian confidence / lifespan information
        |
        v
DGGT anchor scene predictions
        |
        v
TAPIP3D / DGGT interpolation pipeline
        |
        v
Precomputed static + dynamic Gaussian states
        |
        v
dggt_viewer_scene_interp6.pt
        |
        v
dggt_viewer_4d_v6.py
        |
        v
Interactive gsplat / Viser viewer
```

The current six-camera exporter stores:

- one shared static Gaussian set
- dynamic Gaussian states indexed by time and source camera
- camera extrinsics and intrinsics
- temporal metadata
- source camera images / thumbnails
- optional learned-sky information
- optional dynamic-probability maps

The viewer assembles the active static and dynamic Gaussian tensors for the selected time and renders them from the interactive camera.

---

## Dynamic Banks

The current six-camera viewer stores separate dynamic Gaussian predictions for each source camera.

At a given temporal state, the viewer can select the dynamic bank corresponding to:

- the source camera most closely aligned with the current viewer direction,
- a manually selected camera, or
- all six cameras for debugging.

The `all-six` option is mainly diagnostic. Different cameras may independently reconstruct the same moving object, so rendering all dynamic banks simultaneously can create duplicate or blurred moving objects.

The static scene is different: static Gaussians from the input observations are combined into a shared scene representation.

---

## Requirements

The viewer requires a CUDA-capable PyTorch environment for practical performance.

Main Python packages:

```text
torch
numpy
gsplat
viser
nerfview
```

Install PyTorch and `gsplat` using versions compatible with your CUDA environment.

The lightweight viewer dependencies can typically be installed with:

```bash
python3 -m pip install numpy viser nerfview
```

Make sure `gsplat` is already installed and working before launching the viewer.

---

## Running the Viewer

Basic usage:

```bash
python3 dggt_viewer_4d_v6.py \
    --scene /path/to/dggt_viewer_scene_interp6.pt
```

Example with explicit GPU and port:

```bash
python3 dggt_viewer_4d_v6.py \
    --scene /path/to/dggt_viewer_scene_interp6.pt \
    --device cuda:0 \
    --port 8080
```

The script will print a Viser URL. Open that URL in a browser.

---

## Command-Line Options

```text
--scene PATH
    Required path to the exported DGGT viewer scene.

--device DEVICE
    Torch device. Default: cuda:0

--port PORT
    Viser server port. Default: 8080

--frames N
    0 uses the saved temporal states exactly.
    A positive value resamples only the viewer timeline and does not create
    additional DGGT/TAPIP3D scene states.

--fps FPS
    Playback speed. Default: 8

--near VALUE
    Near rendering plane. Default: 0.2

--far VALUE
    Far rendering plane. Default: 400

--radius-clip VALUE
    gsplat radius clipping threshold. Default: 0.0

--interactive-radius-clip VALUE
    Radius-clip floor used during fast preview rendering.
    Default: 1.0

--nav-focus VALUE
    Initial viewer navigation / look-at distance in meters.
    Default: 2.5
```

---

## Viewer Behavior

### Static Geometry

The static Gaussian bank is loaded once from the scene file.

Static Gaussian positions are not regenerated when the timeline moves. Instead, their opacity can vary with time using saved source-time and confidence/lifespan information.

### Dynamic Geometry

Interpolated scene files contain saved dynamic states over time.

The viewer normally selects one camera-specific dynamic bank for the current temporal state. Optional crossfade blends opacity between adjacent saved states; it does **not** create new physical DGGT motion states.

### Camera Movement

Moving the interactive viewer camera does not rerun DGGT or TAPIP3D.

It only changes the virtual camera used by `gsplat` to rasterize the currently assembled Gaussian scene.

---

## Example Data

The original experiments for this viewer use NuScenes multi-camera driving scenes.

A typical current reconstruction pipeline uses multiple temporal anchor frames and six cameras per anchor. The DGGT model processes the input images jointly, after which per-camera temporal streams are used by the interpolation/export pipeline.

The exact experiment setup may change as the project evaluates more paper-faithful DGGT interpolation protocols.

---

## Large Files

Do not commit large `.pt`, `.pth`, dataset, or generated-output files directly to normal Git history.

Recommended alternatives include:

- GitHub Releases
- Hugging Face
- Google Drive
- institutional storage
- Git LFS

Then place public/sample download links in the **Sample Files** section above.

---

## Acknowledgements

This viewer builds on the DGGT reconstruction pipeline and uses:

- DGGT
- TAPIP3D
- gsplat
- Viser
- nerfview

Please refer to the original projects and papers for their respective licenses and citation requirements.

---

## Status

This repository is research code and is actively evolving.

Current work focuses on:

- validating DGGT interpolation against real held-out driving frames
- improving dynamic-object reconstruction
- evaluating camera/depth effects on TAPIP3D motion
- improving multi-camera dynamic fusion
- improving interactive 4D Gaussian visualization
