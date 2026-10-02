#!/usr/bin/env python3
"""
DGGT 4D viewer V6 — cached DGGT/TAPIP3D viewer for responsive navigation.

Goals compared with V5:
  * Cache the assembled Gaussian bank so mouse motion does not rebuild/filter
    millions of Gaussians on every render.
  * Update the camera rig only when viewer time changes, avoiding repeated
    websocket traffic during orbit/pan.
  * Use a configurable radius-clip floor for interactive preview renders.
  * Navigation behaves like the existing STORM viewer: normal viser/nerfview
    controls in a conventional +Z-up world frame.
  * DGGT scene is transformed once into a vehicle-centric coordinate system:
      origin = six-camera rig center at t0
      +Y     = CAM_FRONT forward direction
      +Z     = estimated vehicle/world up
      +X     = vehicle right
  * Initial camera starts at the car and looks forward.
  * Optional ego-follow translates the viewer along the predicted car path.
    Any manual camera motion automatically stops ego-follow so the user can
    pan/orbit freely. "Resume ego follow" snaps back to the car.
  * Dynamic rendering defaults to the nearest DGGT input camera, which matches
    released DGGT mode-2 semantics much better than stacking all six dynamic
    banks at once.
  * Dynamic time is still discrete in mode 2. Optional crossfade changes
    opacity between adjacent DGGT states; it is NOT physical interpolation.

This script loads either:
  * DGGT_VIEWER_SCENE_V1 (the original 4-state mode-2 export), or
  * DGGT_VIEWER_SCENE_INTERP_V1 (the new official per-camera mode-3/TAPIP3D export).

It does not load the DGGT or TAPIP3D networks on the lab PC.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_render_size(render_arg) -> Tuple[int, int]:
    if hasattr(render_arg, "preview_render"):
        if bool(render_arg.preview_render):
            return int(render_arg.render_width), int(render_arg.render_height)
        return int(render_arg.viewer_width), int(render_arg.viewer_height)
    if isinstance(render_arg, (tuple, list, np.ndarray)) and len(render_arg) == 2:
        return int(render_arg[0]), int(render_arg[1])
    if hasattr(render_arg, "width") and hasattr(render_arg, "height"):
        return int(render_arg.width), int(render_arg.height)
    raise TypeError(f"Unknown render argument: {type(render_arg).__name__}")


def quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product, both wxyz. q1 can be [4], q2 [...,4]."""
    q1 = q1.reshape(*([1] * (q2.ndim - 1)), 4)
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ],
        dim=-1,
    )


class DGGT4DViewer:
    def __init__(
        self,
        scene: Dict[str, Any],
        device: torch.device,
        port: int,
        frames: int,
        fps: float,
        near: float,
        far: float,
        radius_clip: float,
        nav_focus: float,
        interactive_radius_clip: float,
    ):
        try:
            import viser
            import viser.transforms as vtf
            import nerfview
            from gsplat.rendering import rasterization
        except Exception as exc:
            raise RuntimeError(
                "Missing viewer dependencies. In the existing STORM/viewer env run:\n"
                "  python3 -m pip install viser nerfview\n"
                "and make sure gsplat is installed."
            ) from exc

        supported = {
            "DGGT_VIEWER_SCENE_V1",
            "DGGT_VIEWER_SCENE_INTERP_V1",
        }
        if scene.get("format") not in supported:
            raise ValueError(
                f"Unsupported scene format: {scene.get('format')!r}; "
                f"expected one of {sorted(supported)}"
            )
        self.interpolated_scene = (
            scene.get("format") == "DGGT_VIEWER_SCENE_INTERP_V1"
        )

        self.viser = viser
        self.vtf = vtf
        self.nerfview = nerfview
        self.rasterization = rasterization
        self.device = device
        self.scene = scene
        self.near = float(near)
        self.far = float(far)
        self.nav_focus = max(float(nav_focus), 0.25)
        self.interactive_radius_clip = max(float(interactive_radius_clip), 0.0)

        self.frame_times_s = torch.as_tensor(scene["frame_times_s"]).float()
        self.frame_times_norm = torch.as_tensor(scene["frame_times_norm"]).float()
        self.camera_order = list(scene["camera_order"])
        self.T = len(self.frame_times_s)
        self.V = len(self.camera_order)

        # For an interpolated scene, default to exactly the saved DGGT/TAPIP3D
        # temporal states instead of inventing extra GUI frames.
        if int(frames) <= 0:
            self.frames = self.T
            self.timeline_s = self.frame_times_s.clone()
        else:
            self.frames = max(2, int(frames))
            if self.frames == self.T:
                self.timeline_s = self.frame_times_s.clone()
            else:
                self.timeline_s = torch.linspace(
                    float(self.frame_times_s[0]),
                    float(self.frame_times_s[-1]),
                    self.frames,
                )

        # ------------------------------------------------------------
        # Load DGGT Gaussian banks.
        # ------------------------------------------------------------
        st = scene["static"]
        self.static_means = st["means"].to(device).float()
        self.static_colors = st["colors"].to(device).float()
        self.static_opacities = st["opacities"].to(device).float().reshape(-1)
        self.static_scales = st["scales"].to(device).float()
        self.static_quats = st["quats"].to(device).float()
        self.static_conf = st["confidence"].to(device).float().reshape(-1)
        self.static_source_time = st["source_time_norm"].to(device).float().reshape(-1)

        tokens = scene["dynamic_tokens"]
        self.dynamic_by_time_cam = []
        for ti in range(self.T):
            row = []
            for vi in range(self.V):
                matches = [
                    x for x in tokens
                    if int(x["time_index"]) == ti
                    and int(x["camera_index"]) == vi
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected exactly one dynamic bank for time={ti}, cam={vi}; "
                        f"got {len(matches)}"
                    )
                x = matches[0]
                row.append({
                    "means": x["means"].to(device).float(),
                    "colors": x["colors"].to(device).float(),
                    "opacities": x["opacities"].to(device).float().reshape(-1),
                    "scales": x["scales"].to(device).float(),
                    "quats": x["quats"].to(device).float(),
                    "dynamic_prob": x["dynamic_prob"].to(device).float().reshape(-1),
                })
            self.dynamic_by_time_cam.append(row)

        # ------------------------------------------------------------
        # Load predicted cameras.
        # ------------------------------------------------------------
        self.images_u8 = torch.as_tensor(scene["images_u8"]).cpu()
        ext_w2c = torch.as_tensor(scene["extrinsics_w2c"]).float().cpu()
        self.intrinsics = torch.as_tensor(scene["intrinsics"]).float().cpu()

        c2w_world = torch.linalg.inv(ext_w2c).numpy().reshape(
            self.T, self.V, 4, 4
        )
        self.K = self.intrinsics.numpy().reshape(self.T, self.V, 3, 3)
        self.images_grid = self.images_u8.reshape(
            self.T, self.V, *self.images_u8.shape[1:]
        )

        # ------------------------------------------------------------
        # Build a conventional vehicle coordinate system.
        #
        # DGGT camera axes are OpenCV-style:
        #   +X right, +Y down, +Z forward.
        #
        # We transform the whole reconstructed world so the viewer gets:
        #   +X vehicle right
        #   +Y vehicle forward
        #   +Z vehicle/world up
        #
        # This makes viser use the same ordinary +Z-up interaction behavior
        # as the STORM viewer instead of trying to operate in DGGT's arbitrary
        # predicted world frame.
        # ------------------------------------------------------------
        origin_world = c2w_world[0, :, :3, 3].mean(axis=0)

        up_world = (-c2w_world[:, :, :3, 1]).reshape(-1, 3).mean(axis=0)
        up_world /= max(np.linalg.norm(up_world), 1.0e-8)

        forward_world = c2w_world[0, 0, :3, 2].copy()
        forward_world -= up_world * float(np.dot(forward_world, up_world))
        forward_world /= max(np.linalg.norm(forward_world), 1.0e-8)

        right_world = np.cross(forward_world, up_world)
        right_world /= max(np.linalg.norm(right_world), 1.0e-8)

        # Re-orthogonalize forward for a numerically clean right-handed frame.
        forward_world = np.cross(up_world, right_world)
        forward_world /= max(np.linalg.norm(forward_world), 1.0e-8)

        # World -> vehicle coordinates for column vectors.
        R_vw = np.stack([right_world, forward_world, up_world], axis=0).astype(
            np.float32
        )
        self.R_vw_np = R_vw
        self.origin_world = origin_world.astype(np.float32)

        def transform_pose(M):
            out = np.eye(4, dtype=np.float32)
            out[:3, :3] = R_vw @ M[:3, :3]
            out[:3, 3] = R_vw @ (M[:3, 3] - origin_world)
            return out

        self.c2w = np.empty_like(c2w_world, dtype=np.float32)
        for ti in range(self.T):
            for vi in range(self.V):
                self.c2w[ti, vi] = transform_pose(c2w_world[ti, vi])

        self.car_centers = self.c2w[:, :, :3, 3].mean(axis=1)
        self.car_forward = self.c2w[:, 0, :3, 2].copy()
        self.car_forward /= np.maximum(
            np.linalg.norm(self.car_forward, axis=1, keepdims=True), 1.0e-8
        )

        # Transform all Gaussian centers into vehicle coordinates.
        R_vw_t = torch.as_tensor(R_vw, dtype=torch.float32, device=device)
        origin_t = torch.as_tensor(
            origin_world, dtype=torch.float32, device=device
        )
        self.static_means = (self.static_means - origin_t) @ R_vw_t.T

        for ti in range(self.T):
            for vi in range(self.V):
                d = self.dynamic_by_time_cam[ti][vi]
                d["means"] = (d["means"] - origin_t) @ R_vw_t.T

        # Rotating the world frame also rotates every Gaussian orientation.
        q_global_np = vtf.SO3.from_matrix(R_vw).wxyz.astype(np.float32)
        q_global = torch.as_tensor(q_global_np, device=device)

        self.static_quats = quat_mul_wxyz(
            q_global, self.static_quats
        )
        self.static_quats = self.static_quats / self.static_quats.norm(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-8)

        for ti in range(self.T):
            for vi in range(self.V):
                d = self.dynamic_by_time_cam[ti][vi]
                d["quats"] = quat_mul_wxyz(q_global, d["quats"])
                d["quats"] = d["quats"] / d["quats"].norm(
                    dim=-1, keepdim=True
                ).clamp_min(1.0e-8)

        print("[viewer] DGGT world normalized to vehicle frame.")
        print("[viewer] t0 rig center:", self.car_centers[0].tolist())
        print("[viewer] t0 CAM_FRONT forward:", self.car_forward[0].tolist())
        print("[viewer] expected axes: +X right, +Y forward, +Z up")
        print("[viewer] static Gaussians:", f"{self.static_means.shape[0]:,}")
        if self.interpolated_scene:
            print(
                "[viewer] interpolation: OFFICIAL DGGT/TAPIP3D per-camera "
                f"states = {self.T}"
            )
            diag = scene.get("source", {}).get("tracker_diagnostics", [])
            if diag:
                print("[viewer] tracker diagnostics:")
                for d in diag:
                    print("   ", d)
        else:
            print(
                "[viewer] interpolation: mode-2 only (4 true dynamic states)"
            )

        # Dynamic-color diagnostics. If these are not near 1.0, a white-looking
        # object is primarily transparency/background, not missing RGB.
        c0 = self.dynamic_by_time_cam[0][0]["colors"]
        o0 = self.dynamic_by_time_cam[0][0]["opacities"]
        print(
            "[viewer] dynamic CAM_FRONT t0 color mean/min/max:",
            float(c0.mean()), float(c0.min()), float(c0.max()),
        )
        print(
            "[viewer] dynamic CAM_FRONT t0 opacity mean/min/max:",
            float(o0.mean()), float(o0.min()), float(o0.max()),
        )

        # ------------------------------------------------------------
        # Server / initial camera.
        #
        # We deliberately do NOT override scene up-direction here: after the
        # vehicle-frame transform +Z really is up, so viser/nerfview controls
        # behave like the STORM viewer.
        # ------------------------------------------------------------
        self.server = viser.ViserServer(port=port)

        p0 = self.car_centers[0].astype(np.float64)
        p0 = p0 + np.array([0.0, 0.0, 0.10], dtype=np.float64)
        f0 = self.car_forward[0].astype(np.float64)
        f0[2] = 0.0
        f0 /= max(np.linalg.norm(f0), 1.0e-8)

        self.server.initial_camera.position = p0
        self.server.initial_camera.look_at = p0 + self.nav_focus * f0
        self.server.initial_camera.up = np.array([0.0, 0.0, 1.0])
        self.server.initial_camera.fov = self._camera_fov(0, 0)
        self.server.initial_camera.near = max(self.near, 0.01)
        self.server.initial_camera.far = self.far

        self._clients = {}
        self._programmatic_camera_until = 0.0

        @self.server.on_client_connect
        def _client_connected(client):
            key = id(client)
            self._clients[key] = client

            # Give browser initialization a moment before treating camera
            # messages as deliberate user input.
            local_arm_time = time.monotonic() + 1.0

            @client.camera.on_update
            def _camera_changed(_event):
                if time.monotonic() < local_arm_time:
                    return
                if time.monotonic() < self._programmatic_camera_until:
                    return
                if hasattr(self, "follow_ego") and bool(self.follow_ego.value):
                    # User touched the view: stop translating with the car.
                    self.follow_ego.value = False
                    if hasattr(self, "follow_note"):
                        self.follow_note.content = (
                            "**Ego follow:** paused by manual camera movement. "
                            "Click **Resume ego follow** to rejoin the car."
                        )

        # V6 performance caches. Only one assembled scene is retained at a time
        # so we avoid duplicating the full 3M+ Gaussian reconstruction in VRAM.
        self._scene_cache_key = None
        self._scene_cache_bank = None
        self._cached_red_colors_key = None
        self._cached_red_colors = None
        self._last_rig_t_s = None
        self._scene_cache_hits = 0
        self._scene_cache_misses = 0

        self._build_rig()
        self._build_gui(fps=fps, radius_clip=radius_clip)

        # Track previous viewer time for translating clients with ego motion.
        self._last_follow_t_s = self._time_for_frame(0)

        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._render_fn,
            mode="rendering",
        )

    # ------------------------------------------------------------------
    # Time / camera helpers
    # ------------------------------------------------------------------
    def _time_for_frame(self, idx: int) -> float:
        idx = max(0, min(int(idx), self.frames - 1))
        return float(self.timeline_s[idx])

    def _norm_time(self, t_s: float) -> float:
        return float(
            np.interp(
                t_s,
                self.frame_times_s.numpy(),
                self.frame_times_norm.numpy(),
            )
        )

    def _neighbors(self, t_s: float):
        times = self.frame_times_s.numpy()
        if t_s <= times[0]:
            return 0, 0, 0.0
        if t_s >= times[-1]:
            i = len(times) - 1
            return i, i, 0.0
        i1 = int(np.searchsorted(times, t_s))
        i0 = i1 - 1
        a = (t_s - float(times[i0])) / max(
            float(times[i1] - times[i0]), 1.0e-8
        )
        return i0, i1, float(a)

    def _camera_fov(self, ti: int, vi: int) -> float:
        H = int(self.images_grid.shape[-2])
        fy = float(self.K[ti, vi, 1, 1])
        return float(2.0 * np.arctan2(H / 2.0, max(fy, 1.0e-8)))

    def _interpolated_car_center(self, t_s: float):
        i0, i1, a = self._neighbors(t_s)
        return (
            (1.0 - a) * self.car_centers[i0]
            + a * self.car_centers[i1]
        )

    def _translate_clients_with_ego(self, old_t_s: float, new_t_s: float):
        if not bool(self.follow_ego.value):
            self._last_follow_t_s = new_t_s
            return

        old_c = self._interpolated_car_center(old_t_s)
        new_c = self._interpolated_car_center(new_t_s)
        delta = np.asarray(new_c - old_c, dtype=np.float64)

        if float(np.linalg.norm(delta)) < 1.0e-10:
            self._last_follow_t_s = new_t_s
            return

        self._programmatic_camera_until = time.monotonic() + 0.20
        with self.server.atomic():
            for client in self.server.get_clients().values():
                client.camera.position = np.asarray(
                    client.camera.position, dtype=np.float64
                ) + delta
                client.camera.look_at = np.asarray(
                    client.camera.look_at, dtype=np.float64
                ) + delta

        self._last_follow_t_s = new_t_s

    def _snap_all_clients_to_car(self, t_s: float):
        i0, i1, a = self._neighbors(t_s)
        center = self._interpolated_car_center(t_s).astype(np.float64)
        forward = (
            (1.0 - a) * self.car_forward[i0]
            + a * self.car_forward[i1]
        ).astype(np.float64)
        forward[2] = 0.0
        forward /= max(np.linalg.norm(forward), 1.0e-8)

        position = center + np.array([0.0, 0.0, 0.10], dtype=np.float64)
        look_at = position + self.nav_focus * forward

        self._programmatic_camera_until = time.monotonic() + 0.30
        with self.server.atomic():
            for client in self.server.get_clients().values():
                client.camera.position = position
                client.camera.look_at = look_at
                client.camera.up_direction = np.array([0.0, 0.0, 1.0])
                client.camera.near = max(self.near, 0.01)
                client.camera.far = self.far

        self._last_follow_t_s = t_s

    # ------------------------------------------------------------------
    # Rig visualization
    # ------------------------------------------------------------------
    def _build_rig(self):
        self.rig_frustums = []
        poses0 = self.c2w[0]
        _, H, W = self.images_grid.shape[2:]

        for vi, name in enumerate(self.camera_order):
            M = poses0[vi]
            image = np.ascontiguousarray(
                self.images_grid[0, vi].permute(1, 2, 0).numpy()
            )
            h = self.server.scene.add_camera_frustum(
                f"/dggt/current_rig/{name}",
                fov=self._camera_fov(0, vi),
                aspect=float(W) / float(H),
                scale=0.18,
                position=M[:3, 3],
                wxyz=self.vtf.SO3.from_matrix(M[:3, :3]).wxyz,
                image=image,
            )
            self.rig_frustums.append(h)

        center = self.car_centers[0]
        self.rig_center = self.server.scene.add_icosphere(
            "/dggt/current_rig/ego",
            radius=0.12,
            position=center,
        )

        heading_end = center + 1.5 * self.car_forward[0]
        self.heading = self.server.scene.add_line_segments(
            "/dggt/current_rig/forward",
            points=np.asarray([[center, heading_end]], dtype=np.float32),
            colors=(40, 220, 70),
            line_width=4.0,
        )

        pts = self.car_centers
        self.rig_path = None
        if len(pts) >= 2:
            self.rig_path = self.server.scene.add_line_segments(
                "/dggt/ego_path",
                points=np.stack([pts[:-1], pts[1:]], axis=1),
                colors=(255, 170, 40),
                line_width=2.0,
            )

        self._last_thumbnail = -1

    def _update_rig(self, t_s: float):
        i0, i1, a = self._neighbors(t_s)
        centers = []

        for vi, h in enumerate(self.rig_frustums):
            M0 = self.c2w[i0, vi]
            M1 = self.c2w[i1, vi]
            pos = (1.0 - a) * M0[:3, 3] + a * M1[:3, 3]

            Rmix = (1.0 - a) * M0[:3, :3] + a * M1[:3, :3]
            U, _, Vt = np.linalg.svd(Rmix)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt

            h.position = pos
            h.wxyz = self.vtf.SO3.from_matrix(R).wxyz
            centers.append(pos)

        center = np.mean(np.stack(centers), axis=0)
        self.rig_center.position = center

        nearest = int(np.argmin(np.abs(self.frame_times_s.numpy() - t_s)))
        self.heading.points = np.asarray(
            [[center, center + 1.5 * self.car_forward[nearest]]],
            dtype=np.float32,
        )

        if nearest != self._last_thumbnail:
            for vi, h in enumerate(self.rig_frustums):
                h.image = np.ascontiguousarray(
                    self.images_grid[nearest, vi].permute(1, 2, 0).numpy()
                )
            self._last_thumbnail = nearest

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------
    def _build_gui(self, fps: float, radius_clip: float):
        s = self.server

        with s.gui.add_folder("DGGT 4D"):
            self.frame = s.gui.add_slider(
                "Frame",
                min=0,
                max=self.frames - 1,
                step=1,
                initial_value=0,
            )
            self.play = s.gui.add_checkbox("Auto play", initial_value=False)
            self.fps = s.gui.add_slider(
                "Playback FPS",
                min=1.0,
                max=30.0,
                step=1.0,
                initial_value=float(fps),
            )
            self.follow_ego = s.gui.add_checkbox(
                "Follow ego translation",
                initial_value=True,
            )
            self.resume_follow = s.gui.add_button("Resume ego follow")
            self.follow_note = s.gui.add_markdown(
                "**Ego follow:** active. Manually moving/rotating the camera "
                "pauses follow automatically."
            )

        with s.gui.add_folder("Dynamic reconstruction"):
            self.dynamic_policy = s.gui.add_dropdown(
                "Dynamic camera policy",
                ("nearest-view", "manual", "all-six"),
                initial_value="nearest-view",
            )
            self.manual_dynamic_camera = s.gui.add_slider(
                "Manual dynamic camera",
                min=0,
                max=self.V - 1,
                step=1,
                initial_value=0,
            )
            self.crossfade_dynamic = s.gui.add_checkbox(
                "Opacity crossfade between saved states",
                initial_value=False,
            )
            self.dynamic_opacity_boost = s.gui.add_slider(
                "Dynamic opacity boost",
                min=0.1,
                max=3.0,
                step=0.1,
                initial_value=1.0,
            )
            if self.interpolated_scene:
                _dyn_note = (
                    f"**Interpolation:** {self.T} saved DGGT/TAPIP3D states. "
                    "Tracked dynamic clusters have physically interpolated 3D "
                    "positions. Crossfade is optional and only blends opacity "
                    "between already-interpolated neighboring states."
                )
            else:
                _dyn_note = (
                    "**Important:** mode 2 contains only four true dynamic "
                    "states. Crossfade changes opacity only; it does not move "
                    "Gaussians."
                )
            self.dynamic_note = s.gui.add_markdown(_dyn_note)

        with s.gui.add_folder("Gaussian filters"):
            self.show_static = s.gui.add_checkbox(
                "Show static", initial_value=True
            )
            self.show_dynamic = s.gui.add_checkbox(
                "Show dynamic", initial_value=True
            )
            self.min_static_opacity = s.gui.add_slider(
                "Min static base opacity",
                min=0.0,
                max=0.5,
                step=0.005,
                initial_value=0.005,
            )
            self.min_static_conf = s.gui.add_slider(
                "Min static lifespan conf",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=0.0,
            )
            self.min_dynamic_prob = s.gui.add_slider(
                "Min dynamic probability",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=0.0,
            )
            self.min_dynamic_opacity = s.gui.add_slider(
                "Min dynamic opacity",
                min=0.0,
                max=0.5,
                step=0.005,
                initial_value=0.005,
            )
            self.scale_mult = s.gui.add_slider(
                "Scale multiplier",
                min=0.1,
                max=3.0,
                step=0.05,
                initial_value=1.0,
            )
            self.radius_clip = s.gui.add_slider(
                "Radius clip",
                min=0.0,
                max=8.0,
                step=0.25,
                initial_value=float(radius_clip),
            )

        with s.gui.add_folder("Debug"):
            self.dynamic_red = s.gui.add_checkbox(
                "Color dynamic red",
                initial_value=False,
            )
            self.dark_background = s.gui.add_checkbox(
                "Dark background debug",
                initial_value=False,
            )
            self.show_camera_rig = s.gui.add_checkbox(
                "Show camera rig",
                initial_value=False,
            )

        with s.gui.add_folder("Performance"):
            self.fast_interaction = s.gui.add_checkbox(
                "Fast interaction",
                initial_value=True,
            )
            self.preview_radius_clip = s.gui.add_slider(
                "Preview radius clip floor",
                min=0.0,
                max=8.0,
                step=0.25,
                initial_value=float(self.interactive_radius_clip),
            )
            self.performance_note = s.gui.add_markdown(
                "V6 caches the assembled Gaussian bank. During nerfview preview "
                "renders, **Fast interaction** also raises the radius clip to "
                "the preview floor; final/non-preview renders keep the normal "
                "Radius clip setting."
            )

        self.status = s.gui.add_markdown(self._status())

        @self.resume_follow.on_click
        def _resume(_event):
            t_s = self._time_for_frame(int(self.frame.value))
            self._snap_all_clients_to_car(t_s)
            self.follow_ego.value = True
            self.follow_note.content = (
                "**Ego follow:** active. Manually moving/rotating the camera "
                "pauses follow automatically."
            )
            if hasattr(self, "viewer"):
                self.viewer.rerender(None)

        @self.follow_ego.on_update
        def _follow_changed(_event):
            if bool(self.follow_ego.value):
                self._last_follow_t_s = self._time_for_frame(
                    int(self.frame.value)
                )
                self.follow_note.content = "**Ego follow:** active."
            else:
                self.follow_note.content = "**Ego follow:** paused."

        @self.frame.on_update
        def _frame_changed(_event):
            new_t = self._time_for_frame(int(self.frame.value))
            old_t = getattr(self, "_last_follow_t_s", new_t)
            self._translate_clients_with_ego(old_t, new_t)
            self._invalidate_scene_cache()
            self.status.content = self._status()
            if hasattr(self, "viewer"):
                self.viewer.rerender(None)

        rerender = [
            self.dynamic_policy,
            self.manual_dynamic_camera,
            self.crossfade_dynamic,
            self.dynamic_opacity_boost,
            self.show_static,
            self.show_dynamic,
            self.min_static_opacity,
            self.min_static_conf,
            self.min_dynamic_prob,
            self.min_dynamic_opacity,
            self.scale_mult,
            self.radius_clip,
            self.dynamic_red,
            self.dark_background,
            self.fast_interaction,
            self.preview_radius_clip,
        ]

        for h in rerender:
            @h.on_update
            def _changed(_event, self=self):
                self._invalidate_scene_cache()
                self.status.content = self._status()
                if hasattr(self, "viewer"):
                    self.viewer.rerender(None)

        @self.show_camera_rig.on_update
        def _toggle(_event):
            v = bool(self.show_camera_rig.value)
            for h in self.rig_frustums:
                h.visible = v
            self.rig_center.visible = v
            self.heading.visible = v
            if self.rig_path is not None:
                self.rig_path.visible = v

        # Default V6 behavior: hide the rig to reduce websocket/UI work.
        for h in self.rig_frustums:
            h.visible = bool(self.show_camera_rig.value)
        self.rig_center.visible = bool(self.show_camera_rig.value)
        self.heading.visible = bool(self.show_camera_rig.value)
        if self.rig_path is not None:
            self.rig_path.visible = bool(self.show_camera_rig.value)

    def _invalidate_scene_cache(self):
        self._scene_cache_key = None
        self._scene_cache_bank = None
        self._cached_red_colors_key = None
        self._cached_red_colors = None

    def _status(self):
        t = self._time_for_frame(int(self.frame.value))
        i0, i1, a = self._neighbors(t)
        cam = getattr(self, "_last_selected_dynamic_cam", 0)
        cam_name = self.camera_order[cam] if 0 <= cam < self.V else "?"
        if self.interpolated_scene:
            motion_text = (
                "**Motion:** saved DGGT/TAPIP3D interpolation states; "
                "tracked clusters move in 3D between the four original times."
            )
            state_label = "interpolated states"
        else:
            motion_text = (
                "**Motion:** mode-2 dynamic geometry is discrete; only ego "
                "pose and static lifespan are continuously interpolated."
            )
            state_label = "mode-2 groups"

        return (
            f"**Viewer time:** {t:.3f} s  |  {state_label} {i0}/{i1}, "
            f"blend={a:.2f}  \n"
            f"**Static:** {self.static_means.shape[0]:,}  |  "
            f"**Cache:** {self._scene_cache_hits} hit / {self._scene_cache_misses} rebuild  |  "
            f"**Dynamic policy:** "
            f"{self.dynamic_policy.value if hasattr(self, 'dynamic_policy') else 'nearest-view'}  |  "
            f"selected={cam_name}  \n"
            f"{motion_text}"
        )

    # ------------------------------------------------------------------
    # DGGT scene assembly.
    # For DGGT_VIEWER_SCENE_INTERP_V1, every dynamic bank below is already a
    # physically interpolated mode-3 state produced by official interp_all().
    # ------------------------------------------------------------------
    def _static_bank(self, t_norm: float):
        if not bool(self.show_static.value):
            return None

        mask = (
            (self.static_opacities >= float(self.min_static_opacity.value))
            & (self.static_conf >= float(self.min_static_conf.value))
        )
        if not bool(mask.any()):
            return None

        conf = self.static_conf[mask]
        # Exact released DGGT alpha_t form with gamma1 = 0.1.
        sigma = math.log(0.1) / (conf.square() + 1.0e-6)
        temporal = torch.exp(
            sigma * (self.static_source_time[mask] - float(t_norm)).square()
        )

        return {
            "means": self.static_means[mask],
            "colors": self.static_colors[mask],
            "opacities": self.static_opacities[mask] * temporal,
            "scales": self.static_scales[mask],
            "quats": self.static_quats[mask],
            "dynamic": torch.zeros(
                int(mask.sum()),
                dtype=torch.bool,
                device=self.device,
            ),
        }

    def _select_dynamic_camera(self, ti: int, viewer_c2w: torch.Tensor) -> int:
        policy = str(self.dynamic_policy.value)

        if policy == "manual":
            vi = int(self.manual_dynamic_camera.value)
            self._last_selected_dynamic_cam = vi
            return vi

        if policy == "all-six":
            self._last_selected_dynamic_cam = -1
            return -1

        # Nearest input viewing direction. For the same nerfview c2w convention
        # already used by the working STORM viewer, column 2 is forward.
        vf = viewer_c2w[:3, 2].detach().float().cpu().numpy()
        vf /= max(np.linalg.norm(vf), 1.0e-8)

        dirs = self.c2w[ti, :, :3, 2]
        dirs = dirs / np.maximum(
            np.linalg.norm(dirs, axis=1, keepdims=True),
            1.0e-8,
        )
        dots = dirs @ vf
        vi = int(np.argmax(dots))
        self._last_selected_dynamic_cam = vi
        return vi

    def _dynamic_single(self, ti: int, vi: int, weight: float):
        if not bool(self.show_dynamic.value) or weight <= 0.0:
            return None

        d = self.dynamic_by_time_cam[ti][vi]
        mask = (
            (d["dynamic_prob"] >= float(self.min_dynamic_prob.value))
            & (d["opacities"] >= float(self.min_dynamic_opacity.value))
        )
        if not bool(mask.any()):
            return None

        return {
            "means": d["means"][mask],
            "colors": d["colors"][mask],
            "opacities": (
                d["opacities"][mask]
                * float(weight)
                * float(self.dynamic_opacity_boost.value)
            ),
            "scales": d["scales"][mask],
            "quats": d["quats"][mask],
            "dynamic": torch.ones(
                int(mask.sum()),
                dtype=torch.bool,
                device=self.device,
            ),
        }

    def _dynamic_at_time(
        self,
        ti: int,
        viewer_c2w: torch.Tensor,
        weight: float,
    ):
        vi = self._select_dynamic_camera(ti, viewer_c2w)

        if vi >= 0:
            return [self._dynamic_single(ti, vi, weight)]

        # Debug/fallback policy: all six. This is NOT the released DGGT
        # per-render behavior and can create duplicate/blurred moving objects.
        out = []
        for cam in range(self.V):
            x = self._dynamic_single(ti, cam, weight)
            if x is not None:
                out.append(x)
        return out

    def _scene_at(self, t_s: float, viewer_c2w: torch.Tensor):
        # Build a compact signature first. Viewer camera motion only invalidates
        # the bank when nearest-view crosses into a different DGGT camera.
        i0, i1, a = self._neighbors(t_s)
        if bool(self.crossfade_dynamic.value) and i0 != i1:
            dyn_specs = [
                (i0, self._select_dynamic_camera(i0, viewer_c2w), float(1.0 - a)),
                (i1, self._select_dynamic_camera(i1, viewer_c2w), float(a)),
            ]
        else:
            nearest = int(np.argmin(np.abs(self.frame_times_s.numpy() - t_s)))
            dyn_specs = [
                (nearest, self._select_dynamic_camera(nearest, viewer_c2w), 1.0)
            ]

        key = (
            round(float(t_s), 7),
            tuple((int(ti), int(vi), round(float(w), 7)) for ti, vi, w in dyn_specs),
            bool(self.show_static.value),
            bool(self.show_dynamic.value),
            round(float(self.min_static_opacity.value), 6),
            round(float(self.min_static_conf.value), 6),
            round(float(self.min_dynamic_prob.value), 6),
            round(float(self.min_dynamic_opacity.value), 6),
            round(float(self.dynamic_opacity_boost.value), 6),
        )

        if self._scene_cache_key == key and self._scene_cache_bank is not None:
            self._scene_cache_hits += 1
            return self._scene_cache_bank

        self._scene_cache_misses += 1
        banks = []

        st = self._static_bank(self._norm_time(t_s))
        if st is not None:
            banks.append(st)

        for ti, vi, weight in dyn_specs:
            if vi >= 0:
                x = self._dynamic_single(ti, vi, weight)
                if x is not None:
                    banks.append(x)
            else:
                # all-six debug mode
                for cam in range(self.V):
                    x = self._dynamic_single(ti, cam, weight)
                    if x is not None:
                        banks.append(x)

        if not banks:
            bank = None
        else:
            bank = {
                k: torch.cat([b[k] for b in banks], dim=0)
                for k in (
                    "means",
                    "colors",
                    "opacities",
                    "scales",
                    "quats",
                    "dynamic",
                )
            }

        self._scene_cache_key = key
        self._scene_cache_bank = bank
        self._cached_red_colors_key = None
        self._cached_red_colors = None
        return bank

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _render_fn(self, camera_state, render_arg):
        width, height = resolve_render_size(render_arg)

        c2w = torch.as_tensor(
            camera_state.c2w,
            dtype=torch.float32,
            device=self.device,
        )
        if c2w.shape == (3, 4):
            c2w = torch.cat(
                [
                    c2w,
                    torch.tensor(
                        [[0.0, 0.0, 0.0, 1.0]],
                        device=self.device,
                    ),
                ],
                dim=0,
            )

        viewmat = torch.linalg.inv(c2w)
        K = torch.as_tensor(
            camera_state.get_K((width, height)),
            dtype=torch.float32,
            device=self.device,
        )

        t_s = self._time_for_frame(int(self.frame.value))
        if self._last_rig_t_s is None or abs(float(t_s) - float(self._last_rig_t_s)) > 1.0e-9:
            self._update_rig(t_s)
            self._last_rig_t_s = float(t_s)
        bank = self._scene_at(t_s, c2w)

        if bank is None:
            bg = 0.08 if bool(self.dark_background.value) else 1.0
            return np.full((height, width, 3), bg, dtype=np.float32)

        colors = bank["colors"]
        if bool(self.dynamic_red.value):
            red_key = id(bank["colors"])
            if self._cached_red_colors_key != red_key or self._cached_red_colors is None:
                red_colors = colors.clone()
                red_colors[bank["dynamic"]] = torch.tensor(
                    [1.0, 0.08, 0.03],
                    dtype=colors.dtype,
                    device=self.device,
                )
                self._cached_red_colors_key = red_key
                self._cached_red_colors = red_colors
            colors = self._cached_red_colors

        is_preview = bool(getattr(render_arg, "preview_render", False))
        effective_radius_clip = float(self.radius_clip.value)
        if is_preview and bool(self.fast_interaction.value):
            effective_radius_clip = max(
                effective_radius_clip,
                float(self.preview_radius_clip.value),
            )

        render, alpha, _ = self.rasterization(
            means=bank["means"],
            quats=bank["quats"],
            scales=bank["scales"] * float(self.scale_mult.value),
            opacities=bank["opacities"],
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=width,
            height=height,
            near_plane=self.near,
            far_plane=self.far,
            packed=True,
            radius_clip=effective_radius_clip,
            sh_degree=None,
            rasterize_mode="classic",
            backgrounds=None,
        )

        rgb = render[0, ..., :3].float()
        a = alpha[0, ..., :1].float()

        bg = 0.08 if bool(self.dark_background.value) else 1.0
        rgb = (rgb + (1.0 - a) * bg).clamp(0.0, 1.0)
        return rgb.cpu().numpy()

    def run(self):
        print("[viewer] Open the Viser URL printed above.")
        print("[viewer] Navigation is standard +Z-up viser/nerfview, like STORM.")
        print("[viewer] V6 scene-bank cache enabled; camera motion reuses the current bank.")
        print(
            "[viewer] Fast interaction preview radius floor:",
            self.interactive_radius_clip,
        )
        if self.interpolated_scene:
            print(
                f"[viewer] Using {self.T} exported DGGT/TAPIP3D interpolation states."
            )
        print(
            "[viewer] Manual camera movement pauses ego-follow; "
            "use 'Resume ego follow' to rejoin."
        )
        print("[viewer] Ctrl+C to stop.")

        last = time.perf_counter()
        while True:
            time.sleep(0.01)
            if bool(self.play.value):
                now = time.perf_counter()
                if now - last >= 1.0 / max(float(self.fps.value), 1.0):
                    self.frame.value = (
                        int(self.frame.value) + 1
                    ) % self.frames
                    last = now


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--frames",
        type=int,
        default=0,
        help=(
            "0 = use the saved temporal states exactly. A positive value "
            "resamples the viewer timeline but does not create new DGGT states."
        ),
    )
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--near", type=float, default=0.2)
    p.add_argument("--far", type=float, default=400.0)
    p.add_argument("--radius-clip", type=float, default=0.0)
    p.add_argument(
        "--interactive-radius-clip",
        type=float,
        default=1.0,
        help=(
            "Radius-clip floor used only for nerfview preview renders when "
            "Fast interaction is enabled. 1.0 is a good starting point for "
            "multi-million-Gaussian scenes."
        ),
    )
    p.add_argument(
        "--nav-focus",
        type=float,
        default=2.5,
        help=(
            "Initial orbit/look-at distance in meters. Smaller values make "
            "mouse translation/orbit feel less sensitive."
        ),
    )
    args = p.parse_args()

    device = torch.device(
        args.device
        if torch.cuda.is_available() or not args.device.startswith("cuda")
        else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print("[viewer] GPU:", torch.cuda.get_device_name(device))

    viewer = DGGT4DViewer(
        scene=torch_load(args.scene),
        device=device,
        port=args.port,
        frames=args.frames,
        fps=args.fps,
        near=args.near,
        far=args.far,
        radius_clip=args.radius_clip,
        nav_focus=args.nav_focus,
        interactive_radius_clip=args.interactive_radius_clip,
    )
    viewer.run()


if __name__ == "__main__":
    main()
