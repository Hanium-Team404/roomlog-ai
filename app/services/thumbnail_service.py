import asyncio
import base64
import copy
import gc
import io
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger("app")

_openai = AsyncOpenAI(api_key=settings.openai_api_key)
_PROMPT = (Path(__file__).parent.parent / "prompts" / "room_thumbnail.txt").read_text(encoding="utf-8")

_W, _H = 960, 540


def _render_offscreen(mesh: o3d.geometry.TriangleMesh) -> list[bytes]:
    renderer = o3d.visualization.rendering.OffscreenRenderer(_W, _H)
    try:
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

        mat = o3d.visualization.rendering.MaterialRecord()
        if mesh.has_vertex_colors():
            mat.shader = "defaultUnlit"
        else:
            mat.shader = "defaultLit"
            mat.base_color = [0.7, 0.7, 0.7, 1.0]

        renderer.scene.scene.set_sun_light([0.577, -0.577, -0.577], [1.0, 1.0, 1.0], 75000)
        renderer.scene.scene.enable_sun_light(True)

        bounds = mesh.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        extent = np.linalg.norm(bounds.get_extent())

        mesh.compute_triangle_normals()
        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        tnorm = np.asarray(mesh.triangle_normals)
        tri_centers = verts[tris].mean(axis=1)

        frames = []
        for i in range(4):
            az = np.radians(45 + 90 * i)
            el = np.radians(40)
            # ARKit/StrayScanner world frame is Y-up
            eye = center + extent * 1.2 * np.array([
                np.cos(el) * np.cos(az),
                np.sin(el),
                np.cos(el) * np.sin(az),
            ])

            # 이 시점에서 카메라를 향하는 face 제거 (dollhouse view)
            cam_dirs = eye - tri_centers
            cam_dirs /= np.linalg.norm(cam_dirs, axis=1, keepdims=True) + 1e-9
            dot = np.einsum('ij,ij->i', tnorm, cam_dirs)

            view_mesh = copy.deepcopy(mesh)
            view_mesh.remove_triangles_by_mask((dot < 0.0).tolist())
            view_mesh.remove_unreferenced_vertices()
            view_mesh.compute_vertex_normals()

            if i > 0:
                renderer.scene.remove_geometry("mesh")
            renderer.scene.add_geometry("mesh", view_mesh, mat)

            renderer.setup_camera(60.0, center, eye, [0.0, 1.0, 0.0])
            img = renderer.render_to_image()
            buf = io.BytesIO()
            Image.fromarray(np.asarray(img)).save(buf, format="PNG")
            frames.append(buf.getvalue())
            buf.close()
            del view_mesh

        return frames
    finally:
        # Filament 엔진은 생성한 스레드에서 소멸해야 함 (asyncio to_thread 크래시 방지)
        del renderer
        gc.collect()


def _render_visualizer(mesh: o3d.geometry.TriangleMesh) -> list[bytes]:
    bounds = mesh.get_axis_aligned_bounding_box()
    center = bounds.get_center().tolist()

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=_W, height=_H)
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.light_on = True

    frames = []
    for i in range(4):
        az = np.radians(45 + 90 * i)
        el = np.radians(40)
        front = [
            np.cos(el) * np.cos(az),
            np.sin(el),
            np.cos(el) * np.sin(az),
        ]

        ctr = vis.get_view_control()
        ctr.set_lookat(center)
        ctr.set_up([0.0, 1.0, 0.0])
        ctr.set_front(front)
        ctr.set_zoom(0.5)

        vis.poll_events()
        vis.update_renderer()

        fd, tmp_str = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        tmp = Path(tmp_str)
        vis.capture_screen_image(str(tmp), do_render=True)
        frames.append(tmp.read_bytes())
        tmp.unlink()

    vis.destroy_window()
    return frames


def get_render_frames(mesh_path: Path) -> list[bytes]:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0 or len(mesh.vertices) == 0:
        raise ValueError("mesh가 비어있음")
    mesh.compute_vertex_normals()
    logger.info(
        "[Thumbnail] 렌더링 시작 — vertices=%d triangles=%d EGL_PLATFORM=%s",
        len(mesh.vertices),
        len(mesh.triangles),
        os.environ.get("EGL_PLATFORM", "미설정"),
    )
    try:
        frames = _render_offscreen(mesh)
        logger.info("[Thumbnail] OffscreenRenderer 성공 — %d frames", len(frames))
        return frames
    except Exception as e:
        logger.warning("[Thumbnail] OffscreenRenderer 실패 (%s: %s) — Visualizer 폴백", type(e).__name__, e)
        try:
            frames = _render_visualizer(mesh)
            logger.info("[Thumbnail] Visualizer 폴백 성공 — %d frames", len(frames))
            return frames
        except Exception as e2:
            logger.error("[Thumbnail] Visualizer 폴백도 실패 (%s: %s)", type(e2).__name__, e2)
            raise
    finally:
        del mesh
        gc.collect()


async def generate_thumbnail(frames: list[bytes]) -> bytes:
    logger.info("[Thumbnail] gpt-image-2.5-flare 생성 요청 (frames=%d)", len(frames))

    response = await _openai.images.edit(
        model="gpt-image-2.5-flare",
        image=[
            (f"view_{i}.png", frame, "image/png")
            for i, frame in enumerate(frames)
        ],
        prompt=_PROMPT,
        size="1024x1024",
    )

    image_bytes = base64.b64decode(response.data[0].b64_json)
    logger.info("[Thumbnail] 생성 완료 size=%d bytes", len(image_bytes))
    return image_bytes
