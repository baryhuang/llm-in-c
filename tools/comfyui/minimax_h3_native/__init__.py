"""ComfyUI front-end node for cpullama's native MiniMax-H3 C/Metal runner.

Python only marshals the IMAGE tensor to PNG and launches the native binary;
model loading and inference stay entirely in the C/Objective-C/Metal runtime.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
from PIL import Image


_RUN_LOCK = threading.Lock()


def _repository() -> Path:
    configured = os.environ.get("CPULLAMA_REPOSITORY")
    if configured:
        root = Path(configured).expanduser().resolve()
        if (root / "build/minimax-h3-m3-e2e").is_file():
            return root
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "build/minimax-h3-m3-e2e").is_file():
            return parent
    raise RuntimeError(
        "Cannot locate cpullama. Set CPULLAMA_REPOSITORY to its absolute path."
    )


def _comfy_directories() -> tuple[Path, Path]:
    try:
        import folder_paths

        return Path(folder_paths.get_input_directory()), Path(
            folder_paths.get_output_directory()
        )
    except Exception:
        shared = Path.home() / "ComfyUI-Shared"
        return shared / "input", shared / "output"


class MiniMaxH3NativeFL2VA:
    """Run MiniMax-H3 image-to-video+audio with the native Apple Silicon binary."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Animate this image with natural motion and matching ambient sound.",
                    },
                ),
                "condition_at": (["first frame", "last frame"],),
                "width": (
                    "INT",
                    {"default": 128, "min": 32, "max": 1280, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 128, "min": 32, "max": 1280, "step": 32},
                ),
                "frames": (
                    "INT",
                    {"default": 22, "min": 22, "max": 362, "step": 17},
                ),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 0x7FFFFFFFFFFFFFFF},
                ),
                "turbo_4_step": ("BOOLEAN", {"default": True}),
                "output_name": (
                    "STRING",
                    {"default": "minimax-h3-native"},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("mp4_path", "metrics_json")
    FUNCTION = "generate"
    CATEGORY = "cpullama/native"
    OUTPUT_NODE = True

    def generate(
        self,
        image,
        prompt: str,
        condition_at: str,
        width: int,
        height: int,
        frames: int,
        seed: int,
        turbo_4_step: bool,
        output_name: str,
    ):
        if frames < 22 or (frames - 5) % 17:
            raise ValueError("MiniMax-H3 frames must be 17*n+5 (22, 39, 56, ...).")
        if width % 32 or height % 32:
            raise ValueError("MiniMax-H3 width and height must be divisible by 32.")
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        repository = _repository()
        input_root, output_root = _comfy_directories()
        stamp = f"{int(time.time())}-{time.time_ns() % 1_000_000_000:09d}"
        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in output_name.strip()
        ).strip("-") or "minimax-h3-native"
        input_dir = input_root / "minimax-h3-native"
        output_dir = output_root / safe_name / stamp
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = None
        if image is not None:
            image_path = input_dir / f"condition-{stamp}.png"
            pixels = image[0].detach().cpu().numpy()
            pixels = np.clip(np.rint(pixels * 255.0), 0, 255).astype(np.uint8)
            if pixels.shape[-1] == 1:
                pixels = np.repeat(pixels, 3, axis=-1)
            Image.fromarray(pixels[..., :3], mode="RGB").save(image_path)

        runner = repository / "build/minimax-h3-m3-e2e"
        text_encoder = repository / "tmp/minimax-h3-text-encoder.safetensors"
        turbo_adapter = repository / "tmp/minimax_h3_turbo_v4_step600_ema.safetensors"
        required = [runner, text_encoder]
        if turbo_4_step:
            required.append(turbo_adapter)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("Missing native MiniMax-H3 assets: " + ", ".join(missing))

        command = [str(runner)]
        if image_path is not None:
            flag = "--first-image" if condition_at == "first frame" else "--last-image"
            command.extend([flag, str(image_path)])
        command.extend(
            [
                prompt,
                str(output_dir),
                str(width),
                str(height),
                str(frames),
                str(seed),
            ]
        )
        environment = os.environ.copy()
        environment["MINIMAX_H3_TEXT_ENCODER_URL"] = f"file://{text_encoder}"
        environment["MINIMAX_H3_TREE_ATTENTION"] = "0"
        environment["MINIMAX_H3_LORA_MMA"] = "0"
        if turbo_4_step:
            environment["MINIMAX_H3_TURBO_ADAPTER"] = f"file://{turbo_adapter}"
        else:
            environment.pop("MINIMAX_H3_TURBO_ADAPTER", None)

        with _RUN_LOCK:
            completed = subprocess.run(
                command,
                cwd=repository,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        log_path = output_dir / "native-run.log"
        log_path.write_text(completed.stderr + completed.stdout, encoding="utf-8")
        if completed.returncode:
            tail = "\n".join((completed.stderr or completed.stdout).splitlines()[-30:])
            raise RuntimeError(
                f"Native MiniMax-H3 failed with exit code {completed.returncode}.\n{tail}\n"
                f"Full log: {log_path}"
            )
        try:
            metrics = json.loads(completed.stdout)
        except json.JSONDecodeError as exception:
            raise RuntimeError(f"Native runner returned invalid JSON; see {log_path}") from exception
        mp4_path = Path(metrics["output_path"]).resolve()
        if not mp4_path.is_file():
            raise RuntimeError(f"Native runner did not create {mp4_path}; see {log_path}")
        metrics["native_log_path"] = str(log_path)
        rendered_metrics = json.dumps(metrics, ensure_ascii=False, indent=2)
        try:
            relative_mp4 = mp4_path.relative_to(output_root.resolve())
        except ValueError as exception:
            raise RuntimeError(
                f"Native output {mp4_path} is outside ComfyUI output directory {output_root}"
            ) from exception
        preview = {
            "filename": relative_mp4.name,
            "subfolder": "" if relative_mp4.parent == Path(".") else relative_mp4.parent.as_posix(),
            "type": "output",
            "format": "video/mp4",
        }
        return {
            "ui": {
                "images": [preview],
                "animated": (True,),
                "text": [str(mp4_path), rendered_metrics],
            },
            "result": (str(mp4_path), rendered_metrics),
        }


class MiniMaxH3NativeT2VA:
    """Generate video and audio from text with the native Apple Silicon binary."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A cinematic scene with natural motion and matching ambient sound.",
                    },
                ),
                "width": (
                    "INT",
                    {"default": 128, "min": 32, "max": 1280, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 128, "min": 32, "max": 1280, "step": 32},
                ),
                "frames": (
                    "INT",
                    {"default": 22, "min": 22, "max": 362, "step": 17},
                ),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 0x7FFFFFFFFFFFFFFF},
                ),
                "turbo_4_step": ("BOOLEAN", {"default": True}),
                "output_name": ("STRING", {"default": "minimax-h3-native-text"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("mp4_path", "metrics_json")
    FUNCTION = "generate"
    CATEGORY = "cpullama/native"
    OUTPUT_NODE = True

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        frames: int,
        seed: int,
        turbo_4_step: bool,
        output_name: str,
    ):
        return MiniMaxH3NativeFL2VA().generate(
            None,
            prompt,
            "first frame",
            width,
            height,
            frames,
            seed,
            turbo_4_step,
            output_name,
        )


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3NativeFL2VA": MiniMaxH3NativeFL2VA,
    "MiniMaxH3NativeT2VA": MiniMaxH3NativeT2VA,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3NativeFL2VA": "MiniMax-H3 Native FL2VA (C/Metal)",
    "MiniMaxH3NativeT2VA": "MiniMax-H3 Native Text to Video+Audio (C/Metal)",
}
