# adapted from sglang and fastvideo
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark online serving for diffusion models (Image/Video Generation).
If you want to use i2v, i2i dataset, you should `uv pip install gdown` first

Usage:
    # Video
    t2v:
    python3 benchmarks/diffusion_benchmark_serving.py \
        --dataset vbench --task t2v --num-prompts 10 \
        --height 480 --width 640 --fps 16 --num-frames 80

    i2v:
    python3 benchmarks/diffusion_benchmark_serving.py \
        --dataset vbench --task i2v --num-prompts 10


    # Image
    t2i:
    python3 benchmarks/diffusion_benchmark_serving.py \
        --dataset vbench --task t2i --num-prompts 10 \
        --height 1024 --width 1024

    i2i:
    python3 benchmarks/diffusion_benchmark_serving.py \
        --dataset vbench --task i2i --num-prompts 10

"""

import argparse
import ast
import asyncio
import base64
import glob
import json
import mimetypes
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field, replace
from typing import Any

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    model: str
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    num_inference_steps: int | None = None
    seed: int | None = None
    fps: int | None = None
    timestamp: float | None = None
    slo_ms: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    image_paths: list[str] | None = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RequestFuncOutput:
    success: bool = False
    latency: float = 0.0
    error: str = ""
    start_time: float = 0.0
    response_body: dict[str, Any] = field(default_factory=dict)
    peak_memory_mb: float = 0.0
    measured_latency_s: float | None = None
    slo_ms: float | None = None
    slo_achieved: bool | None = None


class BaseDataset(ABC):
    def __init__(self, args, api_url: str, model: str):
        self.args = args
        self.api_url = api_url
        self.model = model

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> RequestFuncInput:
        pass

    @abstractmethod
    def get_requests(self) -> list[RequestFuncInput]:
        pass


class VBenchDataset(BaseDataset):
    """
    Dataset loader for VBench prompts.
    Supports t2v, i2v.
    """

    T2V_PROMPT_URL = (
        "https://raw.githubusercontent.com/Vchitect/VBench/master/prompts/prompts_per_dimension/subject_consistency.txt"
    )
    I2V_DOWNLOAD_SCRIPT_URL = (
        "https://raw.githubusercontent.com/Vchitect/VBench/master/vbench2_beta_i2v/download_data.sh"
    )

    def __init__(self, args, api_url: str, model: str):
        super().__init__(args, api_url, model)
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "vllm-omni")
        self.items = self._load_data()

    def _load_data(self) -> list[dict[str, Any]]:
        if self.args.task == "t2v":
            return self._load_t2v_prompts()
        elif self.args.task in ["i2v", "ti2v", "ti2i", "i2i"]:
            return self._load_i2v_data()
        else:
            return self._load_t2v_prompts()

    def _download_file(self, url: str, dest_path: str) -> None:
        """Download a file from URL to destination path."""
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        resp = requests.get(url)
        resp.raise_for_status()
        with open(dest_path, "w") as f:
            f.write(resp.text)

    def _load_t2v_prompts(self) -> list[dict[str, Any]]:
        path = self.args.dataset_path

        if not path:
            path = os.path.join(self.cache_dir, "vbench_subject_consistency.txt")
            if not os.path.exists(path):
                print(f"Downloading VBench T2V prompts to {path}...")
                try:
                    self._download_file(self.T2V_PROMPT_URL, path)
                except Exception as e:
                    print(f"Failed to download VBench prompts: {e}")
                    return [{"prompt": "A cat sitting on a bench"}] * 50

        prompts = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append({"prompt": line})

        return self._resize_data(prompts)

    def _auto_download_i2v_dataset(self) -> str:
        """Auto-download VBench I2V dataset and return the dataset directory."""
        vbench_i2v_dir = os.path.join(self.cache_dir, "vbench_i2v", "vbench2_beta_i2v")
        info_json_path = os.path.join(vbench_i2v_dir, "data", "i2v-bench-info.json")

        if os.path.exists(info_json_path):
            return vbench_i2v_dir

        print(f"Downloading VBench I2V dataset to {vbench_i2v_dir}...")
        try:
            cache_root = os.path.join(self.cache_dir, "vbench_i2v")
            script_path = os.path.join(cache_root, "download_data.sh")

            self._download_file(self.I2V_DOWNLOAD_SCRIPT_URL, script_path)
            os.chmod(script_path, 0o755)

            print("Executing download_data.sh (this may take a while)...")
            import subprocess

            result = subprocess.run(
                ["bash", script_path],
                cwd=cache_root,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Download script failed: {result.stderr}")

            print(f"Successfully downloaded VBench I2V dataset to {vbench_i2v_dir}")
        except Exception as e:
            print(f"Failed to download VBench I2V dataset: {e}")
            print("Please manually download following instructions at:")
            print("https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v#22-download")
            return None

        return vbench_i2v_dir if os.path.exists(info_json_path) else None

    def _load_from_i2v_json(self, json_path: str) -> list[dict[str, Any]]:
        """Load I2V data from i2v-bench-info.json format."""
        with open(json_path) as f:
            items = json.load(f)

        base_dir = os.path.dirname(os.path.dirname(json_path))  # Go up to vbench2_beta_i2v
        origin_dir = os.path.join(base_dir, "data", "origin")

        data = []
        for item in items:
            img_path = os.path.join(origin_dir, item.get("file_name", ""))
            if os.path.exists(img_path):
                data.append({"prompt": item.get("caption", ""), "image_path": img_path})
            else:
                print(f"Warning: Image not found: {img_path}")

        print(f"Loaded {len(data)} I2V samples from VBench I2V dataset")
        return data

    def _scan_directory_for_images(self, path: str) -> list[dict[str, Any]]:
        """Scan directory for image files."""
        exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        files = []

        for ext in exts:
            files.extend(glob.glob(os.path.join(path, ext)))
            files.extend(glob.glob(os.path.join(path, ext.upper())))

            # Also check in data/origin subdirectory
            origin_dir = os.path.join(path, "data", "origin")
            if os.path.exists(origin_dir):
                files.extend(glob.glob(os.path.join(origin_dir, ext)))
                files.extend(glob.glob(os.path.join(origin_dir, ext.upper())))

        return [{"prompt": os.path.splitext(os.path.basename(f))[0], "image_path": f} for f in files]

    def _create_dummy_data(self) -> list[dict[str, Any]]:
        """Create dummy data with a placeholder image in cache directory."""
        print("No I2V data found. Using dummy placeholders.")

        dummy_image = os.path.join(self.cache_dir, "dummy_image.jpg")
        if not os.path.exists(dummy_image):
            try:
                from PIL import Image

                os.makedirs(self.cache_dir, exist_ok=True)
                img = Image.new("RGB", (100, 100), color="red")
                img.save(dummy_image)
                print(f"Created dummy image at {dummy_image}")
            except ImportError:
                print("PIL not installed, cannot create dummy image.")
                return []

        return [{"prompt": "A moving cat", "image_path": dummy_image}] * 10

    def _load_i2v_data(self) -> list[dict[str, Any]]:
        """Load I2V data from VBench I2V dataset or user-provided path."""
        path = self.args.dataset_path

        # Auto-download if no path provided
        if not path:
            path = self._auto_download_i2v_dataset()
            if not path:
                return self._resize_data(self._create_dummy_data())

        # Try to load from i2v-bench-info.json
        info_json_candidates = [
            os.path.join(path, "data", "i2v-bench-info.json"),
            path if path.endswith(".json") else None,
        ]

        for json_path in info_json_candidates:
            if json_path and os.path.exists(json_path):
                try:
                    return self._resize_data(self._load_from_i2v_json(json_path))
                except Exception as e:
                    print(f"Failed to load {json_path}: {e}")

        # Fallback: scan directory for images
        if os.path.isdir(path):
            data = self._scan_directory_for_images(path)
            if data:
                return self._resize_data(data)

        # Last resort: dummy data
        return self._resize_data(self._create_dummy_data())

    def _resize_data(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Resize data to match num_prompts."""
        if not self.args.num_prompts:
            return data

        if len(data) < self.args.num_prompts:
            factor = (self.args.num_prompts // len(data)) + 1
            data = data * factor

        return data[: self.args.num_prompts]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestFuncInput:
        item = self.items[idx]
        image_paths = [item["image_path"]] if "image_path" in item else None

        return RequestFuncInput(
            prompt=item.get("prompt", ""),
            api_url=self.api_url,
            model=self.model,
            width=self.args.width,
            height=self.args.height,
            num_frames=self.args.num_frames,
            num_inference_steps=self.args.num_inference_steps,
            seed=self.args.seed,
            fps=self.args.fps,
            image_paths=image_paths,
        )

    def get_requests(self) -> list[RequestFuncInput]:
        return [self[i] for i in range(len(self))]


class TraceDataset(BaseDataset):
    """Trace-based dataset loader for heterogeneous diffusion requests."""

    DEFAULT_REPO_ID = "asukaqaqzz/Dit_Trace"
    DEFAULT_FILENAME = "sd3_trace.txt"

    def __init__(self, args, api_url: str, model: str):
        super().__init__(args, api_url, model)
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "vllm-omni", "trace")
        dataset_root = args.dataset_path
        if not dataset_root:
            dataset_root = self._download_default_trace()
        self.items = self._load_items(dataset_root)

    @staticmethod
    def _coerce_int(x: Any) -> int | None:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        try:
            s = str(x).strip()
            if not s:
                return None
            return int(float(s))
        except Exception:
            return None

    @staticmethod
    def _coerce_float(x: Any) -> float | None:
        if x is None:
            return None
        if isinstance(x, float):
            return x
        if isinstance(x, int):
            return float(x)
        try:
            s = str(x).strip()
            if not s:
                return None
            return float(s)
        except Exception:
            return None

    def _download_default_trace(self) -> str:
        """Download default trace file from HuggingFace Hub if not provided."""

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to download the default trace dataset. "
                "Install via `pip install huggingface_hub`."
            ) from exc

        os.makedirs(self.cache_dir, exist_ok=True)
        return hf_hub_download(
            repo_id=self.DEFAULT_REPO_ID,
            filename=self.DEFAULT_FILENAME,
            repo_type="dataset",
            local_dir=self.cache_dir,
            local_dir_use_symlinks=False,
        )

    def _expand_paths(self, dataset_path: str | None) -> list[str]:
        if not dataset_path:
            return []

        parts = [p.strip() for p in str(dataset_path).split(",") if p.strip()]
        paths: list[str] = []
        for p in parts:
            if any(ch in p for ch in ["*", "?", "["]):
                paths.extend(sorted(glob.glob(p)))
            elif os.path.isdir(p):
                paths.extend(sorted(glob.glob(os.path.join(p, "**", "*.txt"), recursive=True)))
            else:
                paths.append(p)

        seen = set()
        unique_paths = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)
        return unique_paths

    def _parse_trace_file(self, path: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        def parse_request_repr_line(line: str) -> dict[str, Any] | None:
            text = line.strip()
            if not text:
                return None
            if not (text.startswith("Request(") and text.endswith(")")):
                return None
            inner = text[len("Request(") : -1]
            try:
                expr = ast.parse(f"f({inner})", mode="eval")
                if not isinstance(expr.body, ast.Call):
                    return None
                call = expr.body
                out: dict[str, Any] = {}
                for kw in call.keywords:
                    if kw.arg is None:
                        continue
                    out[kw.arg] = ast.literal_eval(kw.value)
                return out
            except Exception:
                return None

        # detect first non-empty line to pick parser
        first_non_empty = None
        with open(path, encoding="utf-8") as f:
            for _ in range(50):
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    first_non_empty = line.strip()
                    f.seek(pos)
                    break

        if first_non_empty is None:
            return rows

        if first_non_empty.startswith("Request("):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parsed = parse_request_repr_line(line)
                    if isinstance(parsed, dict):
                        rows.append(parsed)
            return rows

        # txt fallback: parse Request(...) lines only
        with open(path, encoding="utf-8") as f:
            for line in f:
                parsed = parse_request_repr_line(line)
                if isinstance(parsed, dict):
                    rows.append(parsed)
        return rows

    def _load_items(self, dataset_root: str) -> list[dict[str, Any]]:
        paths = self._expand_paths(dataset_root)
        if not paths:
            raise ValueError("No trace files found. Provide --dataset-path or rely on default HuggingFace download.")

        items: list[dict[str, Any]] = []
        for p in paths:
            if not os.path.exists(p):
                continue
            for row in self._parse_trace_file(p):
                if isinstance(row, dict):
                    row = dict(row)
                    row.setdefault("_source", p)
                    items.append(row)

        if not items:
            raise ValueError("Trace dataset is empty after parsing provided paths.")

        if self.args.num_prompts is not None:
            items = items[: self.args.num_prompts]

        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestFuncInput:
        row = self.items[idx]
        prompt = row.get("prompt") or row.get("text") or ""

        row_height = self._coerce_int(row.get("height"))
        row_width = self._coerce_int(row.get("width"))
        num_frames = self._coerce_int(row.get("num_frames"))
        num_steps = self._coerce_int(row.get("num_inference_steps"))
        seed = self._coerce_int(row.get("seed"))
        fps = self._coerce_int(row.get("fps"))
        timestamp = self._coerce_float(row.get("timestamp"))
        slo_ms = self._coerce_float(row.get("slo_ms"))

        override_w = self.args.width
        override_h = self.args.height
        if override_w is not None or override_h is not None:
            width = override_w
            height = override_h
        else:
            width = row_width
            height = row_height

        return RequestFuncInput(
            prompt=str(prompt),
            api_url=self.api_url,
            model=self.model,
            width=width,
            height=height,
            num_frames=num_frames if num_frames is not None else self.args.num_frames,
            num_inference_steps=num_steps if num_steps is not None else self.args.num_inference_steps,
            seed=seed if seed is not None else self.args.seed,
            fps=fps if fps is not None else self.args.fps,
            timestamp=timestamp,
            slo_ms=slo_ms,
            request_id=str(row.get("request_id")) if row.get("request_id") is not None else str(uuid.uuid4()),
        )

    def get_requests(self) -> list[RequestFuncInput]:
        return [self[i] for i in range(len(self))]


class RandomDataset(BaseDataset):
    def __init__(self, args, api_url: str, model: str):
        self.args = args
        self.api_url = api_url
        self.model = model
        self.num_prompts = args.num_prompts

    def __len__(self) -> int:
        return self.num_prompts

    def __getitem__(self, idx: int) -> RequestFuncInput:
        return RequestFuncInput(
            prompt=f"Random prompt {idx} for benchmarking diffusion models",
            api_url=self.api_url,
            model=self.model,
            width=self.args.width,
            height=self.args.height,
            num_frames=self.args.num_frames,
            num_inference_steps=self.args.num_inference_steps,
            seed=self.args.seed,
            fps=self.args.fps,
        )

    def get_requests(self) -> list[RequestFuncInput]:
        return [self[i] for i in range(len(self))]


def _compute_slo_ms_from_base(req: RequestFuncInput, args, slo_base_time_ms: float | None) -> float | None:
    """Compute SLO target based on a base per-step-per-frame unit time.

    We assume linear scaling with pixel area, frame count, and num_inference_steps.
    The base unit represents latency for a 16x16 resolution, single frame, single step.
    """

    if slo_base_time_ms is None:
        return None

    width = req.width if req.width is not None else args.width
    height = req.height if req.height is not None else args.height
    if width is None or height is None:
        return None

    frames = req.num_frames if req.num_frames is not None else args.num_frames
    steps = req.num_inference_steps if req.num_inference_steps is not None else args.num_inference_steps

    frame_scale = frames if isinstance(frames, int) and frames > 0 else 1
    step_scale = steps if isinstance(steps, int) and steps > 0 else 1

    area_units = max((float(width) * float(height)) / float(16 * 16), 1.0)
    return float(slo_base_time_ms) * area_units * frame_scale * step_scale


async def _infer_slo_base_time_ms(
    args,
    api_url: str,
    model: str,
    request_func,
    session: aiohttp.ClientSession,
) -> tuple[float | None, tuple[int, int] | None]:
    """Send a small offline request to approximate the base SLO unit if unset.

    Tries 256x256, 512x512, then 1024x1024. Returns per-unit (16x16, 1 frame, 1 step)
    latency in milliseconds and the resolution used.
    """

    candidate_sizes = [(256, 256), (512, 512), (1024, 1024)]
    frames = args.num_frames if args.num_frames is not None else 1
    steps = args.num_inference_steps if args.num_inference_steps is not None else 1

    for width, height in candidate_sizes:
        sample_req = RequestFuncInput(
            prompt="SLO calibration request",
            api_url=api_url,
            model=model,
            width=width,
            height=height,
            num_frames=frames,
            num_inference_steps=steps,
            seed=args.seed,
            fps=args.fps,
        )
        out = await request_func(sample_req, session, None)
        if out.success and out.latency > 0:
            area_units = max((float(width) * float(height)) / float(16 * 16), 1.0)
            denom = area_units * max(frames, 1) * max(steps, 1)
            base_ms = (out.latency * 1000.0) / denom if denom > 0 else None
            if base_ms is not None:
                return base_ms, (width, height)

    return None, None


async def iter_requests(
    requests_list: list[RequestFuncInput],
    request_rate: float,
) -> AsyncGenerator[RequestFuncInput, None]:
    """Yield requests using a fixed interval if request_rate is set.

    - If request_rate is inf, all requests are yielded immediately (no sleep).
    - Otherwise, requests are emitted at a fixed cadence of 1 / request_rate seconds.
    """

    if request_rate != float("inf"):
        if request_rate <= 0:
            raise ValueError(f"request_rate must be positive or inf, got {request_rate}.")
        interval_s = 1.0 / float(request_rate)

    for req in requests_list:
        yield req

        if request_rate == float("inf"):
            continue

        await asyncio.sleep(interval_s)


def _guess_mime_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def _encode_image_as_data_url(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    mime = _guess_mime_type(path)
    return f"data:{mime};base64,{encoded}"


async def async_request_chat_completions(
    input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    output = RequestFuncOutput()
    output.start_time = time.perf_counter()

    extra_body = dict(input.extra_body)
    if input.width and input.height:
        extra_body.setdefault("height", input.height)
        extra_body.setdefault("width", input.width)
    if input.num_frames:
        extra_body.setdefault("num_frames", input.num_frames)
    if input.num_inference_steps:
        extra_body.setdefault("num_inference_steps", input.num_inference_steps)
    if input.seed is not None:
        extra_body.setdefault("seed", input.seed)
    if input.fps:
        extra_body.setdefault("fps", input.fps)

    if input.image_paths and len(input.image_paths) > 0:
        content = []
        if input.prompt:
            content.append({"type": "text", "text": input.prompt})
        for img_path in input.image_paths:
            if not os.path.exists(img_path):
                output.error = f"Image file not found: {img_path}"
                output.success = False
                if pbar:
                    pbar.update(1)
                return output
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image_as_data_url(img_path)},
                }
            )
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": input.prompt}]

    payload = {
        "model": input.model,
        "messages": messages,
    }
    if extra_body:
        payload["extra_body"] = extra_body

    try:
        async with session.post(input.api_url, json=payload) as response:
            if response.status == 200:
                resp_json = await response.json()
                output.response_body = resp_json
                output.success = True
                if "peak_memory_mb" in resp_json:
                    output.peak_memory_mb = resp_json["peak_memory_mb"]
            else:
                output.error = f"HTTP {response.status}: {await response.text()}"
                output.success = False
    except Exception as e:
        output.error = str(e)
        output.success = False

    output.latency = time.perf_counter() - output.start_time

    if pbar:
        pbar.update(1)
    return output


def calculate_metrics(
    outputs: list[RequestFuncOutput],
    total_duration: float,
    requests_list: list[RequestFuncInput],
    args,
    slo_base_time_ms: float | None,
    slo_enabled: bool,
):
    success_outputs = [o for o in outputs if o.success]
    error_outputs = [o for o in outputs if not o.success]

    num_success = len(success_outputs)
    latencies = [o.latency for o in success_outputs]
    peak_memories = [o.peak_memory_mb for o in success_outputs if o.peak_memory_mb > 0]

    metrics = {
        "duration": total_duration,
        "completed_requests": num_success,
        "failed_requests": len(error_outputs),
        "throughput_qps": num_success / total_duration if total_duration > 0 else 0,
        "latency_mean": np.mean(latencies) if latencies else 0,
        "latency_median": np.median(latencies) if latencies else 0,
        "latency_p99": np.percentile(latencies, 99) if latencies else 0,
        "latency_p50": np.percentile(latencies, 50) if latencies else 0,
        "peak_memory_mb_max": max(peak_memories) if peak_memories else 0,
        "peak_memory_mb_mean": np.mean(peak_memories) if peak_memories else 0,
        "peak_memory_mb_median": np.median(peak_memories) if peak_memories else 0,
    }

    if slo_enabled:
        success_pairs: list[tuple[RequestFuncInput, RequestFuncOutput]] = [
            (req, out) for req, out in zip(requests_list, outputs) if out.success
        ]
        slo_met_flags: list[bool] = []

        for req, out in success_pairs:
            out.measured_latency_s = out.latency
            slo_ms = _compute_slo_ms_from_base(req, args, slo_base_time_ms)
            out.slo_ms = slo_ms
            if slo_ms is not None:
                met = out.latency * 1000.0 <= slo_ms
                out.slo_achieved = met
                slo_met_flags.append(met)

        slo_total = len(requests_list)
        slo_defined_success = len(slo_met_flags)
        slo_met_success = int(sum(1 for x in slo_met_flags if x))
        slo_attain_success_only = (slo_met_success / slo_defined_success) if slo_defined_success > 0 else 0.0
        slo_attain_all = slo_met_success / slo_total if slo_total > 0 else 0.0

        metrics.update(
            {
                "slo_attainment_rate": slo_attain_all,
                "slo_attainment_rate_success_only": slo_attain_success_only,
                "slo_defined_success": slo_defined_success,
                "slo_met_success": slo_met_success,
                "slo_base_time_ms": slo_base_time_ms,
            }
        )

    return metrics


def wait_for_service(base_url: str, timeout: int = 120) -> None:
    print(f"Waiting for service at {base_url}...")
    start_time = time.time()
    while True:
        try:
            # Try /health endpoint first
            resp = requests.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200:
                print("Service is ready.")
                break
        except requests.exceptions.RequestException:
            pass

        if time.time() - start_time > timeout:
            raise TimeoutError(f"Service at {base_url} did not start within {timeout} seconds.")

        time.sleep(1)


async def benchmark(args):
    # Construct base_url if not provided
    if args.base_url is None:
        args.base_url = f"http://{args.host}:{args.port}"

    # Setup dataset (vLLM-Omni supports diffusion via /v1/chat/completions)
    api_url = f"{args.base_url}/v1/chat/completions"
    request_func = async_request_chat_completions

    if args.dataset == "vbench":
        dataset = VBenchDataset(args, api_url, args.model)
    elif args.dataset == "trace":
        dataset = TraceDataset(args, api_url, args.model)
    elif args.dataset == "random":
        dataset = RandomDataset(args, api_url, args.model)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print("Loading requests...")
    requests_list = dataset.get_requests()
    print(f"Prepared {len(requests_list)} requests from {args.dataset} dataset.")

    # Limit concurrency
    if args.max_concurrency is not None:
        semaphore = asyncio.Semaphore(args.max_concurrency)
    else:
        semaphore = None

    async def limited_request_func(req, session, pbar):
        if semaphore:
            async with semaphore:
                return await request_func(req, session, pbar)
        else:
            return await request_func(req, session, pbar)

    slo_base_time_ms = args.slo_base_time
    slo_base_resolution = None

    # Run benchmark
    pbar = tqdm(total=len(requests_list), disable=args.disable_tqdm)

    async with aiohttp.ClientSession() as session:
        if args.slo and slo_base_time_ms is None:
            print("Inferring SLO base unit via offline request (per 16x16 px, 1 frame, 1 step)...")
            slo_base_time_ms, slo_base_resolution = await _infer_slo_base_time_ms(
                args=args,
                api_url=api_url,
                model=args.model,
                request_func=request_func,
                session=session,
            )
            if slo_base_time_ms is not None:
                res_str = f"{slo_base_resolution[0]}x{slo_base_resolution[1]}" if slo_base_resolution else "unknown"
                print(f"Inferred SLO base time: {slo_base_time_ms:.2f} ms from {res_str} sample.")
            else:
                print("Failed to infer SLO base time; SLO metrics will be skipped unless provided.")

        if args.warmup_requests and requests_list:
            print(
                f"Running {args.warmup_requests} warmup request(s) \
                with num_inference_steps={args.warmup_num_inference_steps}..."
            )
            for i in range(args.warmup_requests):
                warm_req = requests_list[i % len(requests_list)]
                if args.warmup_num_inference_steps is not None:
                    warm_req = replace(
                        warm_req,
                        num_inference_steps=args.warmup_num_inference_steps,
                    )
                await limited_request_func(warm_req, session, None)

        start_time = time.perf_counter()
        tasks = []
        async for req in iter_requests(requests_list=requests_list, request_rate=args.request_rate):
            task = asyncio.create_task(limited_request_func(req, session, pbar))
            tasks.append(task)

        outputs = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_time

    pbar.close()

    # Calculate metrics
    metrics = calculate_metrics(outputs, total_duration, requests_list, args, slo_base_time_ms, args.slo)

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=60, c="="))

    # Section 1: Configuration
    print("{:<40} {:<15}".format("Model:", args.model))
    print("{:<40} {:<15}".format("Dataset:", args.dataset))
    print("{:<40} {:<15}".format("Task:", args.task))

    # Section 2: Execution & Traffic
    print(f"{'-' * 50}")
    print("{:<40} {:<15.2f}".format("Benchmark duration (s):", metrics["duration"]))
    print("{:<40} {:<15}".format("Request rate:", str(args.request_rate)))
    print(
        "{:<40} {:<15}".format(
            "Max request concurrency:",
            str(args.max_concurrency) if args.max_concurrency else "not set",
        )
    )
    print("{:<40} {}/{:<15}".format("Successful requests:", metrics["completed_requests"], len(requests_list)))

    # Section 3: Performance Metrics
    print(f"{'-' * 50}")

    print("{:<40} {:<15.2f}".format("Request throughput (req/s):", metrics["throughput_qps"]))
    print("{:<40} {:<15.4f}".format("Latency Mean (s):", metrics["latency_mean"]))
    print("{:<40} {:<15.4f}".format("Latency Median (s):", metrics["latency_median"]))
    print("{:<40} {:<15.4f}".format("Latency P99 (s):", metrics["latency_p99"]))

    if args.slo:
        print(f"{'-' * 50}")
        print("{:<40} {:<15.2%}".format("SLO Attainment Rate (all):", metrics.get("slo_attainment_rate", 0.0)))
        print(
            "{:<40} {:<15.2%}".format(
                "SLO Attainment Rate (success-only):",
                metrics.get("slo_attainment_rate_success_only", 0.0),
            )
        )
        print(
            "{:<40} {:<15}".format(
                "SLO evaluated (success count):",
                f"{metrics.get('slo_met_success', 0)}/{metrics.get('slo_defined_success', 0)}",
            )
        )
        if metrics.get("slo_base_time_ms"):
            print("{:<40} {:<15.2f}".format("SLO base time (ms/unit):", float(metrics.get("slo_base_time_ms", 0.0))))

    if metrics["peak_memory_mb_max"] > 0:
        print(f"{'-' * 50}")
        print("{:<40} {:<15.2f}".format("Peak Memory Max (MB):", metrics["peak_memory_mb_max"]))
        print("{:<40} {:<15.2f}".format("Peak Memory Mean (MB):", metrics["peak_memory_mb_mean"]))
        print("{:<40} {:<15.2f}".format("Peak Memory Median (MB):", metrics["peak_memory_mb_median"]))

    print("\n" + "=" * 60)

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark serving for diffusion models.")
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL of the server (e.g., http://localhost:30000). Overrides host/port.",
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host.")
    parser.add_argument("--port", type=int, default=30000, help="Server port.")
    parser.add_argument("--model", type=str, default="default", help="Model name.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="vbench",
        choices=["vbench", "trace", "random"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v",
        choices=["t2v", "i2v", "ti2v", "ti2i", "i2i", "t2i"],
        help="Task type.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to local dataset file (optional).",
    )
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of prompts to benchmark.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,  ### 1
        help="Maximum number of concurrent requests, default to `1`. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="Number of warmup requests to run before measurement.",
    )
    parser.add_argument(
        "--warmup-num-inference-steps",
        type=int,
        default=1,
        help="num_inference_steps used for warmup requests.",
    )
    parser.add_argument("--width", type=int, default=None, help="Image/Video width.")
    parser.add_argument("--height", type=int, default=None, help="Image/Video height.")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames (for video).")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Number of inference steps (for diffusion models).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (for diffusion models).",
    )
    parser.add_argument("--fps", type=int, default=None, help="FPS (for video).")
    parser.add_argument("--output-file", type=str, default=None, help="Output JSON file for metrics.")
    parser.add_argument(
        "--slo",
        action="store_true",
        help=(
            "Enable SLO calculation and reporting. SLO targets are derived from --slo-base-time "
            "(per 16x16 px, one frame, one step). If not provided, a calibration request tries to "
            "infer it assuming linear scaling by resolution, frames, and steps."
        ),
    )
    parser.add_argument(
        "--slo-base-time",
        type=float,
        default=None,
        help=(
            "Base latency in milliseconds for a 16x16 resolution, single-frame, single-step request "
            "(roughly the minimal unit, typically ~= step_time * 3). Used to linearly scale SLO to "
            "the actual resolution, frame count, and num_inference_steps."
        ),
    )
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable progress bar.")

    args = parser.parse_args()

    asyncio.run(benchmark(args))
