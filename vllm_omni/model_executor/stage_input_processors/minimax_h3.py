# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import copy
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image

from vllm_omni.data_entry_keys import unflatten_payload
from vllm_omni.diffusion.models.minimax_h3.time_request import (
    MINIMAX_H3_SHAPE_PLANNER,
    minimax_h3_align_frame_count,
)
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.minimax_h3.conditioning import (
    MINIMAX_H3_CONDITION_LABELS_KEY,
    MINIMAX_H3_ENCODER_REQUEST_KEY,
    MINIMAX_H3_PRESENTATION_TASK_KEY,
    MiniMaxH3EncoderConditioning,
    MiniMaxH3EncoderMediaInput,
)
from vllm_omni.model_executor.models.minimax_h3.preprocessing import (
    MINIMAX_H3_OUTPUT_SHORT_EDGE,
    load_minimax_h3_images,
    resolve_minimax_h3_aspect_ratio,
    resolve_minimax_h3_output_canvas,
    resolve_minimax_h3_reference_image_shape,
)
from vllm_omni.model_executor.models.minimax_h3.reference_video import (
    MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS,
    load_audio_file,
    load_video_audio,
    load_video_frames,
    prepare_reference_videos,
    sample_reference_video_frames,
    validate_reference_audio_files,
    validate_reference_audio_waveforms,
)

MINIMAX_H3_FPS = 24
MINIMAX_H3_MIN_OUTPUT_SECONDS = 4.0
MINIMAX_H3_MAX_OUTPUT_SECONDS = 15.0


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) and not (len(value) == 2 and isinstance(value[1], Mapping)):
        return list(value)
    return [value]


def _audio_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[1], (int, np.integer)):
        return [value]
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _load_audio(value: Any) -> tuple[torch.Tensor, int]:
    if isinstance(value, (list, tuple)) and not (len(value) == 2 and isinstance(value[1], (int, np.integer))):
        audios = _load_audios(value)
        if len(audios) != 1:
            raise OmniClientError(f"MiniMax H3 expected one audio, got {len(audios)}")
        return audios[0]
    if isinstance(value, (str, os.PathLike)):
        return load_audio_file(str(value))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        waveform, sample_rate = value
        return torch.as_tensor(waveform).float(), int(sample_rate)
    if isinstance(value, dict):
        waveform = value.get("waveform", value.get("array"))
        sample_rate = value.get("sample_rate", value.get("sampling_rate"))
        if waveform is not None and sample_rate is not None:
            return torch.as_tensor(waveform).float(), int(sample_rate)
    raise OmniClientError("MiniMax H3 audio input must be a path, (waveform, sample_rate), or a waveform mapping")


def _load_audios(value: Any) -> list[tuple[torch.Tensor, int]]:
    if isinstance(value, (list, tuple)) and not (len(value) == 2 and isinstance(value[1], (int, np.integer))):
        if not value:
            raise OmniClientError("MiniMax H3 audio input must not be empty")
        return [_load_audio(item) for item in value]
    return [_load_audio(value)]


def _as_int_list(value: Any, *, name: str) -> list[int]:
    if isinstance(value, bool):
        raise OmniClientError(f"{name} must be an integer or a list of integers")
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = list(value)
        if not result:
            raise OmniClientError(f"{name} must not be empty")
        if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in result):
            raise OmniClientError(f"{name} must contain only integers")
        return [int(item) for item in result]
    raise OmniClientError(f"{name} must be an integer or a list of integers")


def _resolve_fl2va_keyframe_indices(extra: Mapping[str, Any], image_count: int) -> list[int]:
    target = extra.get("target")
    target = target if isinstance(target, Mapping) else {}
    raw = extra.get("frame_indices", extra.get("frame_index"))
    if raw is None:
        raw = target.get("frame_indices", target.get("frame_index"))
    raw_indices = ([0] if image_count == 1 else [0, -1]) if raw is None else _as_int_list(raw, name="frame_indices")
    if len(raw_indices) != image_count:
        raise OmniClientError(
            f"MiniMax H3 FL2VA requires one frame index per image: got {raw_indices!r} for {image_count} image(s)"
        )
    if tuple(raw_indices) not in ((0,), (-1,), (0, -1)):
        raise OmniClientError("MiniMax H3 FL2VA frame_indices must be [0], [-1], or [0, -1]")
    return raw_indices


def _validate_ref2va_reference_counts(image_count: int, video_count: int, audio_count: int) -> None:
    if min(image_count, video_count, audio_count) < 0:
        raise OmniClientError("MiniMax H3 reference counts must be non-negative")
    if image_count + video_count == 0:
        raise OmniClientError("ref2va requires at least one image or video reference")
    if image_count > 9:
        raise OmniClientError("ref2va accepts at most 9 image references")
    if video_count > 3:
        raise OmniClientError("ref2va accepts at most 3 video references")
    if audio_count > 3:
        raise OmniClientError("ref2va accepts at most 3 standalone audio references")
    if image_count + video_count + audio_count > 12:
        raise OmniClientError("ref2va accepts at most 12 total references")


def _validate_reference_image(image: Image.Image) -> None:
    resolve_minimax_h3_reference_image_shape(image)


def resolve_minimax_h3_shape(
    task: str,
    sampling: Any,
    image: Image.Image | None,
) -> tuple[int, int, int, int, int]:
    fps = int(getattr(sampling, "fps", None) or MINIMAX_H3_FPS)
    if fps != MINIMAX_H3_FPS:
        raise OmniClientError(f"MiniMax H3 output fps is fixed at {MINIMAX_H3_FPS}")
    extra = sampling.extra_args or {}
    target = extra.get("target")
    if target is not None and not isinstance(target, Mapping):
        raise OmniClientError("MiniMax H3 extra_args['target'] must be an object")
    target = target if isinstance(target, Mapping) else {}
    duration = target.get("duration_seconds", extra.get("duration_seconds", extra.get("duration")))
    if duration is not None:
        if isinstance(duration, bool):
            raise OmniClientError(f"MiniMax H3 output duration must be in [4, 15] seconds, got {duration!r}")
        try:
            duration = float(duration)
        except (TypeError, ValueError) as exc:
            raise OmniClientError(f"MiniMax H3 output duration must be in [4, 15] seconds, got {duration!r}") from exc
        if (
            not math.isfinite(duration)
            or not MINIMAX_H3_MIN_OUTPUT_SECONDS <= duration <= MINIMAX_H3_MAX_OUTPUT_SECONDS
        ):
            raise OmniClientError(f"MiniMax H3 output duration must be in [4, 15] seconds, got {duration}")
        requested_frames = int(round(duration * fps))
    elif int(getattr(sampling, "num_frames", None) or 1) > 1:
        requested_frames = int(sampling.num_frames)
    else:
        requested_frames = 124 if task == "ref2va" else 209
    if not MINIMAX_H3_MIN_OUTPUT_SECONDS <= requested_frames / fps <= MINIMAX_H3_MAX_OUTPUT_SECONDS:
        raise OmniClientError(
            f"MiniMax H3 output duration must be in [4, 15] seconds, got {requested_frames / fps:.3f}"
        )
    num_frames = minimax_h3_align_frame_count(requested_frames)

    height = getattr(sampling, "height", None)
    width = getattr(sampling, "width", None)
    aspect_ratio = resolve_minimax_h3_aspect_ratio(
        task,
        target.get("aspect_ratio", extra.get("aspect_ratio")),
        image,
    )
    raw_short_edge = target.get("short_edge", extra.get("short_edge", MINIMAX_H3_OUTPUT_SHORT_EDGE))
    if isinstance(raw_short_edge, bool) or not isinstance(raw_short_edge, (int, np.integer)):
        raise OmniClientError(
            f"MiniMax H3 target.short_edge must be {MINIMAX_H3_OUTPUT_SHORT_EDGE}, got {raw_short_edge!r}"
        )
    if height is None or width is None:
        height, width = resolve_minimax_h3_output_canvas(aspect_ratio, int(raw_short_edge))
    height = int(height) // 32 * 32
    width = int(width) // 32 * 32
    if min(height, width) <= 0:
        raise OmniClientError(f"invalid MiniMax H3 canvas {width}x{height}")
    if width > 4 * height or height > 4 * width:
        raise OmniClientError("MiniMax H3 canvas aspect ratio must be in [1:4, 4:1]")
    return (
        height,
        width,
        num_frames,
        MINIMAX_H3_SHAPE_PLANNER.video_latent_t(num_frames),
        MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(num_frames / fps),
    )


def _resolve_task(
    extra_args: Mapping[str, Any],
    multi_modal_data: Mapping[str, Any],
) -> str:
    requested = extra_args.get("task")
    if requested is not None:
        return str(requested).lower()
    if multi_modal_data.get("video") is not None or multi_modal_data.get("audio") is not None:
        return "ref2va"
    if multi_modal_data.get("image") is not None:
        return "fl2va"
    return "t2va"


def _diffusion_sampling_params(sampling_params_list: Sequence[Any]) -> Any:
    diffusion_params = [
        sampling_params
        for sampling_params in sampling_params_list
        if isinstance(sampling_params, OmniDiffusionSamplingParams)
    ]
    if len(diffusion_params) != 1:
        raise RuntimeError(
            "MiniMax H3 encoding requires exactly one OmniDiffusionSamplingParams stage parameter, "
            f"got {len(diffusion_params)}"
        )
    return diffusion_params[0]


def _prepare_encoder_images(
    task: str,
    images: list[Image.Image],
    *,
    height: int,
    width: int,
) -> list[Any]:
    if not images:
        return []
    if task == "ref2va":
        return [
            image.resize(
                resolve_minimax_h3_reference_image_shape(image),
                Image.Resampling.LANCZOS,
            )
            for image in images
        ]
    if task != "fl2va":
        return images
    return [image.resize((width, height), Image.Resampling.LANCZOS) for image in images]


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(np.array(array, copy=True)).contiguous()


def _frames_to_tensor(frames: Sequence[Any]) -> torch.Tensor:
    if len(frames) == 0:
        raise ValueError("MiniMax H3 reference video must contain frames")
    return torch.stack(
        [
            _image_to_tensor(frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame)))
            for frame in frames
        ]
    )


def prepare_encoder_prompt(
    prompt: Any,
    sampling_params_list: Sequence[Any],
) -> Any:
    if isinstance(prompt, str):
        prompt = {"prompt": prompt}
    if not isinstance(prompt, dict):
        raise TypeError(f"MiniMax H3 expects a string or dict prompt, got {type(prompt)!r}")

    text = str(prompt.get("prompt") or "")
    if not text:
        raise OmniClientError("MiniMax H3 requires a non-empty prompt")
    multi_modal_data = prompt.get("multi_modal_data") or {}
    if not isinstance(multi_modal_data, Mapping):
        raise TypeError("multi_modal_data must be a mapping")

    image_values = _items(multi_modal_data.get("image"))
    videos = _items(multi_modal_data.get("video"))
    raw_audio = multi_modal_data.get("audio")
    audio_values = _audio_items(raw_audio)
    diffusion_sampling_params = _diffusion_sampling_params(sampling_params_list)
    extra_args = getattr(diffusion_sampling_params, "extra_args", None) or {}
    task = _resolve_task(extra_args, multi_modal_data)
    raw_images = load_minimax_h3_images(image_values) if image_values else []

    if task == "t2va":
        if raw_images or videos or audio_values:
            raise OmniClientError("t2va does not accept image, video, or audio conditions")
    elif task == "fl2va":
        if not raw_images or videos or audio_values:
            raise OmniClientError("fl2va requires image conditions only")
        if len(raw_images) > 2:
            raise OmniClientError("fl2va accepts at most first and last images")
        for image in raw_images:
            _validate_reference_image(image)
    elif task == "ref2va":
        _validate_ref2va_reference_counts(len(raw_images), len(videos), len(audio_values))
        if not raw_images and not videos:
            raise OmniClientError("ref2va requires an image or video condition")
    else:
        raise OmniClientError(f"unsupported MiniMax H3 task {task!r}")

    height, width, num_frames, latent_t, audio_t = resolve_minimax_h3_shape(
        task,
        diffusion_sampling_params,
        raw_images[0] if raw_images else None,
    )
    images = _prepare_encoder_images(
        task,
        raw_images,
        height=height,
        width=width,
    )
    keyframe_indices = _resolve_fl2va_keyframe_indices(extra_args, len(images)) if task == "fl2va" else []
    qwen_video_inputs: list[tuple[np.ndarray, dict[str, Any]]] = []
    condition_labels: list[tuple[str, int]] = []
    encoded_video_inputs: list[torch.Tensor] = []
    video_audio_inputs: list[tuple[torch.Tensor, int] | None] = []

    if task == "fl2va":
        condition_labels.extend(("image", index) for index in range(1, len(images) + 1))
    elif task == "ref2va":
        condition_labels.extend(("image", index) for index in range(1, len(images) + 1))
        prepared_videos: list[dict[str, Any]] = []
        if videos:
            with tempfile.TemporaryDirectory(prefix="minimax_h3_encoder_") as workdir:
                prepared_videos = prepare_reference_videos(
                    videos,
                    target_frame_count=num_frames,
                    workdir=workdir,
                    start_time_seconds=extra_args.get("start_time_seconds"),
                )
                for item in prepared_videos:
                    full_frames = load_video_frames(item["prepared_path"])
                    encoded_video_inputs.append(_frames_to_tensor(full_frames))
                    sampled = sample_reference_video_frames(
                        item["prepared_path"],
                        decoded_frames=full_frames,
                    )
                    frames = np.stack(sampled["frames"])
                    frame_count = int(frames.shape[0])
                    qwen_video_inputs.append(
                        (
                            frames,
                            {
                                "total_num_frames": frame_count,
                                "fps": MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS,
                                "duration": frame_count / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS,
                                "video_backend": "minimax_h3",
                                "frames_indices": list(range(frame_count)),
                                "do_sample_frames": False,
                            },
                        )
                    )
                    if item["input_has_audio"]:
                        waveform, sample_rate = load_video_audio(
                            item["original_path"],
                            start_time_seconds=float(item.get("start_time_seconds", 0.0)),
                            duration_seconds=item.get(
                                "audio_duration_seconds",
                                item.get("duration_seconds"),
                            ),
                        )
                        video_audio_inputs.append((waveform.float().contiguous(), int(sample_rate)))
                    else:
                        video_audio_inputs.append(None)
        audio_index = 0
        for video_index, item in enumerate(prepared_videos, start=1):
            if item["input_has_audio"]:
                audio_index += 1
                condition_labels.append(("audio", audio_index))
            condition_labels.append(("video", video_index))
        for _ in audio_values:
            audio_index += 1
            condition_labels.append(("audio", audio_index))

    if raw_audio is not None:
        validate_reference_audio_files(raw_audio)
    standalone_audios = _load_audios(raw_audio) if raw_audio is not None else []
    validate_reference_audio_waveforms(standalone_audios)
    media_input = MiniMaxH3EncoderMediaInput(
        task=task,
        height=height,
        width=width,
        num_frames=num_frames,
        latent_t=latent_t,
        audio_t=audio_t,
        images=tuple(_image_to_tensor(image) for image in images),
        videos=tuple(encoded_video_inputs),
        video_audios=tuple(video_audio_inputs),
        audios=tuple((waveform.float().contiguous(), int(sample_rate)) for waveform, sample_rate in standalone_audios),
        keyframe_frame_indices=tuple(keyframe_indices),
    )

    transformed = copy.copy(prompt)
    additional_information = dict(prompt.get("additional_information") or {})
    transformed["prompt"] = text
    qwen_mm_data: dict[str, Any] = {}
    if images:
        qwen_mm_data["image"] = images
    if qwen_video_inputs:
        qwen_mm_data["video"] = qwen_video_inputs
    transformed["multi_modal_data"] = qwen_mm_data or None

    mm_processor_kwargs = dict(prompt.get("mm_processor_kwargs") or {})
    mm_processor_kwargs[MINIMAX_H3_PRESENTATION_TASK_KEY] = task
    mm_processor_kwargs[MINIMAX_H3_CONDITION_LABELS_KEY] = condition_labels
    media_tensors = media_input.to_mm_tensors()
    transformed["mm_processor_kwargs"] = mm_processor_kwargs

    hidden_states = dict(additional_information.get("hidden_states") or {})
    hidden_states["layers"] = dict(enumerate(media_tensors))
    additional_information["hidden_states"] = hidden_states
    meta = dict(additional_information.get("meta") or {})
    meta[MINIMAX_H3_ENCODER_REQUEST_KEY] = media_input.to_metadata()
    additional_information["meta"] = meta
    transformed["additional_information"] = additional_information
    return transformed


def _original_prompt(prompt: Any) -> dict[str, Any]:
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else {}
    if isinstance(prompt, dict):
        return copy.copy(prompt)
    if isinstance(prompt, str):
        return {"prompt": prompt}
    raise TypeError(f"invalid MiniMax H3 prompt type {type(prompt)!r}")


def _global_request_id(prompt: Mapping[str, Any]) -> str | None:
    additional_information = prompt.get("additional_information")
    if not isinstance(additional_information, Mapping):
        return None
    value = additional_information.get("global_request_id")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return str(value) if value is not None else None


def encoder2diffusion(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> dict[str, Any] | None:
    """Reuse the text-encoder handoff for all three H3 encoders."""
    del requires_multimodal_data, streaming_context
    if not source_outputs:
        return None
    if len(source_outputs) != 1:
        raise RuntimeError(f"MiniMax H3 DiT requires exactly one encoder source, got {len(source_outputs)}")
    if not getattr(source_outputs[0], "finished", True):
        return None

    diffusion_prompt = _original_prompt(prompt)
    source_output = source_outputs[0]
    source_request_id = getattr(source_output, "request_id", None)
    expected_request_id = _global_request_id(diffusion_prompt)
    if (
        source_request_id is not None
        and expected_request_id is not None
        and str(source_request_id) != expected_request_id
    ):
        raise RuntimeError(
            "MiniMax H3 encoder request ID does not match the diffusion request: "
            f"source={source_request_id!r}, expected={expected_request_id!r}"
        )

    outputs = getattr(source_output, "outputs", None)
    if not isinstance(outputs, list) or len(outputs) != 1:
        output_count = len(outputs) if isinstance(outputs, list) else 0
        raise RuntimeError(f"MiniMax H3 encoder must return exactly one completion, got {output_count}")
    payload = getattr(outputs[0], "multimodal_output", None)
    if not isinstance(payload, Mapping):
        raise RuntimeError("MiniMax H3 encoder returned no conditioning payload")
    try:
        conditioning = MiniMaxH3EncoderConditioning.from_omni_payload(unflatten_payload(dict(payload)))
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    additional_information = dict(diffusion_prompt.get("additional_information") or {})
    additional_information.pop("hidden_states", None)
    meta = dict(additional_information.get("meta") or {})
    meta.pop(MINIMAX_H3_ENCODER_REQUEST_KEY, None)
    if meta:
        additional_information["meta"] = meta
    else:
        additional_information.pop("meta", None)
    additional_information["text_encoder_output"] = conditioning.to_omni_payload()
    diffusion_prompt["additional_information"] = additional_information
    diffusion_prompt["multi_modal_data"] = None
    diffusion_prompt.pop("model_intermediate_buffer", None)
    return diffusion_prompt
