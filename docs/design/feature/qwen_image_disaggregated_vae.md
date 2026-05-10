# Qwen-Image Disaggregated VAE

This document describes the Qwen-Image disaggregated VAE path in vLLM-Omni.
The feature splits the monolithic Qwen-Image diffusion pipeline into separate
encode, denoise, and decode stages while keeping the existing single-stage
diffusion path unchanged.

## Overview

The original Qwen-Image pipeline runs prompt encoding, DiT denoising, and VAE
decoding inside one diffusion stage. The disaggregated path separates these
steps so each stage loads only the components it needs:

```text
Encode stage -> Denoise stage -> Decode stage
GPU 0          GPU 1            GPU 0
```

This is intended to reduce per-stage memory pressure and make it possible to
place text encoder/VAE work separately from the transformer denoise loop.

## Goals

- Keep the existing `model_stage=None` and `model_stage=diffusion` behavior
  compatible with the monolithic Qwen-Image path.
- Load Qwen-Image components by stage instead of loading the full pipeline in
  every process.
- Reuse the existing multi-stage orchestrator and
  `custom_process_input_func` mechanism.
- Support both request-level `execute_model` and stepwise `execute_stepwise`
  for the denoise stage.
- Keep Qwen-specific tensor schema handling inside the Qwen stage input
  processors and pipeline code.

This feature is not a general-purpose arbitrary submodule graph. It provides a
minimal, model-specific encode -> denoise -> decode path for Qwen-Image.

## Stage Layout

The reference stage config is
`vllm_omni/model_executor/stage_configs/qwen_image_3stage.yaml`.

| Stage | Runtime `stage_type` | Runtime `worker_type` | `model_stage` | Responsibility |
| :--- | :--- | :--- | :--- | :--- |
| 0 | `diffusion` | `submodule` | `encode` | Tokenizer, text encoder, scheduler inputs, initial latents. |
| 1 | `diffusion` | unset | `denoise` | Transformer denoise loop and scheduler steps. |
| 2 | `diffusion` | `submodule` | `decode` | VAE decode and image postprocessing. |

`model_stage=diffusion` remains the full coupled path:

| `model_stage` | Components |
| :--- | :--- |
| `diffusion` | scheduler, text encoder, tokenizer, VAE, transformer |
| `encode` | scheduler, text encoder, tokenizer |
| `denoise` | scheduler, transformer |
| `decode` | VAE |

## Runtime Flow

`DIFFUSION_SUBMODULE` has the same architectural position for diffusion that
`LLM_GENERATION` has for LLM: it is not a new runtime `StageType`. The merged
execution type expands to an existing stage family plus a worker selector:

| Merged execution type | Runtime shape | StagePool view |
| :--- | :--- | :--- |
| `LLM_GENERATION` | `stage_type=llm`, `worker_type=generation` | LLM stage |
| `DIFFUSION_SUBMODULE` | `stage_type=diffusion`, `worker_type=submodule` | Diffusion stage |

The difference is the output contract. `LLM_GENERATION` still runs through the
vLLM engine-core path and emits token `EngineCoreOutputs`; diffusion submodule
stages use the existing direct diffusion-output path and emit
`OmniRequestOutput`.

`worker_type=submodule` only changes how a diffusion replica is initialized:
`initialize_diffusion_stage()` builds `StageSubModuleClient` instead of the
full `StageDiffusionClient`. StagePool and the orchestrator still see a normal
`stage_type=diffusion` stage and poll it through `get_diffusion_output_nowait`.

The orchestrator flow is:

1. Submit the user request to stage 0.
2. Poll the direct diffusion output from the current stage pool.
3. Run the next stage's configured `custom_process_input_func`.
4. Submit the converted prompt to the next diffusion stage.
5. Return the final decode output to the caller.

The Qwen-Image conversion functions live in
`vllm_omni/model_executor/stage_input_processors/qwen_image.py`:

- `encode_to_denoise`
- `denoise_to_decode`

The orchestrator does not inspect the Qwen tensor schema.

## Data Contract

Intermediate tensors are passed through the existing multimodal output path:

```text
DiffusionOutput.multimodal_output
  -> OmniRequestOutput.multimodal_output
  -> custom_process_input_func
  -> OmniTokensPrompt.additional_information
  -> downstream pipeline
```

Encode output contains prompt embeddings, masks, initial latents, timesteps,
image shapes, and CFG metadata. Denoise output contains final latents, latent
shape, target image size, and output type.

The current implementation uses the existing ZMQ/msgpack path. Large tensors
therefore still take a device-to-host-to-device path. Connector-backed tensor
transfer is a future optimization, not part of this initial feature.

## Execution

`QwenImagePipeline` provides separate stage entry points:

- `execute_encode()` for the encode submodule stage.
- `prepare_encode()`, `denoise_step()`, and `step_scheduler()` for the denoise
  diffusion stage.
- `post_intermediate_output()` to emit denoised latents instead of decoding.
- `execute_decode()` for the decode submodule stage.

`DiffusionModelRunner` uses `post_intermediate_output()` when a stepwise stage
declares `produces_intermediate_stage_output`. For Qwen-Image denoise-only
request-level execution, the pipeline exposes `execute_stage_model()` and
returns the same intermediate payload.

## Compatibility

Existing Qwen-Image users do not need to change their configuration. Without a
multi-stage config, Qwen-Image runs as before and performs text encode,
denoise, and VAE decode inside the same diffusion stage.

## Related Files

- `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`
- `vllm_omni/diffusion/stage_submodule_client.py`
- `vllm_omni/diffusion/stage_submodule_proc.py`
- `vllm_omni/diffusion/worker/vae_model_runner.py`
- `vllm_omni/model_executor/stage_input_processors/qwen_image.py`
- `vllm_omni/model_executor/stage_configs/qwen_image_3stage.yaml`

## Limitations

- Tensor transfer is still inline ZMQ/msgpack, not connector-backed D2D.
- The submodule runner is currently Qwen-Image encode/decode oriented.
- The reference topology is a fixed linear 1 encode : 1 denoise : 1 decode
  pipeline.
- Image-conditioning/VAE-encode expansion can be added on top of the same
  encode-stage boundary.
