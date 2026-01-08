
# Diffusion Serving Benchmark (Image/Video)

This folder contains an online-serving benchmark script for diffusion models.
It sends requests to a vLLM OpenAI-compatible endpoint and reports throughput,
latency percentiles, and optional SLO attainment.

The main entrypoint is:

- `benchmarks/diffusion/diffusion_benchmark_serving.py`

## Quick Start

1. Start the server:

```bash
vllm serve Qwen/Qwen-Image --omni --port 8099
```

2. Run a small benchmark (trace dataset):

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
	--base-url http://localhost:8099 \
	--model Qwen/Qwen-Image \
	--task t2i \
	--dataset trace \
	--num-prompts 5 \
	--request-rate 0.1 \
	--max-concurrency 5 \
	--slo \
	--warmup-requests 2
```

## Notes

- The benchmark talks to `http://<host>:<port>/v1/chat/completions`.
- If you run the server on another host or port, pass `--base-url` accordingly.

## Run the Benchmark

You can run the script either from the repo root or from the `benchmarks/` directory.

### Trace + SLO (recommended for SLO testing)

When testing SLO, avoid `--max-concurrency 1`.
With a small concurrency limit, requests queue behind the semaphore and the achieved QPS may be
lower than `--request-rate`, which will also skew SLO results.

Using more warmup requests can make the SLO estimation less noisy:

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
	--base-url http://localhost:8099 \
	--model Qwen/Qwen-Image \
	--task t2i \
	--dataset trace \
	--num-prompts 5 \
	--request-rate 0.1 \
	--max-concurrency 5 \
	--slo \
	--warmup-requests 2
```

### VBench + SLO

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
	--base-url http://localhost:8099 \
	--dataset vbench \
	--task t2i \
	--num-prompts 10 \
	--height 1024 \
	--width 1024 \
	--slo \
	--max-concurrency 5 \
	--request-rate 0.1
```

### Sweep QPS (SLO vs QPS)

Example with higher QPS:

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
	--base-url http://localhost:8099 \
	--model Qwen/Qwen-Image \
	--task t2i \
	--dataset trace \
	--num-prompts 5 \
	--request-rate 1 \
	--max-concurrency 5 \
	--slo \
	--warmup-requests 2
```

### VBench examples (built-in)

The `vbench` dataset provides ready-to-use prompts for diffusion benchmarks.
It supports text-to-video (`t2v`) and image-conditioned tasks (`i2v`, `i2i`).

Text-to-video:

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
	--dataset vbench --task t2v --num-prompts 10 \
	--height 480 --width 640 --fps 16 --num-frames 80
```

Image-to-video:

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
	--dataset vbench --task i2v --num-prompts 10
```

Text-to-image:

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
	--dataset vbench --task t2i --num-prompts 10 \
	--height 1024 --width 1024
```

Image-to-image:

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
	--dataset vbench --task i2i --num-prompts 10
```

If you use i2v/i2i datasets and need auto-download support, you may need:

```bash
uv pip install gdown
```

## 3) Supported Datasets

The benchmark supports three dataset modes via `--dataset`:

- `trace`: Heterogeneous request traces (each request can have different resolution/frames/steps).
- `vbench`: VBench prompt/data loader.
- `random`: Synthetic prompts for quick smoke tests.

### Trace dataset

Use `--dataset trace` to replay a trace file. The trace can specify per-request fields such as:

- `width`, `height`
- `num_frames` (video)
- `num_inference_steps`
- `seed`, `fps`
- optional `slo_ms` (per-request SLO target)

By default (when `--dataset-path` is not provided), the script downloads a default trace from
the HuggingFace dataset repo `asukaqaqzz/Dit_Trace`. The default filename can depend on `--task`
(e.g., `t2v` uses a video trace).

You can point to your own trace using `--dataset-path`.

## 4) Important Arguments

This section highlights the most important knobs for interpreting results.

### SLO (`--slo`, `--slo-scale`, warmup)

When `--slo` is enabled:

- If the trace already contains `slo_ms` for a request, that value is used.
- Otherwise, the script runs warmup requests and infers a base unit time, then estimates an SLO
	for each request using a simple linear scaling model.

Warmup knobs:

- `--warmup-requests`: number of warmup requests to execute before measurement.
- `--warmup-num-inference-steps`: the `num_inference_steps` used during warmup.

Important:

- SLO estimation quality depends on warmup representativeness. If your workload has multiple
	resolutions/steps/frames, increasing `--warmup-requests` can help.
- For SLO-vs-QPS sweeps, set `--max-concurrency` high enough to avoid throttling. If concurrency
	is too low, you will not actually hit the configured `--request-rate`.


Where `--slo-scale` controls how strict the target is (default `3.0`).

### Traffic shaping (`--request-rate`, `--max-concurrency`)

- `--request-rate`: target request rate in requests/second.
	- If it is `inf`, all requests are launched immediately.
	- Otherwise, requests are emitted at a fixed cadence of $1 / \text{request\_rate}$ seconds.

- `--max-concurrency`: maximum number of in-flight requests allowed at once (default `1`).
	This can *cap* the achieved request rate: if `--request-rate` is high but `--max-concurrency`
	is low, requests will queue behind the concurrency limit and the achieved send rate may be
	lower than the configured `--request-rate`.

Practical guidance:

- To approximate the configured `--request-rate`, set `--max-concurrency` sufficiently large
	for your model/latency (there is no `inf` value for this flag).

### Resolution/frames/steps (`--width`, `--height`, `--num-frames`, `--num-inference-steps`)

These flags can be used as global defaults (and also interact with the `trace` dataset):

- `--width`, `--height`: image/video resolution.
	- For `trace`: if either `--width` or `--height` is set, the script forces *all* requests to
		use the CLI-provided values (overriding the per-request values from the trace).
	- If they are not set, per-request `width/height` from the trace are used when available.

- `--num-frames`:
	- For `trace`: per-request `num_frames` (if present) takes precedence; otherwise it falls back
		to the CLI `--num-frames`.

- `--num-inference-steps`:
	- For `trace`: per-request `num_inference_steps` (if present) takes precedence; otherwise it
		falls back to the CLI `--num-inference-steps`.

If you are using `vbench` or `random`, these flags act as global defaults for all requests
constructed by that dataset loader.

Other useful flags:

- `--seed`: random seed (diffusion).
- `--fps`: frames per second (video).
- `--output-file`: write metrics JSON.
- `--disable-tqdm`: disable progress bar.
