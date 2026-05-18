# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Subprocess entry point for lightweight submodule stages."""

from __future__ import annotations

import asyncio
import signal
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq
import zmq.asyncio
from PIL import Image
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.utils.system_utils import get_mp_context
from vllm.v1.utils import shutdown

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.submodule_worker import SubModuleWorker
from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


@dataclass
class _PendingSubModuleRequest:
    request_id: str
    request: OmniDiffusionRequest
    future: Future
    result_index: int = 0
    aborted: bool = False


@dataclass
class _QueuedSubModuleRequest:
    request_id: str
    prompts: list[Any]
    sampling_params_dict: dict[str, Any]


class StageSubModuleProc:
    """Owns one lightweight submodule worker and serves ZMQ requests."""

    def __init__(self, model: str, od_config: OmniDiffusionConfig, batch_size: int = 1) -> None:
        del model
        self._od_config = od_config
        self._batch_size = max(1, int(batch_size or 1))
        self._batch_wait_s = 0.005
        self._worker: SubModuleWorker | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._closed = False

    def initialize(self) -> None:
        self._od_config.enrich_config()
        self._worker = SubModuleWorker(
            local_rank=0,
            rank=0,
            od_config=self._od_config,
        )
        self._worker.load_model(load_format=self._od_config.diffusion_load_format)
        self._executor = ThreadPoolExecutor(max_workers=1)
        logger.info(
            "StageSubModuleProc initialized with model=%s stage=%s batch_size=%d",
            self._od_config.model,
            self._od_config.model_stage,
            self._batch_size,
        )

    @staticmethod
    def _reconstruct_sampling_params(sampling_params_dict: dict[str, Any]) -> OmniDiffusionSamplingParams:
        lora_req = sampling_params_dict.get("lora_request")
        if lora_req is not None:
            from vllm.lora.request import LoRARequest

            if not isinstance(lora_req, LoRARequest):
                sampling_params_dict["lora_request"] = msgspec.convert(lora_req, LoRARequest)

        return OmniDiffusionSamplingParams(**sampling_params_dict)

    def _submit_stage_batch(
        self,
        queued_requests: list[_QueuedSubModuleRequest],
    ) -> list[_PendingSubModuleRequest]:
        assert self._worker is not None
        assert self._executor is not None
        requests: list[OmniDiffusionRequest] = []
        for queued in queued_requests:
            if len(queued.prompts) != 1:
                raise ValueError(
                    f"StageSubModuleProc only supports one prompt per request, got {len(queued.prompts)}."
                )
            sampling_params = self._reconstruct_sampling_params(dict(queued.sampling_params_dict))
            requests.append(
                OmniDiffusionRequest(
                    prompts=queued.prompts,
                    sampling_params=sampling_params,
                    request_ids=[queued.request_id] * len(queued.prompts),
                )
            )

        future = self._executor.submit(self._worker.execute_submodule_batch, requests)
        return [
            _PendingSubModuleRequest(
                request_id=queued.request_id,
                request=request,
                future=future,
                result_index=index,
            )
            for index, (queued, request) in enumerate(zip(queued_requests, requests, strict=True))
        ]

    @staticmethod
    def _has_active_request(pending_requests: dict[str, _PendingSubModuleRequest]) -> bool:
        return any(not pending.future.done() for pending in pending_requests.values())

    def _maybe_submit_ready_batch(
        self,
        ready_requests: list[_QueuedSubModuleRequest],
        ready_since: float | None,
        pending_requests: dict[str, _PendingSubModuleRequest],
        *,
        force: bool = False,
    ) -> float | None:
        if not ready_requests or self._has_active_request(pending_requests):
            return ready_since

        waited_long_enough = ready_since is not None and (time.monotonic() - ready_since) >= self._batch_wait_s
        if not force and len(ready_requests) < self._batch_size and not waited_long_enough:
            return ready_since

        batch = ready_requests[: self._batch_size]
        pendings = self._submit_stage_batch(batch)
        del ready_requests[: len(batch)]
        for pending in pendings:
            pending_requests[pending.request_id] = pending
        return time.monotonic() if ready_requests else None

    def _build_outputs(
        self,
        pending: _PendingSubModuleRequest,
        output: DiffusionOutput,
    ) -> OmniRequestOutput:
        request = pending.request
        prompt = request.prompts[0] if request.prompts else None
        metrics = {}
        multimodal_output = output.multimodal_output or {}

        if output.output is None:
            return OmniRequestOutput.from_diffusion(
                request_id=pending.request_id,
                images=[],
                prompt=prompt,
                metrics=metrics,
                multimodal_output=multimodal_output,
                final_output_type=f"stage_{self._od_config.model_stage}",
                stage_durations=output.stage_durations,
                peak_memory_mb=output.peak_memory_mb,
            )

        output_data = output.output
        if isinstance(output_data, Image.Image):
            images = [output_data]
            latents = None
            final_output_type = "image"
        elif isinstance(output_data, list):
            images = output_data
            latents = None
            final_output_type = "image"
        elif isinstance(output_data, torch.Tensor):
            images = []
            latents = output_data
            final_output_type = "latents"
        else:
            images = []
            latents = None
            final_output_type = "image"

        return OmniRequestOutput.from_diffusion(
            request_id=pending.request_id,
            images=images,
            prompt=prompt,
            metrics=metrics,
            latents=latents,
            multimodal_output=multimodal_output,
            final_output_type=final_output_type,
            stage_durations=output.stage_durations,
            peak_memory_mb=output.peak_memory_mb,
        )

    async def _drain_completed_requests(
        self,
        pending_requests: dict[str, _PendingSubModuleRequest],
        response_socket,
        encoder: OmniMsgpackEncoder,
    ) -> None:
        completed = [rid for rid, pending in pending_requests.items() if pending.future.done()]
        for request_id in completed:
            pending = pending_requests.pop(request_id)
            if pending.aborted:
                continue
            try:
                outputs = pending.future.result()
                output = outputs[pending.result_index]
                if output.error:
                    await response_socket.send(
                        encoder.encode({"type": "error", "request_id": request_id, "error": output.error})
                    )
                    continue
                result = self._build_outputs(pending, output)
                await response_socket.send(encoder.encode({"type": "result", "output": result}))
            except Exception as exc:
                logger.exception("Submodule request %s failed: %s", request_id, exc)
                await response_socket.send(
                    encoder.encode({"type": "error", "request_id": request_id, "error": str(exc)})
                )

    async def _handle_collective_rpc(
        self,
        method: str,
        timeout: float | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        del timeout
        assert self._worker is not None
        if method == "profile":
            return None
        func = getattr(self._worker, method)
        return func(*args, **(kwargs or {}))

    async def run_loop(
        self,
        request_address: str,
        response_address: str,
    ) -> None:
        ctx = zmq.asyncio.Context()
        request_socket = ctx.socket(zmq.PULL)
        request_socket.bind(request_address)
        response_socket = ctx.socket(zmq.PUSH)
        response_socket.bind(response_address)

        encoder = OmniMsgpackEncoder()
        decoder = OmniMsgpackDecoder()
        pending_requests: dict[str, _PendingSubModuleRequest] = {}
        ready_requests: list[_QueuedSubModuleRequest] = []
        ready_since: float | None = None
        poller = zmq.asyncio.Poller()
        poller.register(request_socket, zmq.POLLIN)
        shutdown_requested = False

        try:
            while not shutdown_requested:
                events = dict(await poller.poll(timeout=10))
                if request_socket in events:
                    while True:
                        try:
                            raw = await request_socket.recv(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break

                        msg = decoder.decode(raw)
                        msg_type = msg.get("type")
                        if msg_type == "add_request":
                            request_id = msg["request_id"]
                            ready_requests.append(
                                _QueuedSubModuleRequest(
                                    request_id,
                                    [msg["prompt"]],
                                    msg["sampling_params"],
                                )
                            )
                            if ready_since is None:
                                ready_since = time.monotonic()
                        elif msg_type == "add_batch_request":
                            request_id = msg["request_id"]
                            ready_requests.append(
                                _QueuedSubModuleRequest(
                                    request_id,
                                    msg["prompts"],
                                    msg["sampling_params"],
                                )
                            )
                            if ready_since is None:
                                ready_since = time.monotonic()
                        elif msg_type == "abort":
                            abort_ids = set(msg.get("request_ids", []))
                            if abort_ids:
                                ready_requests[:] = [
                                    queued for queued in ready_requests if queued.request_id not in abort_ids
                                ]
                            for rid in abort_ids:
                                pending = pending_requests.get(rid)
                                if pending is not None:
                                    pending.aborted = True
                        elif msg_type == "collective_rpc":
                            rpc_id = msg["rpc_id"]
                            try:
                                result = await self._handle_collective_rpc(
                                    msg["method"],
                                    msg.get("timeout"),
                                    tuple(msg.get("args", ())),
                                    msg.get("kwargs", {}),
                                )
                                await response_socket.send(
                                    encoder.encode({"type": "rpc_result", "rpc_id": rpc_id, "result": result})
                                )
                            except Exception as exc:
                                logger.exception("Submodule collective RPC %s failed: %s", msg["method"], exc)
                                await response_socket.send(
                                    encoder.encode({"type": "error", "rpc_id": rpc_id, "error": str(exc)})
                                )
                        elif msg_type == "shutdown":
                            shutdown_requested = True
                            break

                await self._drain_completed_requests(pending_requests, response_socket, encoder)
                try:
                    ready_since = self._maybe_submit_ready_batch(
                        ready_requests,
                        ready_since,
                        pending_requests,
                        force=shutdown_requested,
                    )
                except Exception as exc:
                    failed = ready_requests[: self._batch_size]
                    del ready_requests[: self._batch_size]
                    ready_since = time.monotonic() if ready_requests else None
                    logger.exception("Submodule batch failed to submit: %s", exc)
                    for queued in failed:
                        await response_socket.send(
                            encoder.encode({"type": "error", "request_id": queued.request_id, "error": str(exc)})
                        )
        finally:
            request_socket.close()
            response_socket.close()
            ctx.term()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            try:
                self._worker.shutdown()
            except Exception as exc:
                logger.warning("Error closing submodule worker: %s", exc)
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    @classmethod
    def run_submodule_proc(
        cls,
        model: str,
        od_config: OmniDiffusionConfig,
        batch_size: int,
        handshake_address: str,
        request_address: str,
        response_address: str,
    ) -> None:
        shutdown_requested = False

        def signal_handler(signum: int, frame: Any) -> None:
            del signum, frame
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        proc = cls(model, od_config, batch_size=batch_size)
        try:
            proc.initialize()
            handshake_ctx = zmq.Context()
            handshake_socket = handshake_ctx.socket(zmq.DEALER)
            handshake_socket.connect(handshake_address)
            handshake_socket.send(msgspec.msgpack.encode({"status": "READY"}))
            handshake_socket.close()
            handshake_ctx.term()
            asyncio.run(proc.run_loop(request_address, response_address))
        except SystemExit:
            logger.debug("StageSubModuleProc exiting.")
            raise
        except Exception:
            logger.exception("StageSubModuleProc encountered a fatal error.")
            raise
        finally:
            proc.close()


def spawn_submodule_proc(
    model: str,
    od_config: OmniDiffusionConfig,
    batch_size: int = 1,
) -> tuple[BaseProcess, str, str, str]:
    handshake_address = get_open_zmq_ipc_path()
    request_address = get_open_zmq_ipc_path()
    response_address = get_open_zmq_ipc_path()

    ctx = get_mp_context()
    proc = ctx.Process(
        target=StageSubModuleProc.run_submodule_proc,
        name="StageSubModuleProc",
        kwargs={
            "model": model,
            "od_config": od_config,
            "batch_size": batch_size,
            "handshake_address": handshake_address,
            "request_address": request_address,
            "response_address": response_address,
        },
    )
    proc.start()
    deadline = time.monotonic() + 10
    while not proc.is_alive():
        if proc.exitcode is not None:
            raise RuntimeError(f"StageSubModuleProc failed to start (exit code {proc.exitcode})")
        if time.monotonic() > deadline:
            raise TimeoutError("StageSubModuleProc did not become alive within 10s")
        time.sleep(0.01)
    return proc, handshake_address, request_address, response_address


def complete_submodule_handshake(
    proc: BaseProcess,
    handshake_address: str,
    handshake_timeout: int,
) -> None:
    try:
        _perform_submodule_handshake(proc, handshake_address, handshake_timeout)
    except Exception:
        shutdown([proc])
        raise


def _perform_submodule_handshake(
    proc: BaseProcess,
    handshake_address: str,
    handshake_timeout: int,
) -> None:
    with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
        poller = zmq.Poller()
        poller.register(handshake_socket, zmq.POLLIN)
        poller.register(proc.sentinel, zmq.POLLIN)
        timeout_ms = handshake_timeout * 1000
        while True:
            events = dict(poller.poll(timeout=timeout_ms))
            if not events:
                raise TimeoutError(
                    f"Timed out waiting for READY from StageSubModuleProc after {handshake_timeout}s. "
                    f"Consider increasing `stage_init_timeout` for large submodule stages."
                )
            if handshake_socket in events:
                _identity, raw = handshake_socket.recv_multipart()
                msg = msgspec.msgpack.decode(raw)
                if msg.get("status") == "READY":
                    return
                raise RuntimeError(f"Expected READY, got: {msg}")
            if proc.exitcode is not None:
                raise RuntimeError(f"StageSubModuleProc died during handshake (exit code {proc.exitcode})")
