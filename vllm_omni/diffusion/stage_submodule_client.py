# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage client for lightweight submodule stages."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any

import zmq
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
from vllm_omni.diffusion.stage_submodule_proc import (
    complete_submodule_handshake,
    spawn_submodule_proc,
)
from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)
from vllm_omni.engine.stage_init_utils import StageMetadata, terminate_alive_proc
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType

logger = init_logger(__name__)


class StageSubModuleClient:
    """Communicates with StageSubModuleProc via ZMQ."""

    stage_type: str = "diffusion"
    replica_id: int = 0
    _sampling_params_to_dict = staticmethod(StageDiffusionClient._sampling_params_to_dict)

    def __init__(
        self,
        model: str,
        od_config: OmniDiffusionConfig,
        metadata: StageMetadata,
        stage_init_timeout: int,
        batch_size: int = 1,
    ) -> None:
        self.stage_id = metadata.stage_id
        self.replica_id = getattr(metadata, "replica_id", 0)
        self.final_output = metadata.final_output
        self.final_output_type = metadata.final_output_type
        self.default_sampling_params = metadata.default_sampling_params
        self.requires_multimodal_data = getattr(metadata, "requires_multimodal_data", False)
        self.custom_process_input_func = getattr(metadata, "custom_process_input_func", None)
        self.engine_input_source = getattr(metadata, "engine_input_source", [])

        proc, handshake_address, request_address, response_address = spawn_submodule_proc(model, od_config)
        complete_submodule_handshake(proc, handshake_address, stage_init_timeout)
        self._proc = proc

        self._zmq_ctx = zmq.Context()
        self._request_socket = self._zmq_ctx.socket(zmq.PUSH)
        self._request_socket.connect(request_address)
        self._response_socket = self._zmq_ctx.socket(zmq.PULL)
        self._response_socket.connect(response_address)

        self._encoder = OmniMsgpackEncoder()
        self._decoder = OmniMsgpackDecoder()
        self._output_queue: asyncio.Queue[OmniRequestOutput] = asyncio.Queue()
        self._rpc_results: dict[str, Any] = {}
        self._pending_rpcs: set[str] = set()
        self._engine_dead = False
        self._shutting_down = False

        logger.info(
            "[StageSubModuleClient] stage-%s [rep-%s] initialized (model_stage=%s, batch_size=%d)",
            self.stage_id,
            self.replica_id,
            getattr(od_config, "model_stage", None),
            batch_size,
        )

    def _drain_responses(self) -> None:
        while True:
            try:
                raw = self._response_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break

            msg = self._decoder.decode(raw)
            msg_type = msg.get("type")
            if msg_type == "result":
                self._output_queue.put_nowait(msg["output"])
            elif msg_type == "rpc_result":
                self._rpc_results[msg["rpc_id"]] = msg["result"]
            elif msg_type == "error":
                req_id = msg.get("request_id")
                rpc_id = msg.get("rpc_id")
                error_msg = msg.get("error")
                logger.error(
                    "[StageSubModuleClient] Stage-%s subprocess error for %s: %s",
                    self.stage_id,
                    rpc_id or req_id,
                    error_msg,
                )
                if rpc_id is not None and rpc_id in self._pending_rpcs:
                    self._rpc_results[rpc_id] = {"error": True, "reason": error_msg}
                elif req_id is not None:
                    self._output_queue.put_nowait(OmniRequestOutput.from_error(req_id, error_msg))

    async def add_request_async(
        self,
        request_id: str,
        prompt: OmniPromptType,
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        del kv_sender_info
        self.check_health()
        self._request_socket.send(
            self._encoder.encode(
                {
                    "type": "add_request",
                    "request_id": request_id,
                    "prompt": prompt,
                    "sampling_params": self._sampling_params_to_dict(sampling_params),
                }
            )
        )

    async def add_batch_request_async(
        self,
        request_id: str,
        prompts: list[OmniPromptType],
        sampling_params: OmniDiffusionSamplingParams,
        kv_sender_info: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        if len(prompts) == 1:
            await self.add_request_async(request_id, prompts[0], sampling_params, kv_sender_info=kv_sender_info)
            return

        self.check_health()
        self._output_queue.put_nowait(
            OmniRequestOutput.from_error(
                request_id,
                f"StageSubModuleClient only supports single-prompt batches, got {len(prompts)} prompts.",
            )
        )

    def get_diffusion_output_nowait(self) -> OmniRequestOutput | None:
        self._drain_responses()
        try:
            return self._output_queue.get_nowait()
        except asyncio.QueueEmpty:
            if self._engine_dead:
                raise EngineDeadError(f"Stage-{self.stage_id} submodule subprocess is dead")
            if not self._shutting_down and self._proc is not None and not self._proc.is_alive():
                self._engine_dead = True
                raise EngineDeadError(f"StageSubModuleProc died unexpectedly (exit code {self._proc.exitcode})")
            return None

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self._request_socket.send(self._encoder.encode({"type": "abort", "request_ids": list(request_ids)}))

    async def collective_rpc_async(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        if self._engine_dead:
            raise EngineDeadError(f"Stage-{self.stage_id} submodule subprocess is dead")
        rpc_id = uuid.uuid4().hex
        self._pending_rpcs.add(rpc_id)
        self._request_socket.send(
            self._encoder.encode(
                {
                    "type": "collective_rpc",
                    "rpc_id": rpc_id,
                    "method": method,
                    "timeout": timeout,
                    "args": list(args),
                    "kwargs": kwargs,
                }
            )
        )
        deadline = time.monotonic() + timeout if timeout else None
        try:
            while True:
                self._drain_responses()
                if rpc_id in self._rpc_results:
                    return self._rpc_results.pop(rpc_id)
                if self._proc is not None and not self._proc.is_alive():
                    self._engine_dead = True
                    raise EngineDeadError(
                        f"StageSubModuleProc died while waiting for collective_rpc "
                        f"{method!r} (exit code {self._proc.exitcode})"
                    )
                if deadline and time.monotonic() > deadline:
                    raise TimeoutError(f"collective_rpc_async '{method}' timed out after {timeout}s")
                await asyncio.sleep(0.01)
        finally:
            self._pending_rpcs.discard(rpc_id)

    def check_health(self) -> None:
        if self._engine_dead:
            raise EngineDeadError(f"Stage-{self.stage_id} submodule subprocess is dead")
        if self._proc is not None and not self._proc.is_alive():
            self._engine_dead = True
            raise EngineDeadError(
                f"Stage-{self.stage_id} submodule subprocess is not alive (exit code: {self._proc.exitcode})."
            )

    def shutdown(self) -> None:
        self._shutting_down = True
        try:
            self._request_socket.send(self._encoder.encode({"type": "shutdown"}))
        except Exception:
            pass

        if self._proc is not None and self._proc.is_alive():
            self._proc.join(timeout=10)
            terminate_alive_proc(self._proc)

        self._request_socket.close(linger=0)
        self._response_socket.close(linger=0)
        self._zmq_ctx.term()
