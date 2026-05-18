"""
vLLM-Omni: Multi-modality models inference and serving with
non-autoregressive structures.

This package extends vLLM beyond traditional text-based, autoregressive
generation to support multi-modality models with non-autoregressive
structures and non-textual outputs.

Architecture:
- 🟡 Modified: vLLM components modified for multimodal support
- 🔴 Added: New components for multimodal and non-autoregressive
  processing
"""

# We import version early, because it will warn if vLLM / vLLM Omni
# are not using the same major + minor version (if vLLM is installed).
# We should do this before applying patch, because vLLM imports might
# throw in patch if the versions differ.
from .version import __version__, __version_tuple__  # isort:skip # noqa: F401


def _apply_vllm_inputs_compat() -> None:
    """Expose renamed vLLM input helpers when running against older vLLM."""
    try:
        import sys
        import types

        import vllm.inputs as inputs
        from vllm.inputs import data
    except ModuleNotFoundError:
        return

    aliases = {
        "TokensInput": getattr(data, "TokensInput", getattr(data, "TokenInputs", dict)),
        "SingletonInput": getattr(data, "SingletonInput", getattr(data, "SingletonInputs", dict)),
        "EmbedsInput": getattr(data, "EmbedsInput", getattr(data, "EmbedsInputs", dict)),
        "MultiModalInput": getattr(data, "MultiModalInput", getattr(data, "MultiModalInputs", dict)),
        "MultiModalDataDict": getattr(data, "MultiModalDataDict", dict),
        "ModalityData": getattr(data, "ModalityData", object),
        "tokens_input": getattr(data, "tokens_input", getattr(data, "token_inputs", None)),
    }
    for name, value in aliases.items():
        if value is not None and not hasattr(inputs, name):
            setattr(inputs, name, value)

    engine_mod = sys.modules.get("vllm.inputs.engine")
    if engine_mod is None:
        engine_mod = types.ModuleType("vllm.inputs.engine")
        sys.modules["vllm.inputs.engine"] = engine_mod
    for name, value in aliases.items():
        if value is not None and not hasattr(engine_mod, name):
            setattr(engine_mod, name, value)

    llm_mod = sys.modules.get("vllm.inputs.llm")
    if llm_mod is None:
        llm_mod = types.ModuleType("vllm.inputs.llm")
        sys.modules["vllm.inputs.llm"] = llm_mod
    if not hasattr(llm_mod, "MultiModalDataDict"):
        llm_mod.MultiModalDataDict = aliases["MultiModalDataDict"]


_apply_vllm_inputs_compat()

try:
    from . import patch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name != "vllm":
        raise
    # Allow importing vllm_omni without vllm (e.g., documentation builds)
    patch = None  # type: ignore

# Register custom configs (AutoConfig, AutoTokenizer) as early as possible.
from vllm_omni.transformers_utils import configs as _configs  # noqa: F401, E402
from vllm_omni.transformers_utils import parsers as _parsers  # noqa: F401, E402

from .config import OmniModelConfig


def __getattr__(name: str):
    # Lazy import for AsyncOmni and Omni to avoid pulling in heavy
    # dependencies (vllm model_loader → fused_moe → pynvml) at package
    # import time.  This prevents crashes in lightweight subprocesses
    # (e.g. model-architecture inspection) that lack a CUDA context.
    # See: https://github.com/vllm-project/vllm-omni/issues/1793
    if name == "AsyncOmni":
        from .entrypoints.async_omni import AsyncOmni

        return AsyncOmni
    if name == "Omni":
        from .entrypoints.omni import Omni

        return Omni
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "__version_tuple__",
    # Main components
    "Omni",
    "AsyncOmni",
    # Configuration
    "OmniModelConfig",
    # All other components are available through their respective modules
    # processors.*, schedulers.*, executors.*, etc.
]
