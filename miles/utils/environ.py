import os


def use_legacy_rollout_v1() -> bool:
    """Always False: the deprecated v1 rollout has no flag, so no run can ask for it.

    Raises:
        AssertionError: ``MILES_USE_LEGACY_ROLLOUT_V1`` is set, which used to pick another rollout
            implementation out of the ambient environment, where no run record names it.
    """
    assert "MILES_USE_LEGACY_ROLLOUT_V1" not in os.environ, (
        "MILES_USE_LEGACY_ROLLOUT_V1 picked the deprecated v1 rollout out of the environment, where no "
        "run record names it; unset it, and run the v1 path from a test entrypoint if you still need it"
    )
    return False


def default_fp8_block_scaling_fp32_scales() -> str:
    """Default for NVTE_FP8_BLOCK_SCALING_FP32_SCALES, decided by hardware.

    On Blackwell (SM100+), TE emulates the blockwise FP8 recipe with MXFP8,
    which requires power-of-two scales, so FP32 scales must stay disabled.
    """
    import torch

    if not torch.cuda.is_available():
        return "1"
    major, _minor = torch.cuda.get_device_capability()
    return "0" if major >= 10 else "1"
