from argparse import Namespace
from unittest.mock import Mock

import pytest
import torch

from miles.utils import dumper_utils


@pytest.mark.parametrize("phase", [dumper_utils.DumperPhase.FWD_ONLY, dumper_utils.DumperPhase.FWD_BWD])
def test_unselected_rollout_never_configures_hooks_or_reconstructs_gradients(
    monkeypatch: pytest.MonkeyPatch, phase: dumper_utils.DumperPhase
) -> None:
    configure = Mock(side_effect=AssertionError("Unselected rollout configured the dumper"))
    register = Mock(side_effect=AssertionError("Unselected rollout registered hooks"))
    gradients = Mock(side_effect=AssertionError("Unselected rollout reconstructed gradients"))
    monkeypatch.setattr(dumper_utils, "_get_phase_override_configs", configure)
    monkeypatch.setattr(dumper_utils.dumper, "register_non_intrusive_dumper", register)
    monkeypatch.setattr(dumper_utils, "_build_full_grad_getter", gradients)
    wrapper = dumper_utils.DumperMegatronUtil(Namespace(dumper_train_rollouts=[3]), [], phase, rollout_id=2)

    def forward() -> None:
        pass

    assert wrapper.enabled is False
    assert wrapper.wrap_forward_step(forward) is forward
    wrapper.finalize([])
    configure.assert_not_called()
    register.assert_not_called()
    gradients.assert_not_called()


@pytest.mark.parametrize("selection", [None, [2], [2, 3]])
def test_selected_rollout_preserves_the_existing_configuration_path(
    monkeypatch: pytest.MonkeyPatch, selection: list[int] | None
) -> None:
    overrides = {"enable": True}
    model = torch.nn.Linear(2, 1)
    configure = Mock(return_value=True)
    register = Mock()
    monkeypatch.setattr(dumper_utils, "_get_phase_override_configs", Mock(return_value=overrides))
    monkeypatch.setattr(dumper_utils.DumperMegatronUtil, "_configure", configure)
    monkeypatch.setattr(dumper_utils.dumper, "register_non_intrusive_dumper", register)
    args = Namespace(dumper_train_rollouts=selection)
    wrapper = dumper_utils.DumperMegatronUtil(args, [model], dumper_utils.DumperPhase.FWD_BWD, rollout_id=2)
    assert wrapper.enabled is True
    configure.assert_called_once_with(
        args, phase=dumper_utils.DumperPhase.FWD_BWD, rollout_id=2, store_prefix="", overrides=overrides
    )
    register.assert_called_once_with(model)


def test_older_callers_without_the_selection_option_keep_enabled_dumps(monkeypatch: pytest.MonkeyPatch) -> None:
    configure = Mock(return_value=True)
    register = Mock()
    model = torch.nn.Linear(2, 1)
    monkeypatch.setattr(dumper_utils, "_get_phase_override_configs", Mock(return_value={"enable": True}))
    monkeypatch.setattr(dumper_utils.DumperMegatronUtil, "_configure", configure)
    monkeypatch.setattr(dumper_utils.dumper, "register_non_intrusive_dumper", register)
    wrapper = dumper_utils.DumperMegatronUtil(Namespace(), [model], dumper_utils.DumperPhase.FWD_BWD, rollout_id=2)
    assert wrapper.enabled is True
    configure.assert_called_once()
    register.assert_called_once_with(model)


def test_selected_rollout_does_not_enable_a_disabled_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    register = Mock()
    monkeypatch.setattr(dumper_utils, "_get_phase_override_configs", Mock(return_value={"enable": False}))
    monkeypatch.setattr(dumper_utils.dumper, "register_non_intrusive_dumper", register)
    wrapper = dumper_utils.DumperMegatronUtil(
        Namespace(dumper_train_rollouts=[2]), [], dumper_utils.DumperPhase.FWD_BWD, rollout_id=2
    )
    assert wrapper.enabled is False
    wrapper.finalize([])
    register.assert_not_called()
