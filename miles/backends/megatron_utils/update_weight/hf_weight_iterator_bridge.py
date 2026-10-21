import dataclasses
import inspect
import itertools

import torch
import torch.distributed as dist

from miles.backends.megatron_utils.update_weight.hf_weight_iterator import (
    MegatronHfWeightIteratorBase,
    _iter_mm_tower_units,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils import megatron_bridge_utils
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.lora.utils import is_lora_weight_name

from ..megatron_to_hf import postprocess_hf_param
from ..megatron_to_hf.processors import quantize_params
from ..misc_utils import strip_param_name_prefix


class HfWeightIteratorBridge(MegatronHfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
        self._reused_tasks = None  # (structure, tasks) while --update-weight-reuse-conversion-tasks holds them

    def _iter_hf_param_units(self, weights, *, materialize):
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            conversion_tasks = self._conversion_tasks()
            conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)
            named_weights = self._bridge.export_hf_weights(
                self.model,
                cpu=False,
                conversion_tasks=conversion_tasks,
                merge_adapter_weights=False,
                **self._source_name_kwargs(self._bridge.export_hf_weights),
            )

            # Apply postprocess + quantization (when targeting a quantized rollout,
            # e.g. FP8 sglang): base weights are quantized to match the rollout's
            # storage format so update_weights_from_tensor lands real weight + scale
            # pairs.
            if not materialize:
                # The export's internal TP collectives must still run on every rank.
                for _ in named_weights:
                    pass
                return

            named_weights = self._postprocess_and_quantize(named_weights, "base")
            # Group by the (tuple of) source names so quantize's weight + scales land in one unit.
            for _megatron_name, group in itertools.groupby(named_weights, key=lambda item: item[2]):
                unit = [(h, w) for h, w, _m in group if not is_lora_weight_name(h)]
                if unit:
                    yield unit
        yield from _iter_mm_tower_units(self.args, materialize=materialize)

    def _conversion_tasks(self):
        """The bridge's conversion tasks, built once and reused while every rank's model structure holds.

        Their resolved mappings keep the pipeline tensor-spec and object caches that a fresh build empties, so later
        syncs skip those object collectives; each sync still exports the current weights, which
        ``_process_conversion_tasks`` substitutes. Whether to rebuild is decided by all ranks together: one rank
        reusing cached mappings while another rebuilds would issue different collectives.
        """
        if not self.args.update_weight_reuse_conversion_tasks:
            return self._bridge.get_conversion_tasks(self.model)
        structure = _model_structure(self.model)
        stale = torch.tensor([int(self._reused_tasks is None or self._reused_tasks[0] != structure)])
        dist.all_reduce(stale, op=dist.ReduceOp.MAX, group=get_gloo_group())
        if stale.item():
            self._reused_tasks = (structure, self._bridge.get_conversion_tasks(self.model))
        return self._reused_tasks[1]

    def _export_pp_local_lora(self, adapter):
        if adapter is None:
            return self._export_current_adapter()

        from megatron.bridge.peft.multi_lora_layers import expose_adapter_slot

        from ..lora.slots import slice_lora_to_rank

        with expose_adapter_slot(self.model, adapter.slot):
            named_tensors = self._export_current_adapter()
        return [(h, slice_lora_to_rank(h, w, adapter.rank)) for h, w in named_tensors]

    def _export_current_adapter(self) -> list:
        with megatron_bridge_utils.patch_megatron_model(self.model):
            named_weights = self._bridge.export_adapter_weights(
                self.model,
                cpu=False,
                show_progress=False,
                **self._source_name_kwargs(self._bridge.export_adapter_weights),
            )
            named_weights = self._postprocess_and_quantize(named_weights, "lora")
            return [(h, w) for h, w, _m in named_weights if is_lora_weight_name(h)]

    @staticmethod
    def _source_name_kwargs(export_fn) -> dict:
        """Request source Megatron names via ``with_megatron_names`` when the bridge accepts it (main only)."""
        if "with_megatron_names" in inspect.signature(export_fn).parameters:
            return {"with_megatron_names": True}
        return {}

    @staticmethod
    def _source_names(item) -> tuple:
        """Normalize the third field to a tuple of 0/1/N source Megatron names (a single str on the bridge branch)."""
        source = item[2] if len(item) > 2 else None
        if source is None:
            return ()
        if isinstance(source, str):
            return (source,)
        return tuple(source)

    def _postprocess_and_quantize(self, named_weights, weight_type: str):
        for item in named_weights:
            hf_param_name, weight = item[0], item[1]
            megatron_param_names = self._source_names(item)
            # Padding/quantization rules key on a Megatron name; packed grouped-expert tensors use the first source.
            megatron_param_name = megatron_param_names[0] if megatron_param_names else None
            hf_name = hf_param_name.replace(".base_layer.", ".")
            weight = postprocess_hf_param(
                args=self.args,
                megatron_param_name=megatron_param_name,
                hf_param_name=hf_name,
                param=weight,
            )
            if weight_type == "base" and self.quantization_config is not None and megatron_param_name is not None:
                # quantize_params expects the megatron name with the `module.module.`
                # prefix that the direct iterator uses; the bridge yields it without.
                # A tensor with no Megatron source (HF-only passthrough) is not a trainable weight: pass it through.
                qmegatron_name = f"module.module.{megatron_param_name}"
                for q_hf_name, q_weight in quantize_params(
                    self.args,
                    qmegatron_name,
                    [(hf_name, weight)],
                    self.quantization_config,
                    self.packed_weight_basenames,
                ):
                    yield q_hf_name, q_weight, megatron_param_names
            else:
                yield hf_name, weight, megatron_param_names


def _model_structure(model) -> tuple:
    """What reused conversion tasks depend on: each parameter's identity, name, shape and dtype, and the process
    groups their mappings were bound to."""
    parameters = tuple(
        (index, name, id(param), tuple(param.shape), str(param.dtype))
        for index, chunk in enumerate(model)
        for name, param in chunk.named_parameters()
    )
    state = get_parallel_state()
    return parameters, tuple(id(info.group) for info in (state.pp, state.tp, state.ep, state.etp))


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    def _handle_one(task):
        if task is None:
            # no HF mapping (e.g. Gemma-4 post_shared_expert_layernorm)
            return task
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        if weight_dict_key not in new_weight_dict:
            # buffer-like params (Gemma-4 layer_scalar/scale) aren't in optimizer state; keep as-is
            return task
        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()
        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLen(_handle_one, vanilla_conversion_tasks)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)
