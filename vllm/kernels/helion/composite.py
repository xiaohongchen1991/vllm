# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Composite Helion kernels: multiple ``@helion.kernel`` children tuned as one.

A composite kernel is defined by a *body* -- a plain function whose first
parameter is ``run`` followed by the composite's full argument list -- that
calls child helion kernels through ``run(child, *args)``. ``run`` is the
framework's single interception point:

* **Binding** (``CompositeKernel.bind``): the body executes once with root
  tunables at their defaults; each ``run`` records the call site, binds the
  child (``child.bind(args)``), and executes it at the child's default config
  so intermediates propagate to later sites. The result is a
  ``BoundCompositeKernel`` holding the discovered sites.
* **Configured execution** (``CompositeKernel.configure``): the body executes
  with each site running its section of a joint config; sites bind lazily per
  actual arguments, so a config may drive the body down branches whose sites
  were never bound (e.g. a root tunable flipping a layout).

Root tunables (``run.tunable(name, fragment)``) are composite-level knobs
searched like any other dimension. Boolean root tunables may be structural
(they may select which child sites execute); the worst case is that the search
explores a section whose site is skipped under another root value, which is
harmless.

The joint config is ``{"root": {...}, "children": {"<kernel>.<occurrence>":
{...}}}`` -- one section per call site, named after the child kernel and its
occurrence index in body execution order (the same kernel may be called
multiple times with independently tuned configs).

``BoundCompositeKernel`` implements helion's ``_AutotunableKernel`` protocol
(``config_spec``/``settings``/``env``/``configs``/``compile_config``), so
helion's own autotuners (e.g. ``LFBOTreeSearch``) can drive the composite as
if it were a single kernel: every candidate is a complete joint config
measured by running the whole body end to end.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import helion
import torch
from helion.autotuner import LFBOTreeSearch
from helion.autotuner.config_fragment import ConfigSpecFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.effort_profile import get_effort_profile

# Args of these positions are mutated by the body and get per-candidate clones
# in accuracy checks (mirrors helion's mutates_args semantics).
JointConfig = Mapping[str, Any]


def is_composite_body(body: Callable) -> bool:
    """Whether ``body`` defines a composite kernel (first parameter ``run``)."""
    params = list(inspect.signature(body).parameters)
    return bool(params) and params[0] == "run"


def body_signature(body: Callable) -> inspect.Signature:
    """The composite's outward signature: the body minus its ``run`` parameter."""
    sig = inspect.signature(body)
    params = [p for name, p in sig.parameters.items() if name != "run"]
    # Preserve the return annotation: torch's infer_schema requires one.
    return inspect.Signature(params, return_annotation=sig.return_annotation)


def _kernel_name(kernel: Any) -> str:
    return getattr(kernel, "name", None) or getattr(kernel, "__name__", "kernel")


def _joint_dict(config: JointConfig | helion.Config | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, helion.Config):
        return dict(config.config)
    return dict(config)


def _section_config(section: Any) -> helion.Config:
    if isinstance(section, helion.Config):
        return section
    return helion.Config.from_dict(section)


def _cfg_key(section: Any) -> str:
    """Deterministic cache key for a config section."""
    if isinstance(section, helion.Config):
        section = section.config
    return repr(sorted(section.items())) if isinstance(section, dict) else repr(section)


def _adapt_section(spec: Any, section: Any) -> helion.Config:
    """Fit a section proposed under another site's spec to ``spec``.

    A section may reach a site that bound lazily under different arguments
    (e.g. a root tunable flipping the body to a layout with one more memory
    operation), so its structure-length lists (indexing, eviction policies,
    loop orders, ...) may not match this spec's. Pad with / truncate to the
    spec's own defaults, then normalize against the spec so genuinely invalid
    values still fail (the search skips such candidates).
    """
    values = section.config if isinstance(section, helion.Config) else dict(section)
    default = spec.default_config().config
    adapted: dict[str, Any] = {}
    for key, value in values.items():
        default_value = default.get(key)
        if (
            isinstance(value, list)
            and isinstance(default_value, list)
            and len(value) != len(default_value)
        ):
            if len(value) < len(default_value):
                value = [*value, *default_value[len(value) :]]
            else:
                value = value[: len(default_value)]
        adapted[key] = value
    return spec.normalized_config(helion.Config.from_dict(adapted))


class _Site:
    """One child-kernel call site discovered during binding."""

    def __init__(self, name: str, kernel: Any, bound: Any):
        self.name = name
        self.kernel = kernel
        self.bound = bound

    @property
    def spec(self) -> Any:
        return self.bound.config_spec


class _BindRun:
    """Bind-mode ``run``: records sites and propagates at default configs.

    Also the root-tunable registration point: ``tunable`` registers the
    fragment on the composite (idempotently, for cross-execution type
    checking) and captures it on this bind -- root fragments may be
    shape-dependent (e.g. a block-size range derived from num_tokens), so the
    bound spec must see the fragments of *this* bind's shapes, not whichever
    shape first registered the name. The value for this bind is always the
    fragment default -- binding only ever sees the default root branch.
    """

    def __init__(self, composite: CompositeKernel):
        self._composite = composite
        self._occurrences: dict[int, int] = {}
        self.sites: list[_Site] = []
        self.root_fragments: dict[str, ConfigSpecFragment] = {}

    def tunable(self, name: str, fragment: ConfigSpecFragment) -> Any:
        self._composite._register_root_tunable(name, fragment)
        self.root_fragments[name] = fragment
        return fragment.default()

    def __call__(self, child: Any, *child_args: Any) -> Any:
        if any(
            isinstance(a, torch.Tensor)
            and type(a).__module__.startswith("torch._subclasses.fake_tensor")
            for a in child_args
        ):
            raise NotImplementedError(
                "Composite kernel binding requires real tensors; run the body "
                "eagerly (config-key discovery only needs get_inputs())."
            )
        occurrence = self._occurrences.get(id(child), 0)
        self._occurrences[id(child)] = occurrence + 1
        name = f"{_kernel_name(child)}.{occurrence}"
        bound = child.bind(child_args)
        self.sites.append(_Site(name=name, kernel=child, bound=bound))
        # Propagate intermediates: execute the child at its default config.
        compiled = bound.compile_config(bound.config_spec.default_config())
        return compiled(*child_args)


class _ConfiguredRun:
    """Configured-mode ``run``: runs each site on its section of a joint config.

    Sites bind lazily per actual arguments (helion caches binds by
    specialization), so the body may follow branches whose sites differ from --
    or were absent in -- the bind pass. Compiled sections are cached on the
    shared ``compiled_cache`` so repeated body executions reuse them.
    """

    def __init__(
        self,
        composite: CompositeKernel,
        joint: dict[str, Any],
        compiled_cache: dict[tuple[str, int, str], Callable],
    ):
        self._composite = composite
        self._root_values = joint.get("root", {})
        self._sections = joint.get("children", {})
        self._compiled_cache = compiled_cache
        self._occurrences: dict[int, int] = {}

    def tunable(self, name: str, fragment: ConfigSpecFragment) -> Any:
        self._composite._register_root_tunable(name, fragment)
        return self._root_values.get(name, fragment.default())

    def __call__(self, child: Any, *child_args: Any) -> Any:
        occurrence = self._occurrences.get(id(child), 0)
        self._occurrences[id(child)] = occurrence + 1
        name = f"{_kernel_name(child)}.{occurrence}"
        bound = child.bind(child_args)
        raw = self._sections.get(name)
        key = (name, id(bound), _cfg_key(raw))
        compiled = self._compiled_cache.get(key)
        if compiled is None:
            if raw is None:
                section = bound.config_spec.default_config()
            else:
                # The section was generated for the bind-pass site, which may
                # have a different structure than this (lazily bound) site.
                section = _adapt_section(bound.config_spec, raw)
            compiled = bound.compile_config(section)
            self._compiled_cache[key] = compiled
        return compiled(*child_args)


class _CompositeBackendProxy:
    """Backend facade for ``CompositeConfigSpec``: delegates everything to a
    child backend except config viability, which validates per site section."""

    def __init__(self, child_backend: Any, sites: Sequence[_Site]):
        self._child_backend = child_backend
        self._sites = sites
        self.name = child_backend.name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._child_backend, name)

    def autotune_config_is_viable(self, config_spec: Any, config: Any) -> bool:
        joint = _joint_dict(config)
        sections = joint.get("children", {})
        for site in self._sites:
            section = sections.get(site.name)
            if section is None:
                continue
            spec = site.spec
            if not spec.backend.autotune_config_is_viable(
                spec, _section_config(section)
            ):
                return False
        return True

    def should_deduplicate_generated_sources(self, config_spec: Any) -> bool:
        return False


class CompositeConfigSpec:
    """Flat search space of a composite: root tunables plus every site's
    child space, namespaced ``root.<name>`` / ``children.<site>.<name>``."""

    def __init__(
        self, root_fragments: Mapping[str, ConfigSpecFragment], sites: Sequence[_Site]
    ):
        self._root = dict(root_fragments)
        self._sites = list(sites)
        child_backend = sites[0].spec.backend
        self.backend = _CompositeBackendProxy(child_backend, sites)
        self.backend_name = child_backend.name
        self.user_defined_tunables: dict[str, ConfigSpecFragment] = {
            **{f"root.{k}": v for k, v in self._root.items()},
            **{
                f"children.{s.name}.{k}": v
                for s in self._sites
                for k, v in s.spec.user_defined_tunables.items()
            },
        }
        self.compiler_seed_configs: tuple[Any, ...] = ()
        self.compiler_seed_timeout_retry_repetitions = None
        self.cute_flash_search_enabled = False
        self.compiler_coverage_groups: tuple[Any, ...] = ()
        # No per-tensor block-numel constraints at the composite level (the
        # children enforce their own).
        self.tensor_numel_constraints: tuple[Any, ...] = ()

    def default_config(self) -> helion.Config:
        root = {name: frag.default() for name, frag in self._root.items()}
        children = {s.name: dict(s.spec.default_config().config) for s in self._sites}
        return helion.Config.from_dict({"root": root, "children": children})

    def normalized_config(self, config: Any) -> helion.Config:
        joint = _joint_dict(config)
        root = {name: frag.default() for name, frag in self._root.items()}
        root.update(joint.get("root", {}))
        sections = joint.get("children", {})
        children = {}
        for site in self._sites:
            section = sections.get(site.name)
            if section is None:
                children[site.name] = dict(site.spec.default_config().config)
            else:
                children[site.name] = dict(
                    site.spec.normalized_config(_section_config(section)).config
                )
        return helion.Config.from_dict({"root": root, "children": children})

    def create_config_generation(
        self,
        *,
        overrides: Mapping[str, Any] | None = None,
        advanced_controls_files: list[str] | None = None,
        process_group_name: str | None = None,
        compiler_coverage_enabled: bool = True,
    ) -> CompositeConfigGeneration:
        return CompositeConfigGeneration(self, overrides=overrides)

    def autotune_reference_config(self) -> helion.Config:
        return self.default_config()

    def _flat_fields(self) -> dict[str, Any]:
        """No compiler-coverage fields: coverage grouping is a single-kernel
        concept, so the composite exposes an empty field mapping (the
        generation's coverage sampler is thereby a no-op)."""
        return {}

    def normalize(self, config: Any) -> None:
        normalized = self.normalized_config(config)
        if isinstance(config, helion.Config):
            config.config.clear()
            config.config.update(normalized.config)

    def shrink_block_sizes_once(self, config: Any) -> helion.Config | None:
        """Halve the largest block size of one site section (compile backoff)."""
        joint = _joint_dict(config)
        sections = joint.get("children", {})
        for site in self._sites:
            section = sections.get(site.name)
            if section is None:
                continue
            shrunk = site.spec.shrink_block_sizes_once(_section_config(section))
            if shrunk is not None:
                sections[site.name] = dict(shrunk.config)
                return helion.Config.from_dict(joint)
        return None

    def cache_fingerprint_hash(
        self, *, advanced_controls_files: list[str] | None = None
    ) -> str:
        return f"composite[{','.join(s.name for s in self._sites)}]"

    def projected_cache_fingerprint_hash(
        self, *, advanced_controls_files: list[str] | None = None
    ) -> str:
        return self.cache_fingerprint_hash(
            advanced_controls_files=advanced_controls_files
        )


class CompositeConfigGeneration(ConfigGeneration):
    """ConfigGeneration over the composite's flat space.

    The flat layout is the root fragments followed by each site's own
    ``ConfigGeneration.flat_spec``. Flatten/unflatten split and join at those
    fixed boundaries, delegating per site to the child generation, so fragment
    behavior (pattern neighbors, encoding, viability) is inherited from the
    children rather than reimplemented.
    """

    def __init__(
        self,
        config_spec: CompositeConfigSpec,
        *,
        overrides: Mapping[str, Any] | None = None,
    ):
        # Deliberately skip ConfigGeneration.__init__: it is coupled to a
        # single kernel's ConfigSpec. The attributes below are the surface the
        # search touches (see helion/autotuner/base_search.py and
        # surrogate_pattern_search.py).
        self.config_spec = config_spec
        self._root: dict[str, ConfigSpecFragment] = dict(config_spec._root)
        self._sites: list[_Site] = list(config_spec._sites)
        self._child_gens = [
            site.spec.create_config_generation() for site in self._sites
        ]
        self._root_names = list(self._root)

        self.flat_spec: list[ConfigSpecFragment] = list(self._root.values())
        self._child_offsets: list[int] = []
        for gen in self._child_gens:
            self._child_offsets.append(len(self.flat_spec))
            self.flat_spec.extend(gen.flat_spec)

        self.block_size_indices: list[int] = []
        self.overridden_flat_indices: set[int] = set()
        self._key_to_flat_indices: dict[str, tuple[list[int], bool]] = {}
        for i, name in enumerate(self._root_names):
            self._key_to_flat_indices[f"root.{name}"] = ([i], False)
        for off, site, gen in zip(
            self._child_offsets, self._sites, self._child_gens, strict=True
        ):
            self.block_size_indices.extend(
                off + i for i in getattr(gen, "block_size_indices", [])
            )
            self.overridden_flat_indices.update(
                off + i for i in getattr(gen, "overridden_flat_indices", [])
            )
            for key, (indices, is_sequence) in gen._key_to_flat_indices.items():
                self._key_to_flat_indices[f"children.{site.name}.{key}"] = (
                    [off + i for i in indices],
                    is_sequence,
                )
        warp = (
            getattr(self._child_gens[0], "num_warps_index", -1)
            if self._child_gens
            else -1
        )
        self.num_warps_index = (
            self._child_offsets[0] + warp
            if self._child_gens and warp is not None and warp >= 0
            else -1
        )

        self.invalid_config_count = 0
        self.process_group_name = None
        self._advanced_controls_files = None
        self._override_values = dict(overrides or {})
        self._initial_sampling = False
        self.compiler_coverage_enabled = False
        self._field_view = None
        # Inert state mirroring ConfigGeneration.__init__: the cute/flash
        # machinery never activates on a composite (triton backend, no
        # compiler-coverage groups), but inherited methods read these
        # unconditionally (_flat_fields, _repair_cute_num_threads,
        # shrink_config).
        self._flash_pipeline_family_override = None
        self.min_block_size = 1
        self.num_threads_indices: list[int] = []
        self._cute_num_thread_block_pairs: list[tuple[int, int]] = []
        self._cute_block_index_by_id: dict[int, int] = {}
        self._cute_num_thread_index_by_id: dict[int, int] = {}
        self._cute_loop_order_index_by_id: dict[int, int] = {}
        self._cute_flatten_loop_groups: list[tuple[int, list[int]]] = []

    def flatten(self, config: Any) -> list[Any]:
        joint = _joint_dict(config)
        root_values = joint.get("root", {})
        sections = joint.get("children", {})
        flat = [
            root_values.get(name, frag.default())
            for name, frag in zip(self._root_names, self._root.values(), strict=True)
        ]
        for off, site, gen in zip(
            self._child_offsets, self._sites, self._child_gens, strict=True
        ):
            section = sections.get(site.name)
            child_config = (
                _section_config(section)
                if section is not None
                else site.spec.default_config()
            )
            child_flat = gen.flatten(child_config)
            flat.extend(child_flat)
        return flat

    def unflatten(self, flat_values: list[Any]) -> helion.Config:
        if len(flat_values) != len(self.flat_spec):
            raise ValueError(
                f"Expected {len(self.flat_spec)} flat values, got {len(flat_values)}."
            )
        n_root = len(self._root_names)
        root = {name: flat_values[i] for i, name in enumerate(self._root_names)}
        children: dict[str, Any] = {}
        pos = n_root
        for site, gen in zip(self._sites, self._child_gens, strict=True):
            end = pos + len(gen.flat_spec)
            children[site.name] = dict(gen.unflatten(list(flat_values[pos:end])).config)
            pos = end
        return helion.Config.from_dict({"root": root, "children": children})

    def canonicalize_flat(
        self, flat_values: list[Any]
    ) -> tuple[list[Any], helion.Config]:
        config = self.unflatten(copy.deepcopy(flat_values))
        return self.flatten(config), config

    def default_flat(self) -> list[Any]:
        return self.flatten(self.config_spec.default_config())

    def seed_flat_config_pairs(
        self, log_func: Callable[[str], None] | None = None
    ) -> list[tuple[list[Any], helion.Config]]:
        return []


class BoundCompositeKernel:
    """A ``CompositeKernel`` bound to concrete args; helion-autotunable.

    Implements the ``_AutotunableKernel`` protocol so helion's autotuners
    drive the composite as one kernel: ``compile_config(joint)`` returns a
    callable running the whole body with that joint config.
    """

    def __init__(
        self,
        kernel: CompositeKernel,
        args: Sequence[Any],
        sites: list[_Site],
        root_fragments: Mapping[str, ConfigSpecFragment] | None = None,
    ):
        self.kernel = kernel
        self.args = tuple(args)
        self.sites = sites
        self.settings = kernel.settings
        self._root_fragments = (
            dict(root_fragments) if root_fragments is not None else None
        )
        self._config: dict[str, Any] | None = None
        self._config_spec: CompositeConfigSpec | None = None

    # --- _AutotunableKernel protocol -----------------------------------------

    @property
    def env(self) -> Any:
        return self.sites[0].bound.env

    @property
    def config_spec(self) -> CompositeConfigSpec:
        if self._config_spec is None:
            root_fragments = (
                self._root_fragments
                if self._root_fragments is not None
                else self.kernel.root_fragments
            )
            self._config_spec = CompositeConfigSpec(root_fragments, self.sites)
        return self._config_spec

    @property
    def configs(self) -> list[helion.Config]:
        return []

    def compile_config(
        self, config: Any = None, *, allow_print: bool = True
    ) -> Callable[..., Any]:
        joint = _joint_dict(config) if config is not None else None
        return self.kernel.configure(joint)

    def bench_compile_config(
        self, config: Any = None, *, allow_print: bool = True
    ) -> Callable[..., Any]:
        return self.compile_config(config, allow_print=allow_print)

    def format_kernel_decorator(self, config: Any, settings: Any) -> str:
        """Placeholder for the search's completion log (single kernels emit the
        matching ``@helion.kernel(config=...)`` decorator here; a composite's
        result is the stored joint config, not a kernel decorator)."""
        return f"# composite joint config: {config!r}"

    def supports_subprocess_benchmark(self) -> bool:
        """Benchmark in process: the composite's runners close over lazily
        bound child kernels, which do not fit the subprocess worker's job
        shape."""
        return False

    def is_cacheable(self) -> bool:
        """Not cacheable via helion's ``Cache[Search]``: the composite's
        config persistence is vLLM's ConfigManager (per CaseKey), and the
        composite cache fingerprint is not shape-sensitive."""
        return False

    def get_cached_path(self, config: Any = None) -> str | None:
        """No single generated source file for a composite (the log line that
        prints the selected kernel's code is skipped)."""
        return None

    def maybe_log_repro(self, log: Any, args: Any, best: Any) -> None:
        """Repro logging for a composite is the stored joint config, handled by
        the caller's config save path."""
        return None

    # --- execution ------------------------------------------------------------

    def set_config(self, config: Any) -> None:
        self._config = _joint_dict(config)

    def __call__(self, *args: Any) -> Any:
        if self._config is None:
            self._config = _joint_dict(self.config_spec.default_config())
        return self.kernel.configure(self._config)(*args)


class CompositeKernel:
    """A composite helion kernel defined by a body calling child kernels.

    Args:
        body: ``def body(run, *args)`` -- the composite operation. ``run`` is
            the framework's child-call context (also exposing ``tunable``);
            the remaining parameters are the composite's outward signature.
        settings: Composite-level helion settings (baseline fn and accuracy
            check for the whole body, search effort).
    """

    def __init__(
        self,
        body: Callable,
        settings: helion.Settings | None = None,
    ):
        if not is_composite_body(body):
            raise ValueError(
                "Composite kernel body must take `run` as its first parameter."
            )
        self.body = body
        self.settings = settings if settings is not None else helion.Settings()
        self.root_fragments: dict[str, ConfigSpecFragment] = {}
        # Search-log surface (helion's _prepare reads both via
        # getattr(self.kernel, "kernel", None)).
        self.name = getattr(body, "__name__", "composite")

    def kernel_source(self) -> str:
        """The composite body's source, for autotune logs (see helion's
        Kernel.kernel_source)."""
        try:
            return inspect.getsource(self.body)
        except OSError:
            return ""

    def _register_root_tunable(self, name: str, fragment: ConfigSpecFragment) -> None:
        existing = self.root_fragments.get(name)
        if existing is None:
            self.root_fragments[name] = fragment
        elif type(existing) is not type(fragment):
            raise ValueError(
                f"Root tunable '{name}' registered with conflicting fragment "
                f"types: {type(existing).__name__} vs {type(fragment).__name__}."
            )

    def bind(self, args: tuple[Any, ...]) -> BoundCompositeKernel:
        """Bind to concrete args: one body pass at default root values,
        recording sites and propagating intermediates at default configs."""
        run = _BindRun(self)
        self.body(run, *args)
        return BoundCompositeKernel(self, args, run.sites, run.root_fragments)

    def configure(self, joint_config: dict[str, Any] | None) -> Callable[..., Any]:
        """Return a runner executing the body with a fixed joint config."""
        joint = joint_config if joint_config is not None else {}
        compiled_cache: dict[tuple[str, int, str], Callable] = {}

        def runner(*args: Any) -> Any:
            run = _ConfiguredRun(self, joint, compiled_cache)
            return self.body(run, *args)

        runner.__signature__ = body_signature(self.body)  # type: ignore[attr-defined]
        return runner

    def autotune(
        self,
        args: tuple[Any, ...],
        *,
        autotune_effort: str = "quick",
    ) -> helion.Config:
        """Search the joint (root + per-site) space with helion's LFBOTreeSearch.

        Every candidate is a complete joint config measured by running the
        whole body, so inter-kernel effects are inside the measurement.
        """
        settings = dataclasses.replace(
            self.settings,
            autotune_effort=autotune_effort,
            autotune_ignore_errors=True,
            # Subprocess precompilation wraps a single kernel's compiled
            # callable (format_kernel_decorator); the composite compiles its
            # children in process via compile_config, so keep it local.
            autotune_precompile=False,
        )
        bound = self.bind(args)
        bound.settings = settings
        kwargs = LFBOTreeSearch.get_kwargs_from_profile(
            get_effort_profile(autotune_effort), settings
        )
        search = LFBOTreeSearch(bound, args, **kwargs)
        return search.autotune()
