# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import torch

from vllm_ascend import dsa_sparse_probe
from vllm_ascend.attention.dsa_sparse import (
    DSASparseBatchMetadata,
    DSASparseCacheConfig,
    DSASparseCohort,
    DSASparseCohortKey,
    DSASparseEagerBatchContext,
    DSASparseEagerContextRouter,
    DSASparseEagerCoordinator,
    DSASparseLayerBinding,
    DSASparseLayerHotCache,
    DSASparseLayerLayout,
    DSASparseLookupUpdateOperator,
    DSASparsePlan,
    DSASparsePlanKey,
    DSASparseResidencyState,
)
from vllm_ascend.attention.dsa_sparse_io import (
    DSASparseIOOperator,
    MockDSASparseIOOperator,
)
from vllm_ascend.attention.dsa_sparse_pd import (
    DSASparsePDLifecycle,
    DSASparseTransferCompletion,
)


class DSASparseEagerLayerMetadata(Protocol):
    """Metadata fields needed by the eager runner adapter."""

    num_input_tokens: int
    num_actual_tokens: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    dsa_sparse_context: DSASparseEagerContextRouter | None


@dataclass(frozen=True)
class DSASparseEagerCohortDescriptor:
    """Runner-visible routing information for one IndexCache cohort."""

    cohort_key: DSASparseCohortKey
    plan_key: DSASparsePlanKey
    layer_names: tuple[str, ...]
    leader_layer: str

    def __post_init__(self) -> None:
        if not isinstance(self.layer_names, tuple):
            object.__setattr__(self, "layer_names", tuple(self.layer_names))
        if not self.layer_names:
            raise ValueError("A DSA Sparse eager cohort must contain at least one layer.")
        if len(set(self.layer_names)) != len(self.layer_names):
            raise ValueError("A DSA Sparse eager cohort cannot contain duplicate layer names.")
        if any(not layer_name for layer_name in self.layer_names):
            raise ValueError("DSA Sparse eager layer names must not be empty.")
        if self.leader_layer not in self.layer_names:
            raise ValueError("The DSA Sparse eager cohort leader must be one of its layers.")
        if self.cohort_key.role != "target" or self.plan_key.role != "target":
            raise ValueError("begin_target_batch only accepts target DSA Sparse cohorts.")


@dataclass(frozen=True)
class DSASparseEagerCohortLayout:
    """Ordered local Main layouts sharing one target residency cohort."""

    cohort_name: str
    layer_layouts: tuple[DSASparseLayerLayout, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.layer_layouts, tuple):
            object.__setattr__(
                self,
                "layer_layouts",
                tuple(self.layer_layouts),
            )
        if not self.cohort_name:
            raise ValueError("A DSA Sparse eager cohort name must not be empty.")
        if not self.layer_layouts:
            raise ValueError("A DSA Sparse eager cohort must contain at least one Main layout.")
        layer_names = tuple(layout.layer_name for layout in self.layer_layouts)
        if len(set(layer_names)) != len(layer_names):
            raise ValueError("A DSA Sparse eager cohort cannot contain duplicate Main layers.")

    @property
    def leader_layer(self) -> str:
        return self.layer_layouts[0].layer_name

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(layout.layer_name for layout in self.layer_layouts)


@dataclass(frozen=True, eq=False)
class _MockDSASparseIOResource:
    """Identity-only resource owned by the eager no-op I/O fixture."""

    layer_name: str
    purpose: str


class _MockDSASparseRequestRegionBackend:
    """Control-plane region fixture paired with the no-op data plane."""

    def release_request(self, request_handle: int) -> None:
        del request_handle


def create_dsa_sparse_eager_mock_runtime(
    config: DSASparseCacheConfig,
    cohort_layouts: Iterable[DSASparseEagerCohortLayout],
    *,
    device: torch.device | str,
    index_operator: DSASparseLookupUpdateOperator | None = None,
    io_operator: DSASparseIOOperator | None = None,
) -> DSASparseEagerRuntime:
    """Allocate the target-only eager runtime with a no-op I/O fixture.

    The lookup/update path uses the formal Ascend 950 custom operator unless a
    test explicitly injects another implementation.  The I/O fixture preserves
    the final one-call-per-layer ABI but moves no payload; it is not a storage
    backend and cannot establish model accuracy when a history miss occurs.
    """

    cohort_layouts = tuple(cohort_layouts)
    if not cohort_layouts:
        raise ValueError("DSA Sparse eager runtime requires at least one cohort layout.")

    cohort_names = tuple(layout.cohort_name for layout in cohort_layouts)
    if len(set(cohort_names)) != len(cohort_names):
        raise ValueError("DSA Sparse eager cohort names must be unique.")
    all_layer_names = [layer_name for cohort_layout in cohort_layouts for layer_name in cohort_layout.layer_names]
    if len(set(all_layer_names)) != len(all_layer_names):
        raise ValueError("Each DSA Sparse Main layer must belong to exactly one eager cohort.")

    if index_operator is None:
        from vllm_ascend.ops.dsa_sparse import (
            DSASparseLookupUpdateTorchOperator,
        )

        index_operator = DSASparseLookupUpdateTorchOperator()
    if io_operator is None:
        io_operator = MockDSASparseIOOperator()

    coordinator = DSASparseEagerCoordinator(
        config,
        index_operator=index_operator,
        io_operator=io_operator,
    )
    plan_key = DSASparsePlanKey(
        token_capacity=(config.max_num_seqs * config.max_query_tokens_per_request),
        request_capacity=config.max_num_seqs,
        query_lane_capacity=config.max_query_tokens_per_request,
        role="target",
    )
    batch_metadata = DSASparseBatchMetadata.allocate(
        config,
        plan_key,
        device=device,
    )
    descriptors: list[DSASparseEagerCohortDescriptor] = []
    for cohort_layout in cohort_layouts:
        cohort_key = DSASparseCohortKey(
            name=cohort_layout.cohort_name,
            role="target",
        )
        coordinator.register_cohort(
            DSASparseCohort(
                key=cohort_key,
                leader_layer=cohort_layout.leader_layer,
                state=DSASparseResidencyState.allocate(
                    config,
                    cohort_key,
                    device=device,
                ),
                plans={
                    plan_key: DSASparsePlan.allocate(
                        config,
                        plan_key,
                        device=device,
                        batch_metadata=batch_metadata,
                    )
                },
            )
        )
        for layer_layout in cohort_layout.layer_layouts:
            layer_name = layer_layout.layer_name
            coordinator.register_layer(
                DSASparseLayerBinding(
                    layer_name=layer_name,
                    cohort=cohort_key,
                    hot_cache=DSASparseLayerHotCache.allocate(
                        layer_layout,
                        config,
                        device=device,
                    ),
                    io_context=_MockDSASparseIOResource(
                        layer_name,
                        "context",
                    ),
                    io_region=_MockDSASparseIOResource(
                        layer_name,
                        "region",
                    ),
                    io_completion=_MockDSASparseIOResource(
                        layer_name,
                        "io_completion",
                    ),
                )
            )
        descriptors.append(
            DSASparseEagerCohortDescriptor(
                cohort_key=cohort_key,
                plan_key=plan_key,
                layer_names=cohort_layout.layer_names,
                leader_layer=cohort_layout.leader_layer,
            )
        )

    coordinator.freeze()
    if dsa_sparse_probe.is_enabled():
        dsa_sparse_probe.emit(
            "runtime_ready",
            cohort_count=len(descriptors),
            layer_count=len(all_layer_names),
            index_topk=config.index_topk,
            cohorts=[
                {
                    "name": descriptor.cohort_key.name,
                    "layers": list(descriptor.layer_names),
                }
                for descriptor in descriptors
            ],
        )
    mock_region_backend = _MockDSASparseRequestRegionBackend()
    return DSASparseEagerRuntime(
        coordinator,
        descriptors,
        mock_pd_lifecycle=DSASparsePDLifecycle(
            coordinator=coordinator,
            backend=mock_region_backend,
        ),
    )


class DSASparseEagerExecution:
    """One attached target-batch execution with deterministic cleanup."""

    def __init__(
        self,
        router: DSASparseEagerContextRouter,
        attached_metadata: Sequence[DSASparseEagerLayerMetadata],
    ) -> None:
        self.router = router
        self._attached_metadata = tuple(attached_metadata)
        self._entered = False
        self._closed = False

    def __enter__(self) -> DSASparseEagerContextRouter:
        if self._closed:
            raise RuntimeError("DSA Sparse eager execution is already closed.")
        if self._entered:
            raise RuntimeError("DSA Sparse eager execution is already active.")
        self._entered = True
        return self.router

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        del exc_type, traceback
        if self._closed:
            raise RuntimeError("DSA Sparse eager execution is already closed.")

        if exc_value is not None:
            try:
                self.router.abort()
            except BaseException as cleanup_error:
                _add_cleanup_note(
                    exc_value,
                    f"DSA Sparse eager abort also failed: {cleanup_error!r}",
                )
            finally:
                self._detach_preserving(exc_value)
                self._closed = True
            return False

        try:
            self.router.finish()
        except BaseException as finish_error:
            self._detach_preserving(finish_error)
            self._closed = True
            raise

        try:
            self._detach()
        finally:
            self._closed = True
        return False

    def _detach_preserving(self, primary_error: BaseException) -> None:
        try:
            self._detach()
        except BaseException as cleanup_error:
            _add_cleanup_note(
                primary_error,
                f"DSA Sparse eager metadata detach also failed: {cleanup_error!r}",
            )

    def _detach(self) -> None:
        first_error: BaseException | None = None
        for metadata in reversed(self._attached_metadata):
            try:
                current_context = metadata.dsa_sparse_context
                if current_context is self.router:
                    metadata.dsa_sparse_context = None
                elif current_context is not None:
                    raise RuntimeError("DSA Sparse metadata context ownership changed before detach.")
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise RuntimeError("Failed to detach DSA Sparse eager metadata.") from first_error


class DSASparseEagerRuntime:
    """Create and attach eager target contexts without runner-global state."""

    def __init__(
        self,
        coordinator: DSASparseEagerCoordinator,
        cohort_descriptors: Iterable[DSASparseEagerCohortDescriptor],
        *,
        mock_pd_lifecycle: DSASparsePDLifecycle | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.cohort_descriptors = tuple(cohort_descriptors)
        if not self.cohort_descriptors:
            raise ValueError("DSA Sparse eager runtime requires at least one cohort.")
        self._mock_pd_lifecycle = mock_pd_lifecycle
        self._mock_request_generations: dict[Hashable, int] = {}
        self._next_mock_region_handle = 1
        self._validate_descriptors()

    def acquire_request(self, request_id: Hashable) -> int:
        """Admit a dual-ready request and return its stable request index."""

        return self.coordinator.acquire_request(request_id)

    def release_request(self, request_id: Hashable) -> int:
        """Release an idle request index after its backend region is retired."""

        return self.coordinator.release_request(request_id)

    def has_mock_request(self, request_id: Hashable) -> bool:
        return request_id in self._mock_request_generations

    def preflight_mock_retire(
        self,
        request_ids: Iterable[Hashable],
    ) -> None:
        """Validate every active mock request before mutating runner state."""

        self._require_mock_pd_lifecycle()
        for request_id in request_ids:
            if self.has_mock_request(request_id):
                self.coordinator.assert_request_idle(request_id)

    @property
    def uses_mock_lifecycle(self) -> bool:
        return self._mock_pd_lifecycle is not None

    def admit_mock_request(
        self,
        request_id: Hashable,
    ) -> int:
        """Simulate dual-ready completion for the eager no-op I/O fixture."""

        lifecycle = self._require_mock_pd_lifecycle()
        if request_id in self._mock_request_generations:
            raise RuntimeError(
                f"DSA Sparse mock request {request_id!r} is already active."
            )
        handle = self._next_mock_region_handle
        self._next_mock_region_handle += 1
        generation = lifecycle.begin_handoff(
            request_id,
            f"mock-transfer-{handle}",
        )
        try:
            completion = DSASparseTransferCompletion(
                request_id=request_id,
                generation=generation,
            )
            lifecycle.mark_main_region_ready(
                completion,
                request_handle=handle,
            )
            lifecycle.mark_indexer_ready(completion)
            notifications = lifecycle.take_ready_notifications()
            if lifecycle.ready_request_ids(notifications) != {
                request_id
            }:
                raise RuntimeError(
                    "DSA Sparse mock dual-ready completion was not "
                    "published."
                )
            request_index = lifecycle.admit(request_id, generation)
        except BaseException as admission_error:
            try:
                lifecycle.abort_handoff(request_id, generation)
            except BaseException as cleanup_error:
                _add_cleanup_note(
                    admission_error,
                    "DSA Sparse mock admission rollback also failed: "
                    f"{cleanup_error!r}",
                )
            raise
        self._mock_request_generations[request_id] = generation
        return request_index

    def retire_mock_request(
        self,
        request_id: Hashable,
        *,
        preempted: bool,
    ) -> None:
        """Release the mock region and request index at a request boundary."""

        lifecycle = self._require_mock_pd_lifecycle()
        try:
            generation = self._mock_request_generations[request_id]
        except KeyError as error:
            raise KeyError(
                f"DSA Sparse mock request {request_id!r} is not active."
            ) from error
        if preempted:
            lifecycle.preempt(request_id, generation)
        else:
            lifecycle.finish(request_id, generation)
        del self._mock_request_generations[request_id]

    def begin_target_batch(
        self,
        *,
        request_ids: Sequence[Hashable],
        query_positions: torch.Tensor,
        query_counts: Sequence[int],
        layer_metadata: Mapping[str, object],
    ) -> DSASparseEagerExecution:
        """Begin every cohort and attach one shared router to layer metadata."""

        request_ids = list(request_ids)
        query_counts = list(query_counts)
        request_indices = [
            self.coordinator.request_index(request_id)
            for request_id in request_ids
        ]
        metadata_by_layer = self._resolve_layer_metadata(layer_metadata)
        unique_metadata = _unique_by_identity(metadata_by_layer.values())
        self._reject_existing_contexts(unique_metadata)

        contexts: list[DSASparseEagerBatchContext] = []
        layer_contexts: dict[str, DSASparseEagerBatchContext] = {}
        try:
            for cohort_index, descriptor in enumerate(
                self.cohort_descriptors
            ):
                leader_metadata = metadata_by_layer[descriptor.leader_layer]
                context = self._begin_cohort(
                    descriptor,
                    leader_metadata=leader_metadata,
                    request_ids=request_ids,
                    request_indices=request_indices,
                    query_positions=query_positions,
                    query_counts=query_counts,
                    stage_batch_metadata=cohort_index == 0,
                )
                contexts.append(context)
                layer_contexts.update(dict.fromkeys(descriptor.layer_names, context))
            router = DSASparseEagerContextRouter(layer_contexts)
        except BaseException as begin_error:
            _abort_contexts_preserving(contexts, begin_error)
            raise

        attempted_metadata: list[DSASparseEagerLayerMetadata] = []
        try:
            for metadata in unique_metadata:
                attempted_metadata.append(metadata)
                current_context = metadata.dsa_sparse_context
                if current_context is not None:
                    raise RuntimeError("DSA Sparse metadata already owns an execution context.")
                metadata.dsa_sparse_context = router
                if metadata.dsa_sparse_context is not router:
                    raise RuntimeError("DSA Sparse metadata rejected its execution context.")
        except BaseException as attach_error:
            _detach_attempted_preserving(
                attempted_metadata,
                router,
                attach_error,
            )
            try:
                router.abort()
            except BaseException as cleanup_error:
                _add_cleanup_note(
                    attach_error,
                    f"DSA Sparse eager attach rollback also failed: {cleanup_error!r}",
                )
            raise

        return DSASparseEagerExecution(
            router,
            unique_metadata,
        )

    def _validate_descriptors(self) -> None:
        cohort_keys: set[DSASparseCohortKey] = set()
        layer_names: set[str] = set()
        shared_batch_metadata: object | None = None
        for descriptor in self.cohort_descriptors:
            if descriptor.cohort_key in cohort_keys:
                raise ValueError("Each DSA Sparse eager cohort may be described only once.")
            duplicate_layers = layer_names.intersection(descriptor.layer_names)
            if duplicate_layers:
                duplicate = sorted(duplicate_layers)[0]
                raise ValueError(f"DSA Sparse layer {duplicate!r} belongs to multiple cohorts.")

            cohort = self.coordinator.get_cohort(descriptor.cohort_key)
            if cohort.leader_layer != descriptor.leader_layer:
                raise ValueError(
                    "DSA Sparse runtime leader does not match the registered "
                    f"cohort leader for {descriptor.cohort_key!r}."
                )
            if descriptor.plan_key not in cohort.plans:
                raise ValueError(
                    f"DSA Sparse plan {descriptor.plan_key!r} is not registered for cohort {descriptor.cohort_key!r}."
                )
            plan = cohort.plans[descriptor.plan_key]
            if shared_batch_metadata is None:
                shared_batch_metadata = plan.batch_metadata
            elif plan.batch_metadata is not shared_batch_metadata:
                raise ValueError(
                    "All target DSA Sparse cohorts must share one role-level "
                    "batch metadata allocation."
                )
            for layer_name in descriptor.layer_names:
                self.coordinator.get_layer_binding(
                    descriptor.cohort_key,
                    layer_name,
                )

            cohort_keys.add(descriptor.cohort_key)
            layer_names.update(descriptor.layer_names)

    def _require_mock_pd_lifecycle(self) -> DSASparsePDLifecycle:
        if self._mock_pd_lifecycle is None:
            raise RuntimeError(
                "This DSA Sparse eager runtime has no mock P/D lifecycle."
            )
        return self._mock_pd_lifecycle

    def _resolve_layer_metadata(
        self,
        layer_metadata: Mapping[str, object],
    ) -> dict[str, DSASparseEagerLayerMetadata]:
        resolved: dict[str, DSASparseEagerLayerMetadata] = {}
        for descriptor in self.cohort_descriptors:
            for layer_name in descriptor.layer_names:
                try:
                    metadata = layer_metadata[layer_name]
                except KeyError as error:
                    raise KeyError(f"Missing DSA Sparse metadata for layer {layer_name!r}.") from error
                if not hasattr(metadata, "dsa_sparse_context"):
                    raise TypeError(f"Metadata for DSA Sparse layer {layer_name!r} does not expose dsa_sparse_context.")
                resolved[layer_name] = cast(
                    DSASparseEagerLayerMetadata,
                    metadata,
                )
        return resolved

    @staticmethod
    def _reject_existing_contexts(
        metadata_objects: Sequence[DSASparseEagerLayerMetadata],
    ) -> None:
        for metadata in metadata_objects:
            if metadata.dsa_sparse_context is not None:
                raise RuntimeError("DSA Sparse metadata already owns an execution context.")

    def _begin_cohort(
        self,
        descriptor: DSASparseEagerCohortDescriptor,
        *,
        leader_metadata: DSASparseEagerLayerMetadata,
        request_ids: list[Hashable],
        request_indices: list[int],
        query_positions: torch.Tensor,
        query_counts: list[int],
        stage_batch_metadata: bool,
    ) -> DSASparseEagerBatchContext:
        num_input_tokens = _metadata_integer(
            leader_metadata,
            "num_input_tokens",
        )
        num_actual_tokens = _metadata_integer(
            leader_metadata,
            "num_actual_tokens",
        )
        num_active_queries = sum(query_counts)
        if num_actual_tokens != num_active_queries:
            raise ValueError(
                "DSA Sparse leader num_actual_tokens must equal the active "
                f"query count, got {num_actual_tokens} and "
                f"{num_active_queries}."
            )
        if num_input_tokens < num_actual_tokens:
            raise ValueError("DSA Sparse leader num_input_tokens must cover all actual tokens.")

        seq_lens = _metadata_tensor(leader_metadata, "seq_lens")
        block_table = _metadata_tensor(leader_metadata, "block_table")
        num_requests = len(request_ids)
        return DSASparseEagerBatchContext.begin(
            self.coordinator,
            descriptor.cohort_key,
            descriptor.plan_key,
            request_ids=request_ids,
            request_indices=request_indices,
            query_positions=query_positions,
            query_counts=query_counts,
            seq_lens=seq_lens[:num_requests],
            block_table=block_table[:num_requests],
            num_sfa_queries=num_input_tokens,
            stage_batch_metadata=stage_batch_metadata,
        )


def _metadata_integer(metadata: object, field_name: str) -> int:
    try:
        value = getattr(metadata, field_name)
    except AttributeError as error:
        raise TypeError(f"DSA Sparse leader metadata is missing {field_name!r}.") from error
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"DSA Sparse leader metadata {field_name!r} must be an integer.")
    if value < 0:
        raise ValueError(f"DSA Sparse leader metadata {field_name!r} must not be negative.")
    return value


def _metadata_tensor(
    metadata: object,
    field_name: str,
) -> torch.Tensor:
    try:
        value = getattr(metadata, field_name)
    except AttributeError as error:
        raise TypeError(f"DSA Sparse leader metadata is missing {field_name!r}.") from error
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"DSA Sparse leader metadata {field_name!r} must be a tensor.")
    return value


def _unique_by_identity(
    values: Iterable[DSASparseEagerLayerMetadata],
) -> tuple[DSASparseEagerLayerMetadata, ...]:
    unique: list[DSASparseEagerLayerMetadata] = []
    identities: set[int] = set()
    for value in values:
        identity = id(value)
        if identity not in identities:
            identities.add(identity)
            unique.append(value)
    return tuple(unique)


def _abort_contexts_preserving(
    contexts: Sequence[DSASparseEagerBatchContext],
    primary_error: BaseException,
) -> None:
    for context in reversed(contexts):
        try:
            context.abort()
        except BaseException as cleanup_error:
            _add_cleanup_note(
                primary_error,
                f"DSA Sparse partial begin rollback also failed: {cleanup_error!r}",
            )


def _detach_attempted_preserving(
    metadata_objects: Sequence[DSASparseEagerLayerMetadata],
    router: DSASparseEagerContextRouter,
    primary_error: BaseException,
) -> None:
    for metadata in reversed(metadata_objects):
        try:
            if metadata.dsa_sparse_context is router:
                metadata.dsa_sparse_context = None
        except BaseException as cleanup_error:
            _add_cleanup_note(
                primary_error,
                f"DSA Sparse partial attach detach also failed: {cleanup_error!r}",
            )


def _add_cleanup_note(
    primary_error: BaseException,
    note: str,
) -> None:
    add_note = getattr(primary_error, "add_note", None)
    if add_note is not None:
        add_note(note)
