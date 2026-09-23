from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .tasking import TaskPackage
from .tools import (
    AnnotationReplayScorer,
    AppearanceProxyScorer,
    CheckpointCvsScorer,
    DatasetReplayScorer,
    PeskaVLPCheckpointScorer,
    PeskaVLPTemporalCheckpointScorer,
    QwenVLEvidenceScorer,
    SmallMllmFusionScorer,
    SpatialFusionCheckpointScorer,
)
from .object_observations import default_object_localizers


@dataclass
class EvidenceToolInstance:
    scorer: Any
    oracle: bool = False
    accepts_requirement: bool = False
    sampling_fps: float | None = None
    runtime_kind: str = "generic"
    checkpoint_backed: bool = False
    notes: list[str] = field(default_factory=list)


EvidenceBackendFactory = Callable[
    [dict[str, Any], TaskPackage, dict[str, Any]], EvidenceToolInstance
]


class EvidenceBackendRegistry:
    """Construct evidence tools without adding task branches to the core pipeline."""

    def __init__(self) -> None:
        self._factories: dict[str, EvidenceBackendFactory] = {}

    def register(self, backend_id: str, factory: EvidenceBackendFactory) -> None:
        if backend_id in self._factories:
            raise ValueError(f"Evidence backend already registered: {backend_id}")
        self._factories[backend_id] = factory

    def build(
        self, backend_id: str, config: dict[str, Any],
        task_package: TaskPackage, sources: dict[str, Any],
    ) -> EvidenceToolInstance:
        if backend_id not in self._factories:
            raise KeyError(
                f"Unknown evidence backend {backend_id!r}; available={sorted(self._factories)}"
            )
        return self._factories[backend_id](config, task_package, sources)

    def available(self) -> list[str]:
        return sorted(self._factories)


def _annotation_replay(
    cfg: dict[str, Any], task: TaskPackage, sources: dict[str, Any],
) -> EvidenceToolInstance:
    if task.dataset_adapter is not None:
        scorer = DatasetReplayScorer(
            task.dataset_adapter, cfg["video_id"], **sources,
        )
    else:
        scorer = AnnotationReplayScorer(
            cfg["cvs_annotation_path"], int(cfg["video_id"]),
        )
    return EvidenceToolInstance(scorer=scorer, oracle=True, runtime_kind="annotation_replay")


def _appearance(
    cfg: dict[str, Any], task: TaskPackage, sources: dict[str, Any],
) -> EvidenceToolInstance:
    return EvidenceToolInstance(
        scorer=AppearanceProxyScorer(cfg["video_path"]), runtime_kind="appearance_proxy",
    )


def _frame_checkpoint(
    cfg: dict[str, Any], task: TaskPackage, sources: dict[str, Any],
) -> EvidenceToolInstance:
    if not cfg.get("visual_checkpoint"):
        raise ValueError("visual_checkpoint is required for checkpoint backend")
    return EvidenceToolInstance(
        scorer=CheckpointCvsScorer(cfg["video_path"], cfg["visual_checkpoint"]),
        checkpoint_backed=True, runtime_kind="checkpoint",
    )


def _peskavlp(
    cfg: dict[str, Any], task: TaskPackage, sources: dict[str, Any],
) -> EvidenceToolInstance:
    if not cfg.get("peskavlp_checkpoint"):
        raise ValueError("peskavlp_checkpoint is required for peskavlp_checkpoint backend")
    return EvidenceToolInstance(
        scorer=PeskaVLPCheckpointScorer(
            cfg["video_path"], cfg["peskavlp_checkpoint"],
            inference_batch_size=int(cfg.get("peskavlp_inference_batch_size", 64)),
        ),
        sampling_fps=(
            float(cfg["peskavlp_sampling_fps"])
            if cfg.get("peskavlp_sampling_fps") is not None else None
        ),
        checkpoint_backed=True,
        runtime_kind="ordinal_checkpoint",
        notes=[
            "Visual evidence uses a frozen image encoder and a separately trained lightweight criterion head."
        ],
    )


def _temporal_peskavlp(
    cfg: dict[str, Any], task: TaskPackage, sources: dict[str, Any],
    temporal_enabled: bool,
) -> EvidenceToolInstance:
    if not cfg.get("peskavlp_temporal_checkpoint"):
        raise ValueError("peskavlp_temporal_checkpoint is required for this backend")
    return EvidenceToolInstance(
        scorer=PeskaVLPTemporalCheckpointScorer(
            cfg["video_path"], cfg["peskavlp_temporal_checkpoint"],
            inference_batch_size=int(cfg.get("peskavlp_inference_batch_size", 96)),
            temporal_enabled=temporal_enabled,
        ),
        sampling_fps=(
            float(cfg["peskavlp_sampling_fps"])
            if cfg.get("peskavlp_sampling_fps") is not None else None
        ),
        checkpoint_backed=True,
        runtime_kind=("temporal_checkpoint" if temporal_enabled else "frame_ablation"),
        notes=[
            (
                "Visual evidence uses a frozen encoder, ordinal frame head, and lightweight temporal head."
                if temporal_enabled else
                "Matched ablation bypasses the temporal head while retaining the same encoder and frame head."
            )
        ],
    )


def _qwen_vl(
    cfg: dict[str, Any], task: TaskPackage, sources: dict[str, Any],
) -> EvidenceToolInstance:
    qwen = cfg.get("qwen_vl", {})
    scorer = QwenVLEvidenceScorer(
        video_path=cfg["video_path"],
        base_url=qwen.get("base_url", "http://127.0.0.1:8000/v1"),
        model=qwen.get("model", ""), api_key=qwen.get("api_key", "EMPTY"),
        timeout_s=float(qwen.get("timeout_s", 120)),
        clip_radius_s=float(qwen.get("clip_radius_s", 3)),
        clip_frames=int(qwen.get("clip_frames", 3)),
        max_queries=qwen.get("max_queries"),
        frame_width=int(qwen.get("frame_width", 640)),
        frame_height=int(qwen.get("frame_height", 360)),
    )
    return EvidenceToolInstance(
        scorer=scorer, accepts_requirement=True,
        sampling_fps=(
            float(qwen["sampling_fps"])
            if qwen.get("sampling_fps") is not None else None
        ),
        runtime_kind="mllm",
    )


def _small_mllm_fusion(
    cfg: dict[str, Any], task: TaskPackage, sources: dict[str, Any],
) -> EvidenceToolInstance:
    fusion = cfg.get("small_mllm_fusion", {})
    small = fusion.get("small_model", {})
    mllm = fusion.get("mllm", {})
    trigger = fusion.get("trigger_policy", {})
    rule = fusion.get("fusion", {})
    checkpoint = small.get("checkpoint", cfg.get("peskavlp_checkpoint"))
    if not checkpoint:
        raise ValueError("small_mllm_fusion.small_model.checkpoint is required")
    small_scorer = PeskaVLPCheckpointScorer(
        cfg["video_path"], checkpoint,
        inference_batch_size=int(small.get("inference_batch_size", 64)),
    )
    mllm_scorer = QwenVLEvidenceScorer(
        video_path=cfg["video_path"],
        base_url=mllm.get("base_url", "http://127.0.0.1:8000/v1"),
        model=mllm.get("model", ""), api_key=mllm.get("api_key", "EMPTY"),
        timeout_s=float(mllm.get("timeout_s", 180)),
        clip_radius_s=float(mllm.get("clip_radius_s", 3)),
        clip_frames=int(mllm.get("clip_frames", 3)),
        frame_width=int(mllm.get("frame_width", 640)),
        frame_height=int(mllm.get("frame_height", 360)),
    )
    scorer = SmallMllmFusionScorer(
        small_scorer=small_scorer, mllm_scorer=mllm_scorer,
        top_k_candidates=int(trigger.get("top_k_candidates", 2)),
        uncertainty_queries=int(trigger.get("uncertainty_queries", 1)),
        transition_queries=int(trigger.get("transition_queries", 1)),
        max_queries_per_criterion=int(trigger.get("max_queries_per_criterion", 4)),
        candidate_threshold=float(trigger.get("candidate_threshold", 0.5)),
        uncertainty_low=float(trigger.get("uncertainty_low", 0.35)),
        uncertainty_high=float(trigger.get("uncertainty_high", 0.65)),
        minimum_separation_s=float(trigger.get("minimum_separation_s", 20)),
        maximum_fusion_weight=float(rule.get("maximum_mllm_weight", 0.55)),
        explicit_state_threshold=float(rule.get("explicit_state_threshold", 0.60)),
        unknown_policy=str(rule.get("unknown_policy", "explicit_unknown")),
        update_policy=str(rule.get("update_policy", "symmetric")),
    )
    return EvidenceToolInstance(
        scorer=scorer, accepts_requirement=True,
        sampling_fps=float(small.get("sampling_fps", 0.2)),
        checkpoint_backed=True, runtime_kind="small_mllm_fusion",
        notes=[
            "A specialized visual tool scans densely; an MLLM is triggered only for selected candidate, uncertainty, and transition observations.",
            "Visible MLLM evidence softly updates small-model scores; insufficient visibility never acts as a hard negative.",
        ],
    )


def _build_spatial_scorer(cfg: dict[str, Any]) -> SpatialFusionCheckpointScorer:
    spatial = cfg.get("spatial_fusion", {})
    small = spatial.get("small_model", {})
    checkpoint = small.get("checkpoint", cfg.get("peskavlp_checkpoint"))
    fusion_checkpoint = spatial.get("checkpoint")
    if not checkpoint or not fusion_checkpoint:
        raise ValueError(
            "spatial_fusion.small_model.checkpoint and spatial_fusion.checkpoint are required"
        )
    localizer_cfg = spatial.get("object_localizer", cfg.get("object_localizer", {}))
    provider_id = localizer_cfg.get("provider")
    if not provider_id:
        raise ValueError("spatial_fusion.object_localizer.provider is required")
    provider = default_object_localizers().build(provider_id, localizer_cfg)
    return SpatialFusionCheckpointScorer(
        video_path=cfg["video_path"],
        small_scorer=PeskaVLPCheckpointScorer(
            cfg["video_path"], checkpoint,
            device=spatial.get("device"),
            inference_batch_size=int(small.get("inference_batch_size", 64)),
        ),
        object_provider=provider,
        checkpoint_path=fusion_checkpoint,
        criteria=list(spatial["criteria"]),
        object_classes=list(spatial["object_classes"]),
        candidate_rules=dict(spatial["candidate_rules"]),
        device=spatial.get("device"),
    )


def _spatial_fusion(
    cfg: dict[str, Any], task: TaskPackage, sources: dict[str, Any],
) -> EvidenceToolInstance:
    scorer = _build_spatial_scorer(cfg)
    small = cfg["spatial_fusion"].get("small_model", {})
    return EvidenceToolInstance(
        scorer=scorer,
        sampling_fps=float(small.get("sampling_fps", 0.2)),
        checkpoint_backed=True,
        runtime_kind="spatial_fusion",
        notes=[
            "A learned task adapter fuses dense visual scores with replaceable object-localization observations before temporal aggregation."
        ],
    )


def _spatial_mllm_fusion(
    cfg: dict[str, Any], task: TaskPackage, sources: dict[str, Any],
) -> EvidenceToolInstance:
    small_scorer = _build_spatial_scorer(cfg)
    fusion = cfg.get("spatial_mllm_fusion", {})
    mllm = fusion.get("mllm", {})
    trigger = fusion.get("trigger_policy", {})
    rule = fusion.get("fusion", {})
    mllm_scorer = QwenVLEvidenceScorer(
        video_path=cfg["video_path"],
        base_url=mllm.get("base_url", "http://127.0.0.1:8000/v1"),
        model=mllm.get("model", ""), api_key=mllm.get("api_key", "EMPTY"),
        timeout_s=float(mllm.get("timeout_s", 180)),
        clip_radius_s=float(mllm.get("clip_radius_s", 3)),
        clip_frames=int(mllm.get("clip_frames", 3)),
        frame_width=int(mllm.get("frame_width", 640)),
        frame_height=int(mllm.get("frame_height", 360)),
    )
    scorer = SmallMllmFusionScorer(
        small_scorer=small_scorer, mllm_scorer=mllm_scorer,
        top_k_candidates=int(trigger.get("top_k_candidates", 1)),
        uncertainty_queries=int(trigger.get("uncertainty_queries", 1)),
        transition_queries=int(trigger.get("transition_queries", 1)),
        max_queries_per_criterion=int(trigger.get("max_queries_per_criterion", 3)),
        candidate_threshold=float(trigger.get("candidate_threshold", 0.5)),
        uncertainty_low=float(trigger.get("uncertainty_low", 0.35)),
        uncertainty_high=float(trigger.get("uncertainty_high", 0.65)),
        minimum_separation_s=float(trigger.get("minimum_separation_s", 20)),
        maximum_fusion_weight=float(rule.get("maximum_mllm_weight", 0.55)),
        explicit_state_threshold=float(rule.get("explicit_state_threshold", 0.60)),
        unknown_policy=str(rule.get("unknown_policy", "explicit_unknown")),
        update_policy=str(rule.get("update_policy", "symmetric")),
    )
    small = cfg["spatial_fusion"].get("small_model", {})
    return EvidenceToolInstance(
        scorer=scorer, accepts_requirement=True,
        sampling_fps=float(small.get("sampling_fps", 0.2)),
        checkpoint_backed=True, runtime_kind="spatial_mllm_fusion",
        notes=[
            "Learned small-model and object-location evidence proposes sparse observations for MLLM verification.",
            "The MLLM uses the same query budget and soft-fusion policy as matched sparse-verification baselines.",
        ],
    )


_DEFAULT_BACKENDS: EvidenceBackendRegistry | None = None


def default_evidence_backends() -> EvidenceBackendRegistry:
    global _DEFAULT_BACKENDS
    if _DEFAULT_BACKENDS is None:
        registry = EvidenceBackendRegistry()
        registry.register("annotation_replay", _annotation_replay)
        registry.register("appearance_proxy", _appearance)
        registry.register("checkpoint", _frame_checkpoint)
        registry.register("peskavlp_checkpoint", _peskavlp)
        registry.register(
            "peskavlp_temporal_checkpoint",
            lambda cfg, task, sources: _temporal_peskavlp(cfg, task, sources, True),
        )
        registry.register(
            "peskavlp_temporal_frame_ablation",
            lambda cfg, task, sources: _temporal_peskavlp(cfg, task, sources, False),
        )
        registry.register("qwen_vl", _qwen_vl)
        registry.register("small_mllm_fusion", _small_mllm_fusion)
        registry.register("spatial_fusion", _spatial_fusion)
        registry.register("spatial_mllm_fusion", _spatial_mllm_fusion)
        _DEFAULT_BACKENDS = registry
    return _DEFAULT_BACKENDS
