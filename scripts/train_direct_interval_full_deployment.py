#!/usr/bin/env python3
"""Train the frozen direct-interval deployment model on all formal train videos."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_direct_interval_nested_oof import DirectIntervalDataset, Trainer, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("external_cvs_labels_accessed_during_freeze") is not False:
        raise ValueError("External labels were not sealed during protocol freeze")
    if protocol.get("LLM_or_MLLM_parameters_updated") is not False:
        raise ValueError("Protocol violates the no-LLM-update requirement")
    consensus_entry = protocol["source_files"]["consensus_policy"]
    consensus_path = Path(consensus_entry["path"])
    if sha256_file(consensus_path) != consensus_entry["sha256"]:
        raise ValueError("Frozen consensus policy changed")
    dataset = DirectIntervalDataset(consensus_path)
    if set(dataset.labeled_ids) & set(protocol["dataset"]["test_video_ids"]):
        # IDs belong to different datasets, but an intersection here is still
        # useful as an explicit warning against accidental Endoscapes loading.
        # Cholec80 formal train is 2..50, so the frozen test 162..201 is disjoint.
        raise ValueError("Unexpected numeric overlap between deployment train and external test")
    deployment = protocol["deployment_training"]
    if deployment["external_images_or_labels_used"] is not False:
        raise ValueError("Deployment protocol unexpectedly permits external data")
    seed = int(deployment["seed"])
    max_epochs = int(deployment["max_epochs"])
    checkpoint_epochs = set(map(int, deployment["checkpoint_epochs"]))
    if max(checkpoint_epochs) > max_epochs:
        raise ValueError("Requested checkpoint exceeds frozen training duration")

    trainer = Trainer(dataset, args.device)
    model = trainer.new_model(seed)
    train_cfg = dataset.frozen["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    all_indices = dataset.indices(set(dataset.labeled_ids))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    history = []
    checkpoints = []
    for epoch in range(1, max_epochs + 1):
        losses = trainer.train_epoch(model, optimizer, all_indices, epoch, seed)
        history.append({"epoch": epoch, **losses})
        print(json.dumps({"epoch": epoch, "losses": losses}), flush=True)
        if epoch in checkpoint_epochs:
            path = args.output_dir / f"epoch{epoch:02d}.pt"
            torch.save({
                "schema_version": "text_conditioned_direct_interval_full_train_deployment_v1",
                "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "epoch": epoch,
                "seed": seed,
                "formal_Cholec80_train_video_ids": dataset.labeled_ids,
                "criterion_decoders": deployment["criterion_decoders"],
                "source_protocol": str(args.protocol.resolve()),
                "source_protocol_sha256": sha256_file(args.protocol),
                "source_consensus_policy_sha256": consensus_entry["sha256"],
                "external_images_used": False,
                "external_CVS_labels_used": False,
                "development_or_consumed_Cholec80_test_accessed": False,
                "LLM_or_MLLM_parameters_updated": False,
            }, path)
            checkpoints.append({"epoch": epoch, "path": str(path.resolve()), "sha256": sha256_file(path)})
            print(json.dumps({"saved_checkpoint": str(path), "sha256": checkpoints[-1]["sha256"]}), flush=True)
    audit = {
        "schema_version": "direct_interval_full_train_deployment_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": str(args.protocol.resolve()), "sha256": sha256_file(args.protocol)},
        "formal_Cholec80_train_video_ids": dataset.labeled_ids,
        "training_sequence_count": len(all_indices),
        "training_video_count": len(dataset.labeled_ids),
        "epochs": max_epochs,
        "history": history,
        "checkpoints": checkpoints,
        "external_images_used": False,
        "external_CVS_labels_used": False,
        "development_or_consumed_Cholec80_test_accessed": False,
        "LLM_or_MLLM_parameters_updated": False,
    }
    audit_path = args.output_dir / "deployment_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "audit": str(audit_path.resolve()),
        "checkpoints": checkpoints,
        "external_CVS_labels_used": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
