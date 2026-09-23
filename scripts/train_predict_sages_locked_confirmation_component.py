#!/usr/bin/env python3
"""Train one frozen deployment recipe on 600 videos and predict 100 without reading their labels."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from scripts.cache_sages_cvsadapt_visual_features import load_visual
from train_direct_interval_nested_oof import sha256_file
import run_sages_nested_oof_outer_fold as old
import run_sages_dinov2_nested_oof_outer_fold as dino_engine
import run_sages_grouped_fusion_nested_oof_outer_fold as grouped
import run_sages_cvsadapt_layer4_nested_oof_outer_fold as layer4_engine

CRITERIA = old.CRITERIA


def label_values(label_root: Path, video_id: str, frame_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    labels = {}
    # This function is called for development IDs only. Locked IDs never reach it.
    with (label_root / video_id / "frame.csv").open(newline="") as handle:
        for item in csv.DictReader(handle):
            labels[int(item["frame_id"])] = (
                [old.majority(item, key) for key in ("c1", "c3", "c2")],
                [sum(int(item[f"{key}_rater{i}"]) for i in (1, 2, 3)) / 3 for key in ("c1", "c3", "c2")],
            )
    if set(frame_ids) != set(labels):
        raise ValueError(f"Development label alignment: {video_id}")
    return (
        torch.tensor([labels[index][0] for index in frame_ids], dtype=torch.float32),
        torch.tensor([labels[index][1] for index in frame_ids], dtype=torch.float32),
    )


def audit_rows(protocol: dict, source_key: str) -> dict:
    audit = json.loads(Path(protocol["sources"][source_key]["path"]).read_text())
    if not audit.get("complete") or audit.get("SAGES_test_labels_accessed") is not False:
        raise ValueError(f"Unsafe cache: {source_key}")
    return {row["video_id"]: row for row in audit["videos"]}


def load_static(protocol: dict, kind: str, development: list[str], confirmation: list[str]):
    visual_rows = audit_rows(protocol, "visual_feature_cache_audit" if "visual_feature_cache_audit" in protocol["sources"] else "feature_cache_audit")
    dino_rows = audit_rows(protocol, "dinov2_feature_cache_audit") if kind in {"dino", "grouped", "cvsadapt"} else None
    cvsa_rows = audit_rows(protocol, "cvsadapt_feature_cache_audit") if kind == "cvsadapt" else None
    label_root = Path(protocol["sources"]["train_label_download_audit"]["path"]).parent / "train" / "labels"
    train_x = []; eval_x = []; train_y = []; train_soft = []; eval_baseline = []
    development_set = set(development)
    for video_id in development + confirmation:
        visual = torch.load(visual_rows[video_id]["path"], map_location="cpu", weights_only=False)
        if visual.get("SAGES_test_labels_accessed") is not False:
            raise ValueError("Unsafe visual cache")
        frame_ids = list(map(int, visual["frame_ids"].tolist()))
        existing = visual["detector_and_dual_moco_features"].float()
        if kind in {"dino", "grouped", "cvsadapt"}:
            dino = torch.load(dino_rows[video_id]["path"], map_location="cpu", weights_only=False)
            if frame_ids != list(map(int, dino["frame_ids"].tolist())):
                raise ValueError("DINO alignment")
            values = torch.cat([existing, dino["dual_view_DINOv2S_features"].float()], dim=-1)
        else:
            values = existing
        if kind == "cvsadapt":
            cvsa = torch.load(cvsa_rows[video_id]["path"], map_location="cpu", weights_only=False)
            if frame_ids != list(map(int, cvsa["frame_ids"].tolist())):
                raise ValueError("CVSAdapt alignment")
            values = torch.cat([existing, cvsa["dual_view_CVSAdapt_visual"].float(), dino["dual_view_DINOv2S_features"].float()], dim=-1)
        if video_id in development_set:
            y, soft = label_values(label_root, video_id, frame_ids)
            train_x.append(values); train_y.append(y); train_soft.append(soft)
        else:
            eval_x.append(values); eval_baseline.append(visual["baseline_probability"].float())
    return torch.stack(train_x), torch.stack(train_y), torch.stack(train_soft), torch.stack(eval_x), torch.stack(eval_baseline)


def load_layer4(protocol: dict, development: list[str], confirmation: list[str]):
    layer3_rows = audit_rows(protocol, "layer3_feature_cache_audit")
    visual_rows = audit_rows(protocol, "visual_feature_cache_audit")
    dino_rows = audit_rows(protocol, "dinov2_feature_cache_audit")
    label_root = Path(protocol["sources"]["train_label_download_audit"]["path"]).parent / "train" / "labels"
    train_maps=[]; train_context=[]; train_y=[]; train_soft=[]; eval_maps=[]; eval_context=[]; eval_baseline=[]
    development_set=set(development)
    for video_id in development + confirmation:
        maps=torch.load(layer3_rows[video_id]["path"],map_location="cpu",weights_only=False)
        visual=torch.load(visual_rows[video_id]["path"],map_location="cpu",weights_only=False)
        dino=torch.load(dino_rows[video_id]["path"],map_location="cpu",weights_only=False)
        if any(value.get("SAGES_test_labels_accessed") is not False for value in (maps,visual,dino)):
            raise ValueError("Unsafe layer4 cache")
        frame_ids=list(map(int,visual["frame_ids"].tolist()))
        if frame_ids != list(map(int,maps["frame_ids"].tolist())) or frame_ids != list(map(int,dino["frame_ids"].tolist())):
            raise ValueError("Layer4 alignment")
        context=torch.cat([visual["detector_and_dual_moco_features"].half(),dino["dual_view_DINOv2S_features"].half()],dim=-1)
        if video_id in development_set:
            y,soft=label_values(label_root,video_id,frame_ids)
            train_maps.append(maps["dual_view_layer3_float16"]);train_context.append(context);train_y.append(y);train_soft.append(soft)
        else:
            eval_maps.append(maps["dual_view_layer3_float16"]);eval_context.append(context);eval_baseline.append(visual["baseline_probability"].float())
    return tuple(torch.stack(value) for value in (train_maps,train_context,train_y,train_soft,eval_maps,eval_context,eval_baseline))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--protocol",type=Path,required=True);parser.add_argument("--component",required=True);parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--device",required=True);args=parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    frozen=json.loads(args.protocol.read_text())
    if frozen["prediction_code_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("Prediction code differs from frozen protocol")
    recipe=frozen["components"][args.component];source_protocol_path=Path(recipe["protocol"]["path"]);deployment_path=Path(recipe["deployment"]["path"])
    if sha256_file(source_protocol_path)!=recipe["protocol"]["sha256"] or sha256_file(deployment_path)!=recipe["deployment"]["sha256"]: raise ValueError("Frozen source changed")
    source=json.loads(source_protocol_path.read_text());deployment=json.loads(deployment_path.read_text());candidate=deployment["candidate"];fixed_epoch=int(deployment["fixed_epoch"]);config=source["training"];device=torch.device(args.device)
    development=sorted(frozen["development_video_ids"]);confirmation=sorted(frozen["locked_confirmation_video_ids"])
    if set(development)&set(confirmation) or len(development)!=600 or len(confirmation)!=100: raise ValueError("Split mismatch")
    skill_payload=torch.load(source["sources"]["frozen_skill_embeddings"]["path"],map_location="cpu",weights_only=False);skills=torch.stack([skill_payload["criterion_embeddings"][key] for key in CRITERIA]).to(device)
    seed=int(recipe["full_development_seed"]);kind=recipe["kind"]
    if kind == "layer4":
        train_maps,train_context,y,soft,eval_maps,eval_context,baseline=load_layer4(source,development,confirmation)
        official=load_visual(Path(source["sources"]["official_cvsadapt_checkpoint"]["path"]));layer4_engine.LAYER4_TEMPLATE=copy.deepcopy(official.backbone.layer4);layer4_engine.PROJECTION_TEMPLATE=copy.deepcopy(official.projection);del official
        prediction=layer4_engine.fit(candidate,train_maps.reshape(-1,2,1024,19,19),train_context.reshape(-1,9234),y.reshape(-1,3),soft.reshape(-1,3),eval_maps.reshape(-1,2,1024,19,19),eval_context.reshape(-1,9234),skills,config,device,seed,[fixed_epoch])[fixed_epoch].reshape(100,18,3)
    else:
        x,y,soft,eval_x,baseline=load_static(source,kind,development,confirmation)
        if kind == "old":
            prediction=old.fit(candidate,x,y,soft,eval_x,skills,config,device,seed,[fixed_epoch])[fixed_epoch]
        else:
            if kind == "grouped": old.build=grouped.build
            prediction=dino_engine.fit(candidate,x,y,soft,eval_x,skills,config,device,seed,[fixed_epoch])[fixed_epoch]
    framework=np.empty_like(prediction)
    for index,criterion in enumerate(CRITERIA):
        weight=float(deployment["fusion_weights"][criterion]);framework[:,:,index]=weight*prediction[:,:,index]+(1-weight)*baseline.numpy()[:,:,index]
    args.output_dir.mkdir(parents=True,exist_ok=False);prediction_path=args.output_dir/"locked_confirmation_predictions.pt"
    torch.save({"schema_version":"sages_cvs_2024_locked_confirmation_component_predictions_v1","component":args.component,"video_ids":confirmation,"baseline_probability":baseline,"selected_skill_probability":torch.from_numpy(prediction),"framework_probability":torch.from_numpy(framework),"criterion_order":CRITERIA,"SAGES_confirmation_labels_read":False,"SAGES_test_labels_accessed":False,"LLM_or_MLLM_parameters_updated":False},prediction_path)
    audit={"schema_version":"sages_cvs_2024_locked_confirmation_component_prediction_audit_v1","created_at":datetime.now(timezone.utc).isoformat(),"protocol":{"path":str(args.protocol.resolve()),"sha256":sha256_file(args.protocol)},"component":args.component,"kind":kind,"development_training_video_count":600,"locked_prediction_video_count":100,"fixed_epoch":fixed_epoch,"seed":seed,"prediction":{"path":str(prediction_path.resolve()),"sha256":sha256_file(prediction_path)},"SAGES_confirmation_labels_read":False,"SAGES_test_labels_accessed":False,"LLM_or_MLLM_parameters_updated":False}
    audit_path=args.output_dir/"PREDICTION_AUDIT.json";audit_path.write_text(json.dumps(audit,indent=2)+"\n");print(json.dumps(audit,indent=2))


if __name__=="__main__": main()
