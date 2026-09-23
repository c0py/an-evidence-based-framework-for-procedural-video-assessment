#!/usr/bin/env python3
"""Train the selected shared frozen-Skill head on all official train frames."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import random,sys

import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cvs_assessment.detector_skill_cvs import SkillConditionedDetectorHead


def sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--protocol",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--device",default="cuda:0");args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    protocol=json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["training_code_sha256"]!=sha256_file(Path(__file__)) or protocol["model_code_sha256"]!=sha256_file(ROOT/"cvs_assessment"/"detector_skill_cvs.py"):raise ValueError("Code differs from frozen protocol")
    audit_source=protocol["sources"]["detector_feature_cache_audit"];audit_path=Path(audit_source["path"])
    if sha256_file(audit_path)!=audit_source["sha256"]:raise ValueError("Cache audit changed")
    audit=json.loads(audit_path.read_text(encoding="utf-8"));expected=set(map(int,protocol["official_train_video_ids"]));features=[];labels=[]
    for row in audit["videos"]:
        path=Path(row["path"])
        if sha256_file(path)!=row["sha256"]:raise ValueError("Cached video changed")
        value=torch.load(path,map_location="cpu",weights_only=False);features.append(torch.cat([value["detector_features"].float(),value["detector_geometry"].float()],1));labels.append(value["labels_C1_C3_C2"].float())
    if {int(row["video_id"]) for row in audit["videos"]}!=expected:raise ValueError("Full train coverage mismatch")
    x=torch.cat(features);y=torch.cat(labels);skill_source=protocol["sources"]["frozen_skill_embeddings"];skill_path=Path(skill_source["path"])
    if sha256_file(skill_path)!=skill_source["sha256"]:raise ValueError("Skill embeddings changed")
    skill=torch.load(skill_path,map_location="cpu",weights_only=False);order=["two_structures","cystic_plate","hepatocystic_triangle"];device=torch.device(args.device);skills=torch.stack([skill["criterion_embeddings"][key] for key in order]).to(device);cfg=protocol["training"];seed=int(cfg["seed"]);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model_cfg=protocol["model"];model=SkillConditionedDetectorHead(int(model_cfg["input_dim"]),int(model_cfg["text_dim"]),int(model_cfg["hidden_dim"]),float(model_cfg["dropout"])).to(device);positives=y.sum(0);pos_weight=((len(y)-positives)/positives.clamp_min(1)).clamp(1,float(cfg["positive_weight_cap"])).to(device);loader=DataLoader(TensorDataset(x,y),batch_size=int(cfg["batch_size"]),shuffle=True,generator=torch.Generator().manual_seed(seed));optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=int(cfg["epochs"]),eta_min=float(cfg["minimum_learning_rate"]));history=[]
    for epoch in range(1,int(cfg["epochs"])+1):
        model.train();total=count=0
        for bx,by in loader:
            optimizer.zero_grad(set_to_none=True);logits=model(bx.to(device),skills);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,by.to(device),pos_weight=pos_weight);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["gradient_clip"]));optimizer.step();total+=float(loss.detach())*len(bx);count+=len(bx)
        scheduler.step();history.append({"epoch":epoch,"loss":total/count});print(json.dumps(history[-1]),flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=False);checkpoint=args.output_dir/"full_train_skill_head.pt";torch.save({"schema_version":"endoscapes_skill_head_full_train_v1","model_state":model.cpu().state_dict(),"model":model_cfg,"criterion_order":order,"fixed_epoch":int(cfg["epochs"]),"official_train_video_ids":sorted(expected),"detector_parameters_updated":False,"skill_text_embeddings_updated":False,"official_val_or_test_labels_used":False,"LLM_or_MLLM_parameters_updated":False},checkpoint);audit_out={"schema_version":"endoscapes_skill_head_full_train_audit_v1","created_at":datetime.now(timezone.utc).isoformat(),"protocol":{"path":str(args.protocol.resolve()),"sha256":sha256_file(args.protocol)},"history":history,"checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha256_file(checkpoint)},"official_val_or_test_used":False,"LLM_or_MLLM_parameters_updated":False};(args.output_dir/"TRAINING_AUDIT.json").write_text(json.dumps(audit_out,indent=2)+"\n",encoding="utf-8");print(json.dumps(audit_out,indent=2))


if __name__=="__main__":main()
