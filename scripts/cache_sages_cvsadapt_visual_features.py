#!/usr/bin/env python3
"""Cache frozen official CVS-AdaptNet dual-view visual embeddings on SAGES."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from cvs_assessment.models import PeskaVLPVisualEncoder
from train_direct_interval_nested_oof import sha256_file


def load_visual(path):
 model=PeskaVLPVisualEncoder();state=torch.load(path,map_location="cpu",weights_only=False)["model_state_dict"];mapped={}
 for key,value in state.items():
  if key.startswith("model.backbone_img.model."):mapped["backbone."+key.removeprefix("model.backbone_img.model.")]=value
  elif key.startswith("model.backbone_img.global_embedder."):mapped["projection."+key.removeprefix("model.backbone_img.global_embedder.")]=value
 result=model.load_state_dict(mapped,strict=True)
 if result.missing_keys or result.unexpected_keys:raise ValueError("Incomplete visual checkpoint")
 return model.freeze()


def prepare(images,view,device):
 x=images.to(device,non_blocking=True).float()/255
 if view=="center":
  x=F.interpolate(x,size=(360,640),mode="bilinear",align_corners=False);top=(360-224)//2;left=(640-224)//2;x=x[:,:,top:top+224,left:left+224];x=F.interpolate(x,size=(299,299),mode="bilinear",align_corners=True)
 elif view=="full":x=F.interpolate(x,size=(299,299),mode="bilinear",align_corners=True)
 else:raise ValueError(view)
 mean=torch.tensor((.485,.456,.406),device=device)[None,:,None,None];std=torch.tensor((.229,.224,.225),device=device)[None,:,None,None];return (x-mean)/std


@torch.inference_mode()
def encode(model,images,view,device,batch=18):
 out=[]
 for start in range(0,len(images),batch):out.append(model(prepare(images[start:start+batch],view,device)).float().cpu())
 return torch.cat(out)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--protocol",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--device",default="cuda:2");p.add_argument("--shard-index",type=int,required=True);p.add_argument("--shard-count",type=int,required=True);a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 protocol=json.loads(a.protocol.read_text());checkpoint=Path(protocol["model"]["checkpoint"]["path"])
 if protocol["extraction_code_sha256"]!=sha256_file(Path(__file__)) or sha256_file(checkpoint)!=protocol["model"]["checkpoint"]["sha256"]:raise ValueError("Code/checkpoint changed")
 audit_path=Path(protocol["sources"]["visual_feature_audit"]["path"]);audit=json.loads(audit_path.read_text());rows=audit["videos"][a.shard_index::a.shard_count];device=torch.device(a.device);model=load_visual(checkpoint).to(device);a.output_dir.mkdir(parents=True,exist_ok=False);done=[]
 for position,row in enumerate(rows,1):
  rgb_path=Path(row["RGB_frame_cache_path"])
  if sha256_file(rgb_path)!=row["RGB_frame_cache_sha256"]:raise ValueError("RGB changed")
  rgb=torch.load(rgb_path,map_location="cpu",weights_only=False)
  if rgb.get("SAGES_train_labels_loaded") is not False or rgb.get("SAGES_test_labels_accessed") is not False:raise ValueError("Unsafe RGB")
  images=rgb["RGB_uint8_384x640"];center=encode(model,images,"center",device);full=encode(model,images,"full",device);features=torch.cat([center,full],dim=1);video_id=row["video_id"];path=a.output_dir/f"{video_id}.pt";torch.save({"schema_version":"sages_cvs_2024_official_cvsadapt_dual_view_visual_features_v1","video_id":video_id,"frame_ids":rgb["frame_ids"],"center_CVSAdapt_visual":center.half(),"full_CVSAdapt_visual":full.half(),"dual_view_CVSAdapt_visual":features.half(),"feature_dim":1536,"SAGES_train_labels_loaded":False,"SAGES_test_labels_accessed":False,"LLM_or_MLLM_parameters_updated":False},path);done.append({"video_id":video_id,"frame_count":len(images),"path":str(path.resolve()),"sha256":sha256_file(path)});print(json.dumps({"shard":a.shard_index,"video":f"{position}/{len(rows)}","video_id":video_id}),flush=True)
 out={"schema_version":"sages_cvs_2024_cvsadapt_visual_feature_shard_audit_v1","created_at":datetime.now(timezone.utc).isoformat(),"protocol":{"path":str(a.protocol.resolve()),"sha256":sha256_file(a.protocol)},"complete":len(done)==len(rows),"shard_index":a.shard_index,"shard_count":a.shard_count,"video_count":len(done),"frame_count":sum(row["frame_count"] for row in done),"videos":done,"SAGES_train_labels_loaded":False,"SAGES_test_labels_accessed":False,"LLM_or_MLLM_parameters_updated":False};path=a.output_dir/"CACHE_AUDIT.json";path.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"audit":str(path.resolve()),"complete":True,"videos":len(done)},indent=2))
if __name__=="__main__":main()
