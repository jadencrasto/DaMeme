"""Development-only OpenCLIP M1 raw retrieval, using recorded baseline query text."""
# ruff: noqa: E401, E701, E702, B905, I001
import argparse, json, time
from pathlib import Path
import faiss, numpy as np, open_clip, torch

MODEL="ViT-B-32-quickgelu"; PRETRAINED="openai"
def main():
 p=argparse.ArgumentParser(); p.add_argument('--baseline',type=Path,required=True); p.add_argument('--index',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 base=json.loads((a.baseline/'results.json').read_text()); mapping=json.loads((a.index/'source_keys.json').read_text()); ix=faiss.read_index(str(a.index/'openclip.faiss'))
 device='cuda' if torch.cuda.is_available() else 'cpu'; model,_,_=open_clip.create_model_and_transforms(MODEL,pretrained=PRETRAINED,device=device); model.eval(); tok=open_clip.get_tokenizer(MODEL)
 rows=[]
 for q in base['queries']:
  if q['split']!='development': raise ValueError('held-out query reached comparison')
  start=time.perf_counter()
  with torch.inference_mode(): v=model.encode_text(tok([q['effective_query']]).to(device)); v=v/v.norm(dim=-1,keepdim=True).clamp_min(1e-12)
  torch.cuda.synchronize() if device=='cuda' else None; scores,pos=ix.search(v.cpu().numpy().astype(np.float32),100)
  rows.append({'query_id':q['query_id'],'effective_query':q['effective_query'],'latency_ms':(time.perf_counter()-start)*1000,'candidates':[{'source_key':mapping[int(i)]['source_key'],'raw_rank':r,'raw_score':float(s)} for r,(s,i) in enumerate(zip(scores[0],pos[0]),1)]})
 a.output.write_text(json.dumps({'model':MODEL,'pretrained':PRETRAINED,'device':device,'queries':rows},indent=2)+'\n')
if __name__=='__main__': main()
