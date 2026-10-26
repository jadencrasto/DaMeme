"""Read-only Phase 7R.6C.2 analysis of completed blinded ratings."""
# ruff: noqa: E401, E701, E702, E703, E721, I001
import json, math
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; D=ROOT/'backend'/'benchmarks'/'phase7r_v1_multimodal_human_eval_20261025T075000Z'; OUT=D
V=('A0','A3-LH','M1','M2'); AX=('situation','tone','sendability')
def dcg(x): return sum((2**v-1)/math.log2(i+2) for i,v in enumerate(x))
def main():
 a=json.loads((D/'ui.assignment.json').read_text()); p=json.loads((D/'provenance.json').read_text())['pairs']; r=json.loads((D/'ratings.multimodal-evaluator-1.json').read_text())['evaluators']['multimodal-evaluator-1']; expected=set(a['pair_keys'])
 if set(r)!=expected or len(expected)!=92 or any(set(x)!=set(AX) or any(type(v)!=int or not 0<=v<=3 for v in x.values()) for x in r.values()): raise ValueError('invalid or incomplete ratings')
 byq={};
 for x in p:
  if x['key'] in expected: byq.setdefault(x['query_id'],[]).append(x)
 metrics={}; per={}
 for v in V:
  agg={axis:[] for axis in AX}; irr=0; rows=[]
  for q,items in byq.items():
   ranked=sorted((x for x in items if any(c['variant']==v for c in x['contributors'])),key=lambda x:next(c['rank'] for c in x['contributors'] if c['variant']==v))[:5]
   row={'query_id':q,'ranking':[x['key'] for x in ranked],'metrics':{}}
   for axis in AX:
    vals=[r[x['key']][axis] for x in ranked]; ideal=sorted((r[x['key']][axis] for x in items),reverse=True)[:5]; val={'ndcg_at_5':dcg(vals)/dcg(ideal) if dcg(ideal) else 0,'precision_at_5':sum(x>=2 for x in vals)/5,'mean_top_5_rating':sum(vals)/5}; row['metrics'][axis]=val; agg[axis].append(val)
   irr+=sum(r[x['key']]['situation']<=1 for x in ranked); rows.append(row)
  metrics[v]={axis:{k:sum(x[k] for x in agg[axis])/8 for k in agg[axis][0]} for axis in AX}; metrics[v]['irrelevant_top5']=irr; per[v]=rows
 out={'validation':{'expected_pairs':92,'rated_pairs':len(r),'complete':True,'selected_queries':sorted(byq)},'metrics':metrics,'per_query':per,'repetition':Counter(x['meme_id'] for x in p if x['key'] in expected).most_common(10)}
 OUT.mkdir(exist_ok=True); (OUT/'final_multimodal_evaluation_metrics.json').write_text(json.dumps(out,indent=2)+'\n'); (OUT/'per_query_comparison.json').write_text(json.dumps(per,indent=2)+'\n'); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
