"""Freeze a blinded, development-only four-way multimodal rating pool."""
# ruff: noqa: E401, E701, E702, I001
import hashlib, json, random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'backend'/'benchmarks'
RESULT=B/'phase7r_v1_four_way_comparison_20261025T074000Z'/'results.json'; QUERIES=B/'phase7r_v1_queries.json'
OLD=[B/'phase7r_v1_human_eval_preparation_20261025T055021Z'/'comparative.assignment.json',B/'phase7r_v1_intent_human_eval_20261025T065752Z'/'complete_comparative.assignment.json']
OUT=B/'phase7r_v1_multimodal_human_eval_20261025T075000Z'; SEED=70601
def ids(path): return {x.split(':',1)[0] for x in json.loads(path.read_text())['pair_keys']}
def main():
 q={x['id']:x for x in json.loads(QUERIES.read_text())['queries'] if x['split']=='development'}; used=set().union(*(ids(x) for x in OLD)); avail=[x for x in q if x not in used]
 groups={}
 for key in avail: groups.setdefault(q[key]['category'],[]).append(key)
 selected=[min(values,key=lambda x:hashlib.sha256(f'{SEED}:{x}'.encode()).hexdigest()) for _,values in sorted(groups.items())]
 if len(selected)!=8: raise ValueError('expected one unused query in each of eight categories')
 data=json.loads(RESULT.read_text()); rows={x['query_id']:x for x in data['queries']}; pool=[]; provenance=[]
 for query_id in selected:
  merged={}
  for variant in ('A0','A3-LH','M1','M2'):
   for item in rows[query_id]['variants'][variant]['final_top_10'][:5]:
    key=f"{query_id}:{item['source_key']}"; entry=merged.setdefault(key,{'key':key,'query_id':query_id,'meme_id':item['source_key'],'contributors':[]}); entry['contributors'].append({'variant':variant,'rank':item['filtered_rank'],'raw_rank':item['raw_rank']})
  pool.extend(merged.values()); provenance.extend(merged.values())
 rnd=random.Random(SEED); order=[x['key'] for x in pool]; rnd.shuffle(order)
 existing={x for p in OLD for x in json.loads(p.read_text())['pair_keys']}; full=set(order); options={ 'full_four_way':full, 'a3_lh_vs_m2':{x['key'] for x in pool if any(c['variant'] in ('A3-LH','M2') for c in x['contributors'])}, 'a0_a3_lh_m2':{x['key'] for x in pool if any(c['variant'] in ('A0','A3-LH','M2') for c in x['contributors'])}}
 OUT.mkdir(parents=True,exist_ok=False); (OUT/'query_selection.manifest.json').write_text(json.dumps({'seed':SEED,'method':'per-category minimum SHA-256(seed:query_id) among unused development queries before reading candidates','selected_query_ids':selected,'previous_comparative_query_ids':sorted(used),'unused_development_query_ids':sorted(avail)},indent=2)+'\n')
 (OUT/'blinded.pool.json').write_text(json.dumps({'format_version':1,'split':'development','pairs':[{k:v for k,v in x.items() if k!='contributors'} for x in pool],'presentation_seed':SEED,'presentation_order':order},indent=2)+'\n'); (OUT/'provenance.json').write_text(json.dumps({'pairs':provenance},indent=2)+'\n'); (OUT/'ratings.seed.json').write_text(json.dumps({'format_version':1,'ratings':{}},indent=2)+'\n')
 report={'selected_query_ids':selected,'full_four_way_unique_pairs':len(full),'maximum_pairs':160,'existing_exact_judgments':len(full&existing),'new_ratings':len(full-existing),'per_query_pairs':{x:sum(p['query_id']==x for p in pool) for x in selected},'options':{n:{'unique_pairs':len(v),'new_ratings':len(v-existing)} for n,v in options.items()}}
 (OUT/'workload.report.json').write_text(json.dumps(report,indent=2)+'\n'); (OUT/'complete_comparative.assignment.json').write_text(json.dumps({'format_version':1,'assignment':'phase7r.6c.1-full-four-way','pair_keys':order,'presentation_seed':SEED,'description':'Blinded four-way top-five union; scores/provenance withheld.'},indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
