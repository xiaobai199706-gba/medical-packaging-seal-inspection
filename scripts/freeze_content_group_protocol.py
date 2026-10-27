"""Verify prior artifacts/current pixels, then freeze conservative content groups. No models."""
from pathlib import Path
import argparse, csv, hashlib, itertools, json, sys
from collections import Counter, defaultdict
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from medical_anomaly.content_groups import PROTOCOL, SPLIT_SEED, connected_groups, freeze_selection

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writejson(p,value):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def writecsv(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    old=args.audit/'reports'
    inventory={r['artifact']:r for r in csvread(ROOT/'docs/evidence_audit_20261023/local_artifact_manifest.csv')}
    names=['image_identity.csv','near_duplicate_candidates.csv','near_duplicate_native_review.csv','proposed_content_groups_NOT_APPLIED.csv']
    verified=[]
    for n in names:
        p=old/n;expected=inventory['reports/'+n]['sha256'];actual=sha(p)
        assert actual==expected,(n,actual,expected)
        verified.append({'artifact':n,'sha256':actual,'bytes':p.stat().st_size,'matched_published_manifest':True})
    identities=csvread(old/names[0]);candidates=csvread(old/names[1]);native=csvread(old/names[2])
    assert len(identities)==1200 and len(candidates)==430 and len(native)==430
    assert all(r['physical_package_id']=='unknown' and r['batch_id']=='unknown' for r in identities)
    rows=[]
    for i,r in enumerate(identities):
        p=args.data_root/r['relative'];raw=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
        assert raw is not None
        h=hashlib.sha256(f'{raw.shape}|{raw.dtype}|'.encode());h.update(np.ascontiguousarray(raw).tobytes())
        assert sha(p)==r['file_sha256'] and h.hexdigest()==r['decoded_sha256'],p
        label=int(r['label']);mask=args.data_root/'Notation'/(p.stem+'.png')
        if label:
            gt=cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE)
            assert gt is not None and gt.shape==raw.shape[:2] and np.any(gt!=0),mask
            assert sha(mask)==r['mask_sha256']
        else:
            assert not mask.exists(),mask
        assert label==int(p.parent.name!='NoLabel')
        rows.append({'image_path':r['relative'],'mask_path':mask.relative_to(args.data_root).as_posix() if label else '',
                     'label':label,'file_sha256':r['file_sha256'],'decoded_sha256':r['decoded_sha256'],
                     'mask_sha256':r['mask_sha256'],'normal_mask_semantics':'constructed all-zero from normal label' if not label else 'source annotation'})
        if (i+1)%100==0:print('current file/pixel/mask verification',i+1,1200,flush=True)
    assert {p.relative_to(args.data_root).as_posix() for part in ['TrainingXVal','Testing'] for p in (args.data_root/part).glob('*/*.bmp')}=={r['image_path'] for r in rows}
    edges=[]
    for field in ['file_sha256','decoded_sha256']:
        buckets=defaultdict(list)
        for r in rows:buckets[r[field]].append(r['image_path'])
        for members in buckets.values():
            for a,b in itertools.combinations(members,2):edges.append({'a':a,'b':b,'kind':field})
    for r in candidates:
        assert int(r['phash_hamming'])<=8 and float(r['rmse256'])<=.03 and float(r['ssim256'])>=.98
        a,b=rows[int(r['index_a'])]['image_path'],rows[int(r['index_b'])]['image_path']
        assert Path(r['path_a']).name==Path(a).name and Path(r['path_b']).name==Path(b).name
        edges.append({'a':a,'b':b,'kind':'initial_candidate'})
    groups=connected_groups([r['image_path'] for r in rows],[(e['a'],e['b']) for e in edges])
    for r in rows:r['group_id']=groups[r['image_path']]
    assignment,selected,one=freeze_selection(rows)
    assert freeze_selection(rows)==(assignment,selected,one),'Non-reproducible split'
    assert sum(r['label'] for r in selected['train'])==3
    assert len({r['group_id'] for r in selected['train'] if r['label']})==3
    assert sum(r['label'] for r in selected['validation'])==20
    for e in edges:assert assignment[groups[e['a']]]==assignment[groups[e['b']]]
    for r in native:
        a,b=rows[int(r['index_a'])]['image_path'],rows[int(r['index_b'])]['image_path']
        assert assignment[groups[a]]==assignment[groups[b]]
    distribution=defaultdict(list)
    for r in rows:distribution[r['group_id']].append(r)
    sizes=Counter(len(v) for v in distribution.values())
    stats=[]
    for split in ['train','validation','test']:
        for stage,rr in [('candidate_pool',[r for r in rows if assignment[r['group_id']]==split]),('selected',selected[split])]:
            stats.append({'split':split,'stage':stage,'normal':sum(r['label']==0 for r in rr),'defect':sum(r['label']==1 for r in rr),'images':len(rr),'content_groups':len({r['group_id'] for r in rr})})
    rules={'protocol':PROTOCOL,'split_seed':SPLIT_SEED,'group_basis':'content similarity, NOT confirmed physical package identity',
           'edges':['file SHA256 equality','decoded pixel SHA256 equality','all 430 initial candidates: pHash63<=8 AND gray256 RMSE<=0.03 AND SSIM>=0.98'],
           'transitive_components':True,'native_confirmed_subset':sum(r['confirmed_extreme_image_correlation']=='True' for r in native),
           'physical_identity_edges':0,'physical_metadata':'no reliable IDs found in previously audited supplied metadata; filename proxies NOT merged',
           'allocation':'seeded largest-component-first greedy minimum incremental class-wise squared target error divided by target; class targets 60/20/20',
           'selection':'seeded content-group round-robin, one image per group before second; sort final lists by label then relative path',
           'calibration_one_defect_selection':'same round-robin with seed 20262023 over selected validation defects; all 84 selected normal validation images retained',
           'unused':'remain in original assigned pool; never moved to satisfy quotas','nature':'retrospective supplementary validation, not prospective production validation',
           'limitations':'only detected content associations isolated; physical entities/batches unknown; no performance-based split/selection'}
    manifest={'protocol':PROTOCOL,'rules':rules,'assignments':[{**r,'pool':assignment[r['group_id']]} for r in rows],
              'splits':selected,'calibration_one_defect_image':one}
    writejson(args.output/'frozen_manifest.json',manifest)
    writejson(args.output/'grouping_rules.json',rules)
    writecsv(args.output/'artifact_verification.csv',verified)
    writecsv(args.output/'all_edges.csv',edges)
    writecsv(args.output/'all_assignments.csv',manifest['assignments'])
    writecsv(args.output/'split_counts.csv',stats)
    largest=sorted(distribution,key=lambda g:(-len(distribution[g]),g))[:20]
    summary={'images':len(rows),'groups':len(distribution),'size_histogram':dict(sorted(sizes.items())),
             'max_group_size':max(sizes),'mixed_label_groups':[g for g,v in distribution.items() if len({r['label'] for r in v})>1],
             'largest_groups':[{'group_id':g,'size':len(distribution[g]),'normal':sum(r['label']==0 for r in distribution[g]),'defect':sum(r['label']==1 for r in distribution[g])} for g in largest],
             'edge_records':len(edges),'candidate_edges':430,'native_subset_edges':406,'cross_pool_edges':0,
             'duplicate_or_group_intersections':0,'file_pixel_mask_checks':1200,'deterministic_regeneration_pass':True,
             'manifest_sha256':sha(args.output/'frozen_manifest.json'),'counts':stats,'calibration_one_defect_image':one,
             'gate':'PASS grouping/file checks; actual datamodule check and short resume test required before formal training'}
    writejson(args.output/'group_audit_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
