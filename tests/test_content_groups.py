import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from medical_anomaly.content_groups import connected_groups, freeze_selection
from medical_anomaly.explicit_splits import load_explicit_splits, resolve_relative


def test_transitive_components_preserve_weak_candidate_edges():
    groups=connected_groups(['a','b','c','d'],[('a','b'),('b','c')])
    assert groups['a']==groups['b']==groups['c']
    assert groups['d']!=groups['a']
    assert groups==connected_groups(['d','c','b','a'],[('c','b'),('b','a')])


def test_frozen_selection_balances_classes_without_splitting_groups():
    rows=[{'image_path':f'{c}/{i}/{j}','label':c,'group_id':f'{c}-{i}'}
          for c in [0,1] for i in range(300) for j in range(2)]
    a,s,one=freeze_selection(rows)
    assert (a,s,one)==freeze_selection(list(reversed(rows)))
    assert [sum(r['label']==c for r in s['train']) for c in [0,1]]==[336,3]
    assert [sum(r['label']==c for r in s['validation']) for c in [0,1]]==[84,20]
    assert len({r['group_id'] for r in s['train'] if r['label']})==3
    # This synthetic fixture has only 60 normal validation groups for 84 normals.
    assert len({r['group_id'] for r in s['validation']})==80
    assert len({r['image_path'] for r in s['test']})==240
    assert one in {r['image_path'] for r in s['validation'] if r['label']}


def make_manifest(tmp_path):
    assignments=[];splits={}
    for i,split in enumerate(['train','validation','test']):
        # Deliberately use historical Testing images for training and vice versa.
        path=('Testing' if split=='train' else 'TrainingXVal')+f'/NoLabel/{i}.bmp'
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'image')
        row={'image_path':path,'mask_path':'','label':0,'group_id':f'g{i}'}
        assignments.append({**row,'pool':split});splits[split]=[row]
    manifest={'assignments':assignments,'splits':splits}
    p=tmp_path/'manifest.json';p.write_text(json.dumps(manifest));return p,manifest


def test_explicit_lists_ignore_historical_split_folder_semantics(tmp_path):
    p,m=make_manifest(tmp_path)
    result,_,_=load_explicit_splits(p,tmp_path,hashlib.sha256(p.read_bytes()).hexdigest())
    assert result['train'][0]['image_paths'].startswith('Testing/')
    with pytest.raises(ValueError,match='SHA256'):load_explicit_splits(p,tmp_path,'bad')
    m['assignments'][1]['group_id']='g0';p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='group crosses'):load_explicit_splits(p,tmp_path)


def test_reject_traversal_and_repeated_sample(tmp_path):
    p,m=make_manifest(tmp_path)
    with pytest.raises(ValueError):resolve_relative(tmp_path,'../outside')
    with pytest.raises(ValueError):resolve_relative(tmp_path,'D:\\data.bmp')
    m['splits']['test']=m['splits']['train'];p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='identity crosses'):load_explicit_splits(p,tmp_path)


def test_update_lr_schedule_boundaries():
    p=torch.nn.Parameter(torch.ones(()));optimizer=torch.optim.AdamW([p],lr=.0002)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[816,918],gamma=.4)
    rates={}
    for step in range(1,1021):
        p.grad=torch.ones_like(p);optimizer.step();scheduler.step()
        if step in [815,816,817,917,918,919,1020]:rates[step]=optimizer.param_groups[0]['lr']
    assert rates[815]==.0002
    assert rates[816]==rates[817]==rates[917]==.0002*.4
    assert rates[918]==rates[919]==rates[1020]==.0002*.4*.4
    assert optimizer.state[p]['step']==1020
