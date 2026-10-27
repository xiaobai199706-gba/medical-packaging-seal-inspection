"""Frozen content-group checks, resume smoke, final-update training, and held-out evaluation."""
from __future__ import annotations

import argparse, csv, gc, hashlib, importlib.metadata, json, os, subprocess, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tools/evidence_audit_20261023'))
from medical_anomaly.baseline import load_upstream, seed_all, best_threshold
from medical_anomaly.checkpointing import git_state, sha256_file, load_checkpoint, restore_rng_state
from medical_anomaly.supervision import configure_supervision
from medical_anomaly.training import train_with_epoch_checkpoints
from medical_anomaly.evaluation import histogram_dice_threshold, _score_bins
from medical_anomaly.explicit_splits import load_explicit_splits
from export_baseline_predictions import collect
from frozen_evaluator import stable_probability, evaluate_bundle

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def dataset_module(args, config):
    from medical_anomaly.data.ds1_upstream import DS1UpstreamDataModule
    dm=DS1UpstreamDataModule(args.data_root,None,tuple(config['image_size']),config['batch_size'],config['num_workers'],
                             config['val_fraction'],config['seed'],config['supervision_mode'],
                             split_manifest=args.split_manifest,manifest_sha256=config['split_manifest_sha256'])
    dm.setup()
    for split,dataset in [('train',dm.train),('validation',dm.val),('test',dm.test)]:
        actual=[Path(p).relative_to(args.data_root).as_posix() for frame in dataset.samples for p in frame.image_path]
        expected=[r['image_path'] for r in dm.explicit_manifest['splits'][split]]
        if actual!=expected:raise ValueError('Actual dataset differs from frozen list: '+split)
    if len(dm.train_dataloader())*config['epochs'] != config['optimizer_updates']:
        raise ValueError('Actual dataloader does not implement requested update budget')
    return dm

def configure(args,config):
    seed_all(config['seed'])
    Model,focal,_=load_upstream(args.upstream_root)
    dm=dataset_module(args,config)
    model=Model(tuple(config['image_size']),config).to('cuda')
    configure_supervision(model,config['supervision_mode'])
    optimizer,_=model.get_optimizers()
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,config['lr_milestones_updates'],gamma=config['gamma'])
    return dm,model,optimizer,scheduler,focal

def state_hash(state):
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def train_segment(args,config,out,stop=None,resume=False):
    dm,model,optimizer,scheduler,focal=configure(args,config)
    initial_hash=state_hash(model.state_dict())
    path=out/'checkpoints/latest.ckpt'
    identity=git_state(ROOT)
    kwargs={}
    if resume:
        saved=load_checkpoint(path)
        if saved['config']!=config or saved['git']!=identity:raise ValueError('Resume code/config mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        dm.load_checkpoint_state(saved['dataloader_state']);restore_rng_state(saved['rng_state'])
        kwargs=dict(start_epoch=saved['completed_epoch'],best_metric=saved['best_metric'],best_model_state=saved['best_model_state'],
                    epoch_losses=saved['epoch_losses'],supervision_exposure=saved['supervision_exposure'],progress_state=saved['training_progress'])
    elif out.exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    _,checkpoint,exposure=train_with_epoch_checkpoints(model,config['epochs'],dm,'cuda',{}, {},focal,
         lambda **kw: (_ for _ in ()).throw(RuntimeError('Training must not access evaluation metrics')),
         path,config,identity,optimizer=optimizer,scheduler=scheduler,clip_grad=config['clip_grad'],
         stop_after_epoch=stop,**kwargs)
    seconds=time.perf_counter()-started
    end=load_checkpoint(path)
    steps=end['training_progress']['optimizer_steps']
    adam_steps=sorted({int(v['step']) for v in optimizer.state.values() if 'step' in v})
    if adam_steps!=[steps]:raise ValueError(('Actual Adam step count mismatch',adam_steps,steps))
    progress=end['training_progress']
    summary={'config':config,'git':identity,'manifest_sha256':sha256_file(args.split_manifest),
             'initial_state_sha256':initial_hash,'final_state_sha256':state_hash(model.state_dict()),
             'optimizer_steps':steps,'adam_state_steps':adam_steps,'completed_epoch':end['completed_epoch'],
             'effective_epoch_samples':len(dm.train),'batches_per_epoch':len(dm.train_dataloader()),
             'seconds_this_segment':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'exposure':exposure,
             'lr_events':progress['lr_events'],'unique_normals_exposed':len([p for p in progress['sample_exposure_counts'] if '/NoLabel/' in p.replace('\\','/')]),
             'checkpoint':checkpoint,'command':[sys.executable,*sys.argv],'resumed':resume}
    if stop is None:
        if steps!=1020:raise ValueError('Incomplete formal run')
        if [e['after_optimizer_step'] for e in progress['lr_events']] != [816,918]:raise ValueError('Wrong LR boundaries')
        model.save_model(out/'checkpoints')
        summary['weights_sha256']=sha256_file(out/'checkpoints/weights.pt')
        summary['weights_selection']='final update 1020, no validation/test checkpoint selection'
    write_json(out/(f'segment_{end["completed_epoch"]}_resume{resume}.json' if stop else 'training_summary.json'),summary)
    del model,optimizer,scheduler,dm,end
    gc.collect();torch.cuda.empty_cache()
    return summary

def preflight(args):
    configs=[json.loads(p.read_text()) for p in sorted(args.config_directory.glob('*.json'))]
    assert len(configs)==6
    observed=[]
    for config in configs:
        load_upstream(args.upstream_root)
        dm=dataset_module(args,config)
        observed.append({'run':config['name'],'seed':config['seed'],'mode':config['supervision_mode'],
                         'manifest_sha256':dm.manifest_sha256,'effective_epoch_samples':len(dm.train),
                         'loader_batches':len(dm.train_dataloader()),'expected_optimizer_steps':len(dm.train_dataloader())*config['epochs'],
                         'actual_rows_equal_frozen_rows':True})
    summary=json.loads((args.split_manifest.parent/'group_audit_summary.json').read_text())
    assert summary['cross_pool_edges']==0 and summary['file_pixel_mask_checks']==1200
    write_json(args.output/'preflight_pass.json',{'passed':True,'checks':observed,'group_audit':summary,'test_metrics_accessed':False})

def smoke(args,config):
    result=[]
    for mode in ['mixed','real_only']:
        c={**config,'seed':42,'supervision_mode':mode}
        continuous=args.output/(mode+'_continuous')
        resumed=args.output/(mode+'_resumed')
        a=train_segment(args,c,continuous,stop=2)
        train_segment(args,c,resumed,stop=1)
        b=train_segment(args,c,resumed,stop=2,resume=True)
        x=load_checkpoint(continuous/'checkpoints/latest.ckpt');y=load_checkpoint(resumed/'checkpoints/latest.ckpt')
        exact=all(torch.equal(x['model'][k],y['model'][k]) for k in x['model'])
        same_progress=x['training_progress']==y['training_progress']
        same_sampling=all(torch.equal(torch.as_tensor(x['dataloader_state']['dataset'][k]),torch.as_tensor(y['dataloader_state']['dataset'][k])) for k in x['dataloader_state']['dataset'])
        assert exact and same_progress and same_sampling,(mode,exact,same_progress,same_sampling)
        result.append({'mode':mode,'optimizer_steps':a['optimizer_steps'],'continuous_resume_model_bitwise_equal':exact,
                       'sampling_and_exposure_equal':same_progress and same_sampling,'initial_states_equal':a['initial_state_sha256']==b['initial_state_sha256']})
    write_json(args.output/'resume_smoke_pass.json',{'passed':True,'runs':result,'test_metrics_accessed':False,
               'note':'independent scratch checks; formal training restarts from same pretrained initialization'})

def save_split(path,values):
    if path.exists():raise FileExistsError(path)
    with path.open('xb') as f:np.savez_compressed(f,**values)

def calibrate(values,manifest):
    labels=values['labels'];prob=stable_probability(values['score_logits'])
    hist={budget:[np.zeros(65536,np.int64),np.zeros(65536,np.int64)] for budget in [20,1]}
    one=manifest['calibration_one_defect_image']
    included={budget:[] for budget in hist}
    for i,(label,logit,size,mask,sid) in enumerate(zip(labels,values['map_logits'],values['source_sizes'],values['mask_paths'],values['sample_ids'])):
        shape=tuple(map(int,size));heat=cv2.resize(stable_probability(logit),(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
        gt=cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE)!=0 if label else np.zeros(shape,bool)
        idx=_score_bins(heat,65536)
        pos=np.bincount(idx[gt],minlength=65536);neg=np.bincount(idx[~gt],minlength=65536)
        for budget in hist:
            if budget==20 or not label or str(sid).replace('\\','/').split('#')[0]==one:
                hist[budget][0]+=pos;hist[budget][1]+=neg;included[budget].append(i)
    thresholds=[]
    for budget in hist:
        ix=included[budget];assert len(ix)==84+budget
        pt,dice=histogram_dice_threshold(*hist[budget]);it=best_threshold(labels[ix],prob[ix],'youden')
        thresholds.append({'calibration_defects':budget,'calibration_normals':84,'image_threshold':it,'pixel_threshold':pt,
                           'validation_pooled_histogram_dice':dice,'sample_ids':[str(values['sample_ids'][i]) for i in ix],
                           'selection':'validation max Youden J / original-scale pooled histogram Dice; highest threshold on ties'})
    return thresholds

def evaluate(args,config):
    run=args.output
    trained=json.loads((run/'training_summary.json').read_text());assert trained['optimizer_steps']==1020
    assert trained['config']==config and trained['manifest_sha256']==sha256_file(args.split_manifest)
    assert sha256_file(run/'checkpoints/weights.pt')==trained['weights_sha256']
    dm,model,opt,sched,focal=configure(args,config)
    model.load_model(run/'checkpoints/weights.pt')
    validation=collect(model,dm.val_dataloader(),'cuda',args.data_root)
    save_split(run/'validation_float32_logits.npz',validation)
    thresholds=calibrate(validation,dm.explicit_manifest)
    write_json(run/'frozen_validation_thresholds.json',{'weights_sha256':trained['weights_sha256'],'thresholds':thresholds,
               'test_predictions_or_metrics_accessed_before_freeze':False})
    test=collect(model,dm.test_dataloader(),'cuda',args.data_root)
    arrays={f'{split}_{k}':v for split,values in [('validation',validation),('test',test)] for k,v in values.items()}
    save_split(run/'predictions_float32_logits.npz',arrays)
    del model,opt,sched,dm;gc.collect();torch.cuda.empty_cache()
    rows=[]
    for threshold in thresholds:
        metrics,per=evaluate_bundle(run/'predictions_float32_logits.npz',threshold['image_threshold'],threshold['pixel_threshold'])
        metrics.update({'run':config['name'],'mode':config['supervision_mode'],'seed':config['seed'],'calibration_defects':threshold['calibration_defects'],
                        'image_fp_denominator':metrics['normal'],'image_fn_denominator':metrics['defect'],
                        'training_defect_annotations':3,'calibration_defect_annotations':threshold['calibration_defects']})
        rows.append(metrics)
        with (run/f'per_image_cal{threshold["calibration_defects"]}.csv').open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,list(per[0]));w.writeheader();w.writerows(per)
    write_json(run/'metrics.json',{'results':rows,'weights_sha256':trained['weights_sha256'],
               'prediction_sha256':sha256_file(run/'predictions_float32_logits.npz'),
               'validation_prediction_sha256':sha256_file(run/'validation_float32_logits.npz'),
               'manifest_sha256':sha256_file(args.split_manifest),'evaluation_commit':git_state(ROOT)['commit'],
               'semantics':'raw float32 map/head logits; stable sigmoid float64->float32, linear original-size restore, 65536-bin pooled AP/Dice including normal false-positive pixels; direct continuous-threshold spatial metrics separate'})
    print(json.dumps(rows,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['preflight','smoke','train','evaluate'],required=True)
    ap.add_argument('--config',type=Path);ap.add_argument('--config-directory',type=Path)
    ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--upstream-root',type=Path,required=True)
    ap.add_argument('--split-manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gate-directory',type=Path);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();args.data_root=args.data_root.resolve()
    config=json.loads(args.config.read_text()) if args.config else None
    if args.phase=='preflight':preflight(args)
    elif args.phase=='smoke':smoke(args,config)
    elif args.phase=='train':
        for gate in ['preflight_pass.json','smoke/resume_smoke_pass.json']:
            assert json.loads((args.gate_directory/gate).read_text())['passed']
        train_segment(args,config,args.output,resume=args.resume)
    else:evaluate(args,config)

if __name__=='__main__':main()
