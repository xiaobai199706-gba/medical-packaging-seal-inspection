"""Streaming evaluation v20261023: one restored image at a time, fixed thresholds."""
from common_audit import *
import math,importlib.util
import numpy as np,cv2
from medical_anomaly.evaluation import PixelHistogram,histogram_binary_metrics,histogram_aupro,_score_bins,image_metrics
_spec=importlib.util.spec_from_file_location('existing_spatial',ROOT/'scripts/summarize_supervision_decomposition.py')
_mod=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_mod)
def stable_probability(x):
    x=np.asarray(x,dtype=np.float64);p=np.empty_like(x);q=x>=0;p[q]=1/(1+np.exp(-x[q]));e=np.exp(x[~q]);p[~q]=e/(1+e)
    return p.astype(np.float32)
def evaluate_bundle(bundle_path,image_threshold,pixel_threshold,split='test',progress=True):
    with np.load(bundle_path,allow_pickle=False) as z:
        labels=z[f'{split}_labels'];maps=z[f'{split}_map_logits'];heads=z[f'{split}_score_logits'];masks=z[f'{split}_mask_paths'].astype(str)
        paths=z[f'{split}_image_paths'].astype(str);sizes=z[f'{split}_source_sizes'];ids=z[f'{split}_sample_ids'].astype(str)
    assert maps.dtype==np.float32 and heads.dtype==np.float32 and np.isfinite(maps).all() and np.isfinite(heads).all()
    positive=np.zeros(65536,np.int64);negative=np.zeros(65536,np.int64);regions=[];boundaries=[];disappear=0
    total_tp=total_fp=total_fn=0;per=[]
    for i,(label,logit,mp,size) in enumerate(zip(labels,maps,masks,sizes)):
        shape=tuple(map(int,size));assert shape==(3008,4110)
        probability=cv2.resize(stable_probability(logit),(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
        gt=cv2.imread(mp,cv2.IMREAD_GRAYSCALE)!=0 if label else np.zeros(shape,bool)
        assert gt.shape==shape
        index=_score_bins(probability,65536)
        positive+=np.bincount(index[gt],minlength=65536);negative+=np.bincount(index[~gt],minlength=65536)
        pred=probability>=pixel_threshold
        tp=int((pred&gt).sum());fp=int((pred&~gt).sum());fn=int((~pred&gt).sum())
        total_tp+=tp;total_fp+=fp;total_fn+=fn
        distance=None
        if label:
            count,components=cv2.connectedComponents(gt.astype(np.uint8),connectivity=8)
            for component in range(1,count):regions.append(np.bincount(index[components==component],minlength=65536))
            disappear+=int(tp==0)
            distance=_mod.normalized_boundary_distance(gt,pred)
            if math.isfinite(distance):boundaries.append(distance)
        per.append({'index':i,'sample_id':ids[i],'label':int(label),'tp':tp,'fp':fp,'fn':fn,'normalized_boundary_distance':distance if distance is not None and math.isfinite(distance) else '',
            'image_score':float(stable_probability(heads[i])),'image_threshold':image_threshold,'pixel_threshold':pixel_threshold})
        if progress and (i+1)%60==0:print('pixel evaluator',split,i+1,len(labels),flush=True)
    ranking=histogram_binary_metrics(positive,negative);tb=int(np.ceil(pixel_threshold*65535))
    tp=int(positive[tb:].sum());fp=int(negative[tb:].sum());fn=int(positive[:tb].sum())
    result={**image_metrics(labels,stable_probability(heads),image_threshold),'image_threshold':float(image_threshold),'pixel_threshold':float(pixel_threshold),
        'pixel_ap':ranking['ap'],'pixel_auroc':ranking['auroc'],'pixel_dice':2*tp/max(1,2*tp+fp+fn),'pixel_iou_histogram':tp/max(1,tp+fp+fn),
        'pixel_aupro':histogram_aupro(PixelHistogram(positive,negative,regions,[],65536)),
        'dice_exact_threshold':2*total_tp/max(1,2*total_tp+total_fp+total_fn),'iou':total_tp/max(1,total_tp+total_fp+total_fn),
        'defect_region_recall':total_tp/max(1,total_tp+total_fn),'normalized_boundary_distance':float(np.mean(boundaries)),
        'normalized_boundary_distance_n':len(boundaries),'complete_disappearance_count':disappear,
        'samples':len(labels),'normal':int((labels==0).sum()),'defect':int((labels==1).sum()),'bins':65536,
        'order_sha256':__import__('hashlib').sha256('\n'.join(ids).encode()).hexdigest()}
    return result,per
