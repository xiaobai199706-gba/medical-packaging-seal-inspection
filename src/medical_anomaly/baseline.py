from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from anomalib.utils.metrics import AUPRO
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def load_upstream(upstream_root: Path):
    root = str(Path(upstream_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from common.loss import focal_loss
    from model.supersimplenet import SuperSimpleNet
    from train import train as upstream_train
    return SuperSimpleNet, focal_loss, upstream_train


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, focal_loss, device: str, threshold: float = 0.5, clip_grad: bool = False) -> float:
    model.train(); losses = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        images = batch["image"].to(device)
        mask = F.interpolate(batch["mask"].float().to(device).unsqueeze(1), size=(model.fh, model.fw), mode="bilinear", align_corners=True)
        mask = (mask >= 0.5).float()
        label = batch["label"].float().to(device)
        anomaly_map, score, mask, label = model(images, mask, label)
        seg_focal = focal_loss(torch.sigmoid(anomaly_map), mask, reduction=None)
        seg_l1 = torch.zeros_like(anomaly_map)
        seg_l1[mask == 0] = torch.clamp(anomaly_map[mask == 0] + threshold, min=0)
        seg_l1[mask > 0] = torch.clamp(-anomaly_map[mask > 0] + threshold, min=0)
        segmented = torch.cat((batch["is_segmented"], batch["is_segmented"])).bool().to(device)
        good, bad, focal = seg_l1[segmented][mask[segmented] == 0], seg_l1[segmented][mask[segmented] > 0], seg_focal[segmented]
        seg_loss = (good.mean() if good.numel() else 0) + (bad.mean() if bad.numel() else 0) + (focal.mean() if focal.numel() else 0)
        loss = seg_loss + focal_loss(torch.sigmoid(score), label)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step(); losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict(model, loader, device: str, progress=None) -> dict[str, object]:
    model.eval(); maps=[]; masks=[]; scores=[]; labels=[]; paths=[]; mask_paths=[]
    processed = 0
    for batch in loader:
        anomaly_map, score = model(batch["image"].to(device))
        maps.append(torch.sigmoid(anomaly_map).cpu().numpy()[:, 0])
        masks.append(batch["mask"].numpy()); scores.append(torch.sigmoid(score).cpu().numpy())
        labels.append(batch["label"].numpy()); paths.extend(batch["image_path"])
        mask_paths.extend(batch.get("mask_path", [""] * len(batch["image_path"])))
        processed += len(batch["image_path"])
        if progress is not None:
            progress(processed, len(loader.dataset))
    return {"maps": np.concatenate(maps), "masks": np.concatenate(masks).astype(np.uint8),
            "scores": np.concatenate(scores), "labels": np.concatenate(labels), "paths": paths,
            "mask_paths": mask_paths}


def best_threshold(target: np.ndarray, score: np.ndarray, objective: str) -> float:
    target, score = np.asarray(target).astype(np.uint8).ravel(), np.asarray(score).ravel()
    order = np.argsort(score)[::-1]
    sorted_score, sorted_target = score[order], target[order]
    true_positive = np.cumsum(sorted_target, dtype=np.int64)
    false_positive = np.cumsum(1 - sorted_target, dtype=np.int64)
    candidate_index = np.flatnonzero(np.r_[sorted_score[:-1] != sorted_score[1:], True])
    tp, fp = true_positive[candidate_index], false_positive[candidate_index]
    if objective == "dice":
        values = 2 * tp / np.maximum(1, tp + fp + sorted_target.sum())
    else:
        values = tp / max(1, sorted_target.sum()) - fp / max(1, len(sorted_target) - sorted_target.sum())
    return float(sorted_score[candidate_index[int(np.argmax(values))]])


def aupro(masks: np.ndarray, maps: np.ndarray, max_fpr: float = 0.3) -> float:
    metric = AUPRO(fpr_limit=max_fpr)
    metric.update(torch.from_numpy(maps).float(), torch.from_numpy(masks).float())
    return float(metric.compute())


def evaluate(test: dict[str, object], image_threshold: float, pixel_threshold: float) -> dict[str, float]:
    labels=np.asarray(test["labels"]); scores=np.asarray(test["scores"]); masks=np.asarray(test["masks"]); maps=np.asarray(test["maps"])
    pred=scores >= image_threshold; tn,fp,fn,tp=confusion_matrix(labels,pred,labels=[0,1]).ravel()
    pixel_pred=maps >= pixel_threshold; intersection=np.logical_and(pixel_pred,masks).sum()
    return {"image_auroc":roc_auc_score(labels,scores), "image_ap":average_precision_score(labels,scores),
            "image_recall":tp/max(1,tp+fn), "image_specificity":tn/max(1,tn+fp),
            "pixel_auroc":roc_auc_score(masks.ravel(),maps.ravel()), "pixel_ap":average_precision_score(masks.ravel(),maps.ravel()),
            "pixel_dice":2*intersection/max(1,pixel_pred.sum()+masks.sum()), "pixel_aupro":aupro(masks,maps),
            "image_threshold":image_threshold, "pixel_threshold":pixel_threshold, "false_positives":int(fp), "false_negatives":int(fn)}


def save_error_panels(result: dict[str, object], image_threshold: float, pixel_threshold: float, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    labels=np.asarray(result["labels"]); scores=np.asarray(result["scores"])
    normal = np.where(labels == 0)[0][np.argsort(scores[labels == 0])[::-1]]
    abnormal = np.where(labels == 1)[0][np.argsort(scores[labels == 1])]
    fp = np.concatenate((normal[scores[normal] >= image_threshold], normal[scores[normal] < image_threshold]))[:5]
    fn = np.concatenate((abnormal[scores[abnormal] < image_threshold], abnormal[scores[abnormal] >= image_threshold]))[:5]
    manifest = []
    for kind, indices in (("false_positive",fp),("false_negative",fn)):
        for rank,index in enumerate(indices,1):
            image=cv2.cvtColor(cv2.imread(result["paths"][index]),cv2.COLOR_BGR2RGB)
            image=cv2.resize(image,(256,256)); mask=result["masks"][index]; heat=result["maps"][index]; prediction=heat >= pixel_threshold
            fig,axes=plt.subplots(1,4,figsize=(12,3)); titles=["image","ground truth","anomaly heatmap","prediction"]
            for ax,data,title in zip(axes,[image,mask,heat,prediction],titles): ax.imshow(data,cmap=None if title=="image" else "inferno"); ax.set_title(title); ax.axis("off")
            fig.suptitle(f"score={scores[index]:.4f}, image_threshold={image_threshold:.4f}"); fig.tight_layout()
            fig.savefig(output/f"{kind}_{rank:02d}.jpg",dpi=120); plt.close(fig)
            is_error = bool(scores[index] >= image_threshold) if kind == "false_positive" else bool(scores[index] < image_threshold)
            manifest.append({"kind": kind, "rank": rank, "image_path": result["paths"][index],
                             "score": float(scores[index]), "threshold": image_threshold, "is_threshold_error": is_error})
    save_json(output / "manifest.json", manifest)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value,indent=2),encoding="utf-8")
