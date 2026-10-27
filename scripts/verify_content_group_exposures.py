from pathlib import Path
import json,csv,sys,torch
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--experiment-directory',type=Path,required=True)
parser.add_argument('--data-root',type=Path,required=True)
args=parser.parse_args()
OUT=args.experiment_directory.resolve()
manifest=json.loads((OUT/'protocol/frozen_manifest.json').read_text())
source_root=args.data_root.resolve()
folder=OUT/'training_exposure_audit';folder.mkdir(exist_ok=True)
for seed in [42,7,2024]:
    paths={m:OUT/'runs'/f'{m}_seed{seed}' for m in ['mixed','real_only']}
    if not all((p/'training_summary.json').exists() for p in paths.values()):continue
    record=folder/f'pair_seed{seed}.json'
    if record.exists():continue
    progress={};summaries={};rows=[]
    for mode,p in paths.items():
        summary=json.loads((p/'training_summary.json').read_text());summaries[mode]=summary
        checkpoint=torch.load(p/'checkpoints/latest.ckpt',map_location='cpu')
        progress[mode]=checkpoint['training_progress']
        for r in manifest['splits']['train']:
            key=str(source_root/r['image_path'])
            rows.append({'mode':mode,'seed':seed,'image_path':r['image_path'],'group_id':r['group_id'],'label':r['label'],
                         'input_exposure_count':progress[mode]['sample_exposure_counts'].get(key,0)})
        del checkpoint
    with (folder/f'per_training_image_exposures_seed{seed}.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    a,b=progress['mixed'],progress['real_only']
    result={'seed':seed,'identical_initial_weights':summaries['mixed']['initial_state_sha256']==summaries['real_only']['initial_state_sha256'],
            'same_input_exposure_counts':a['sample_exposure_counts']==b['sample_exposure_counts'],
            'same_epoch_input_order_hashes':a['epoch_order_sha256']==b['epoch_order_sha256'],
            'same_real_input_mask_positive_pixel_exposures':summaries['mixed']['exposure']['real_mask_positive_pixels']==summaries['real_only']['exposure']['real_mask_positive_pixels'],
            'each_steps_1020':a['optimizer_steps']==b['optimizer_steps']==1020,
            'real_only_synthetic_pixel_count':summaries['real_only']['exposure']['synthetic_mask_positive_pixels'],
            'note':'actual frozen-list sampling checked independently of test predictions or metrics; difference flags are reported, not used to select seeds'}
    with record.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
