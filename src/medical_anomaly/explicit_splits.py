"""Frozen, root-relative DS1 lists, independent of historical folder split names."""
import hashlib
import json
from pathlib import Path, PureWindowsPath


def resolve_relative(root, value):
    if not value or PureWindowsPath(value).is_absolute() or '..' in PureWindowsPath(value).parts:
        raise ValueError(f'Expected safe root-relative path: {value}')
    path = Path(root).joinpath(*PureWindowsPath(value).parts).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes data root')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_explicit_splits(path, root, expected_sha256=None):
    path = Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError('Frozen split manifest SHA256 mismatch')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    assigned = {r['image_path']: r for r in manifest['assignments']}
    if len(assigned) != len(manifest['assignments']):
        raise ValueError('Repeated assigned image identity')
    group_pools = {}
    for r in assigned.values():
        prior = group_pools.setdefault(r['group_id'], r['pool'])
        if prior != r['pool']:
            raise ValueError('Content group crosses candidate pools')
    seen = set()
    result = {}
    for split in ['train', 'validation', 'test']:
        rows = manifest['splits'][split]
        if not rows:
            raise ValueError('Empty explicit split')
        translated = []
        for r in rows:
            if r['image_path'] in seen:
                raise ValueError('Image identity crosses explicit sets')
            seen.add(r['image_path'])
            source = assigned[r['image_path']]
            if source['pool'] != split or any(source[k] != v for k, v in r.items()):
                raise ValueError('Selected row differs from frozen pool assignment')
            resolve_relative(root, r['image_path'])
            if r['label'] not in [0, 1] or bool(r['mask_path']) != bool(r['label']):
                raise ValueError('Inconsistent label/mask')
            if r['label']:
                resolve_relative(root, r['mask_path'])
            translated.append({'image_paths':r['image_path'], 'notation_paths':r['mask_path'],
                               'classes':'[0 1]' if r['label'] else '[1 0]', 'group_id':r['group_id']})
        result[split] = translated
    return result, manifest, actual


def manifest_argument(args, config):
    value = getattr(args, 'split_manifest', None)
    if config.get('protocol') == 'SSI-ContentGroup-Sparse3-v1' and value is None:
        raise ValueError('Content-group configuration requires --split-manifest; archive fallback forbidden')
    return value
