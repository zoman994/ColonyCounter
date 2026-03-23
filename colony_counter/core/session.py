"""Session save/load — pure dict I/O, no tkinter."""
import json
import os
from pathlib import Path


def save_session(filepath, image_paths, display_names, manual_marks,
                 excluded_auto, dish_overrides, params_dict, current_path):
    """Save session to JSON. Returns True on success."""
    session_dir = str(Path(filepath).parent)
    imgs = []
    for path in image_paths:
        rp = path.split('::frame')[0] if '::frame' in path else path
        try:
            rel = os.path.relpath(rp, session_dir)
        except ValueError:
            rel = rp
        fs = path[len(rp):] if '::frame' in path else ''
        imgs.append({
            'path': path,
            'rel': rel + fs,
            'name': display_names.get(path, ''),
            'marks': [list(m) for m in manual_marks.get(path, [])],
            'excl': [list(c) for c in excluded_auto.get(path, [])],
            'dish_ov': dish_overrides.get(path, []),
        })
    session = {
        'v': 2,
        'cur': current_path,
        'params': params_dict,
        'imgs': imgs,
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
    return True


def load_session(filepath):
    """Load session from JSON. Returns dict with keys:
    params, images (list of dicts), current_path, missing (list of paths).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        s = json.load(f)
    session_dir = str(Path(filepath).parent)
    images = []
    missing = []
    for d in s.get('imgs', []):
        path = d['path']
        rp = path.split('::frame')[0] if '::frame' in path else path
        if not Path(rp).exists():
            rel = d.get('rel', '')
            rr = rel.split('::frame')[0] if '::frame' in rel else rel
            afr = os.path.normpath(os.path.join(session_dir, rr))
            if Path(afr).exists():
                path = afr + (path[len(rp):] if '::frame' in path else '')
            else:
                missing.append(rp)
                continue
        images.append({
            'path': path,
            'name': d.get('name') or Path(path).name,
            'marks': [tuple(m) for m in d.get('marks', [])],
            'excl': [tuple(c) for c in d.get('excl', [])],
            'dish_ov': d.get('dish_ov', []),
        })
    return {
        'params': s.get('params', {}),
        'images': images,
        'current_path': s.get('cur'),
        'missing': missing,
    }
