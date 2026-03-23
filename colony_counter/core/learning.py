"""Adaptive threshold correction via EMA."""
import json
import os
from pathlib import Path

from colony_counter.core.constants import C


class LearningEngine:
    def __init__(self):
        d = Path(os.environ.get('APPDATA', str(Path.home()))) / 'ColonyCounter'
        d.mkdir(parents=True, exist_ok=True)
        self._path = d / 'learned_params.json'
        self._state = self._load()

    def _load(self):
        try:
            with open(self._path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {'threshold_ema': None, 'samples': 0}

    def _save(self):
        try:
            with open(self._path, 'w', encoding='utf-8') as f:
                json.dump(self._state, f, indent=2)
        except Exception:
            pass

    def update(self, auto_count, excluded, added, cur_threshold):
        if auto_count < C.LEARN_MIN_AUTO:
            return None
        net = excluded / auto_count - added / auto_count
        if abs(net) < C.LEARN_MIN_RATIO:
            return None
        delta = max(-C.LEARN_MAX_DELTA, min(C.LEARN_MAX_DELTA, round(net * C.LEARN_DELTA_K)))
        sug = max(5, min(100, cur_threshold + delta))
        ema = self._state.get('threshold_ema')
        ne = sug if ema is None else (C.LEARN_ALPHA * sug + (1 - C.LEARN_ALPHA) * ema)
        self._state['threshold_ema'] = ne
        self._state['samples'] = self._state.get('samples', 0) + 1
        self._save()
        r = round(ne)
        return r if r != cur_threshold else None

    @property
    def suggestion(self):
        e = self._state.get('threshold_ema')
        return round(e) if e is not None and self._state.get('samples', 0) >= 2 else None

    @property
    def samples(self):
        return self._state.get('samples', 0)

    def reset(self):
        self._state = {'threshold_ema': None, 'samples': 0}
        self._save()
