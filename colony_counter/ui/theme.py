"""Dynamic theme system — dark/light toggle with persistent preference."""
import json
import os
from pathlib import Path

_THEMES = {
    'dark': dict(
        BG='#09090b', BG1='#18181b', BG2='#27272a', BG3='#3f3f46',
        BORDER='#27272a', BORDER_HI='#059669',
        FG='#d4d4d8', FG2='#a1a1aa', FG3='#71717a', FG4='#52525b',
        ACCENT='#10b981', ACCENT_DIM='#0d3024',
        ACCENT_BTN='#059669', ACCENT_HOV='#047857',
        RED='#ef4444', RED_DIM='#7f1d1d',
        ORANGE='#f59e0b', YELLOW='#eab308',
        CANVAS_BG='#0c0c0e',
    ),
    'light': dict(
        BG='#f4f4f5', BG1='#ffffff', BG2='#e4e4e7', BG3='#d4d4d8',
        BORDER='#d4d4d8', BORDER_HI='#059669',
        FG='#18181b', FG2='#3f3f46', FG3='#52525b', FG4='#71717a',
        ACCENT='#059669', ACCENT_DIM='#d1fae5',
        ACCENT_BTN='#059669', ACCENT_HOV='#047857',
        RED='#dc2626', RED_DIM='#fecaca',
        ORANGE='#d97706', YELLOW='#ca8a04',
        CANVAS_BG='#e8e8ec',
    ),
}


class T:
    """Dynamic theme singleton. Call T.apply('dark'/'light') to switch."""
    FONT = ('Consolas', 11)
    FONT_SM = ('Consolas', 10)
    FONT_XS = ('Consolas', 9)
    FONT_LG = ('Consolas', 13)
    FONT_TITLE = ('Consolas', 15, 'bold')
    FONT_HDR = ('Consolas', 12, 'bold')

    BG = BG1 = BG2 = BG3 = ''
    BORDER = BORDER_HI = ''
    FG = FG2 = FG3 = FG4 = ''
    ACCENT = ACCENT_DIM = ACCENT_BTN = ACCENT_HOV = ''
    RED = RED_DIM = ORANGE = YELLOW = CANVAS_BG = ''
    _current = 'dark'

    @classmethod
    def apply(cls, name='dark'):
        cls._current = name
        for k, v in _THEMES.get(name, _THEMES['dark']).items():
            setattr(cls, k, v)

    @classmethod
    def toggle(cls):
        cls.apply('light' if cls._current == 'dark' else 'dark')

    @classmethod
    def is_dark(cls):
        return cls._current == 'dark'


def _cfg_path():
    return Path(os.environ.get('APPDATA', str(Path.home()))) / 'ColonyCounter' / 'theme.json'


def save_theme_pref():
    p = _cfg_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(p, 'w') as f:
            json.dump({'theme': T._current}, f)
    except Exception:
        pass


def load_theme_pref():
    try:
        with open(_cfg_path(), 'r') as f:
            return json.load(f).get('theme', 'dark')
    except Exception:
        return 'dark'


# Apply default immediately
T.apply('dark')
