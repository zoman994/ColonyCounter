"""App controller — orchestrates UI, processing, and export."""
import atexit
import os
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path
import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk

from colony_counter.core.constants import C
from colony_counter.core.io_utils import cv_imread, load_tiff_frame, count_tiff_frames
from colony_counter.core.processing import ImageProcessor
from colony_counter.core.cache import LazyImageCache
from colony_counter.core.learning import LearningEngine
from colony_counter.core import calculations as calc
from colony_counter.core import session as sess
from colony_counter.core.io_utils import cv_imwrite
from colony_counter.export import excel_export, csv_export, pdf_export, image_export
from colony_counter.ui.theme import T, save_theme_pref, load_theme_pref
from colony_counter.ui.widgets import DarkButton, DarkCheck, DarkSlider, DarkSection
from colony_counter.ui.logo import LOGO_LIGHT_B64, LOGO_DARK_B64

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class App:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Colony Counter v{C.VERSION}")
        self.root.geometry("1440x900")
        self.root.minsize(1100, 700)
        T.apply(load_theme_pref())
        self.root.config(bg=T.BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Core
        self.processor = ImageProcessor()
        self.learner = LearningEngine()
        self._cache = LazyImageCache()
        self._lock = threading.Lock()
        atexit.register(self._cache.cleanup)

        # State
        self.image_paths = []
        self.image_data = {}
        self.current_path = None
        self._prev_path = None
        self._pil_orig = None
        self._pil_proc = None
        self._proc_transform = None
        self.manual_marks = {}
        self.excluded_auto = {}
        self.display_names = {}
        self.dish_overrides = {}
        self._dish_edit = False
        self._dish_drag = None
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self._pan_drag = None
        self._processing = False
        self._compare_pos = 0.5
        self._undo_stack = []
        self._redo_stack = []
        self._annotations = {}

        # Parameters (tk vars)
        self.p = dict(
            min_diam_mm=tk.DoubleVar(value=0.3),   # min colony diameter in mm
            max_diam_mm=tk.DoubleVar(value=3.0),    # max single colony diameter in mm
            threshold=tk.IntVar(value=25),
            filter_bubbles=tk.BooleanVar(value=True),
            use_watershed=tk.BooleanVar(value=True),
            show_numbers=tk.BooleanVar(value=True),
            filter_elongated=tk.BooleanVar(value=True),
            filter_nonconvex=tk.BooleanVar(value=True),
            auto_learn=tk.BooleanVar(value=True),
            detect_label=tk.BooleanVar(value=True),
            detect_light_label=tk.BooleanVar(value=False),
            use_otsu=tk.BooleanVar(value=False),
            use_color_filter=tk.BooleanVar(value=False),
            dish_diameter_mm=tk.DoubleVar(value=90.0),
            plating_volume_ml=tk.DoubleVar(value=0.1),
            dilution_factor=tk.DoubleVar(value=1.0),
            dilution_group=tk.StringVar(value=""),
        )
        s = self.learner.suggestion
        if s:
            self.p['threshold'].set(s)

        self._build_ui()
        self._setup_keys()

    # ═══════════ LIFECYCLE ═══════════════════════════════════════════════

    def _on_close(self):
        self._cache.cleanup()
        self.root.destroy()

    def _setup_keys(self):
        binds = {
            '<Control-o>': lambda e: self._add_images(),
            '<Control-s>': lambda e: self._export_excel(),
            '<Control-e>': lambda e: self._export_image(),
            '<Control-S>': lambda e: self._save_session(),
            '<Control-O>': lambda e: self._load_session(),
            '<Left>': lambda e: self._navigate(-1),
            '<Right>': lambda e: self._navigate(1),
            '<space>': lambda e: self._process_current(),
            '<Delete>': lambda e: self._remove_image(),
            '<Control-z>': lambda e: self._undo(),
            '<Control-y>': lambda e: self._redo(),
            '<F1>': lambda e: self._apply_preset('default'),
            '<F2>': lambda e: self._apply_preset('sensitive'),
            '<F3>': lambda e: self._apply_preset('strict'),
            '<F4>': lambda e: self._apply_preset('large'),
        }
        for k, cmd in binds.items():
            self.root.bind(k, cmd)

    def _toggle_theme(self):
        T.toggle()
        self.root.config(bg=T.BG)
        sel_idx = None
        sel = self.listbox.curselection()
        if sel:
            sel_idx = sel[0]
        for w in self.root.winfo_children():
            w.destroy()
        self._build_ui()
        for path in self.image_paths:
            self.listbox.insert(tk.END, self.display_names.get(path, Path(path).name))
        if sel_idx is not None and sel_idx < len(self.image_paths):
            self.listbox.selection_set(sel_idx)
            self.listbox.see(sel_idx)
        self._refresh_results()
        if self.current_path and self.image_data.get(self.current_path):
            self._refresh_stats(self.image_data[self.current_path])
            self._switch_tab('result')
            self.root.after(80, self._refresh_proc_canvas)
        self._refresh_learn_label()
        save_theme_pref()

    # ═══════════ BUILD UI ════════════════════════════════════════════════

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=T.BG, height=40)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        # RES lab logo
        import base64
        import io
        logo_b64 = LOGO_DARK_B64 if T.is_dark() else LOGO_LIGHT_B64
        logo_data = base64.b64decode(logo_b64)
        logo_pil = Image.open(io.BytesIO(logo_data))
        self._logo_photo = ImageTk.PhotoImage(logo_pil)
        tk.Label(hdr, image=self._logo_photo, bg=T.BG).pack(side=tk.LEFT, padx=(12, 8))
        # App name + version
        tk.Label(hdr, text="ColonyCounter", bg=T.BG, fg=T.ACCENT,
                 font=T.FONT_TITLE).pack(side=tk.LEFT)
        tk.Label(hdr, text=f"v{C.VERSION}", bg=T.BG, fg=T.FG4,
                 font=T.FONT_XS).pack(side=tk.LEFT, padx=(6, 0), pady=(4, 0))
        hdr_r = tk.Frame(hdr, bg=T.BG)
        hdr_r.pack(side=tk.RIGHT, padx=12)
        icon = "\u263e" if T.is_dark() else "\u2600"
        DarkButton(hdr_r, icon, self._toggle_theme, 'ghost', small=True).pack(side=tk.LEFT, padx=2)
        tk.Frame(hdr_r, bg=T.BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=6)
        for txt, cmd in [("Excel", self._export_excel), ("CSV", self._export_csv),
                         ("PDF", self._export_pdf if HAS_MPL else None),
                         ("Сессия", self._save_session), ("Открыть", self._load_session)]:
            if cmd:
                DarkButton(hdr_r, txt, cmd, 'ghost', small=True).pack(side=tk.LEFT, padx=2)

        tk.Frame(self.root, bg=T.BORDER, height=1).pack(fill=tk.X)

        # Progress bar
        pf = tk.Frame(self.root, bg=T.BG, height=3)
        pf.pack(fill=tk.X)
        pf.pack_propagate(False)
        self._prog_bar = tk.Frame(pf, bg=T.ACCENT, height=3)
        self._prog_bar.place(x=0, y=0, height=3, relwidth=0)

        # Body — resizable 3-pane layout
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                               bg=T.BORDER, sashwidth=5, sashrelief=tk.FLAT,
                               opaqueresize=False)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        left = tk.Frame(paned, bg=T.BG1, highlightthickness=1, highlightbackground=T.BORDER)
        self._build_left(left)
        center = tk.Frame(paned, bg=T.BG)
        self._build_center(center)
        right = tk.Frame(paned, bg=T.BG1, highlightthickness=1, highlightbackground=T.BORDER)
        self._build_right(right)
        paned.add(left, minsize=160, width=220)
        paned.add(center, minsize=400)
        paned.add(right, minsize=200, width=300)

        # Status
        tk.Frame(self.root, bg=T.BORDER, height=1).pack(fill=tk.X)
        sf = tk.Frame(self.root, bg=T.BG, height=24)
        sf.pack(fill=tk.X)
        sf.pack_propagate(False)
        self._status = tk.Label(sf, text="Ctrl+O файлы | Space обработка | Ctrl+Z отмена",
                                bg=T.BG, fg=T.FG4, font=T.FONT_XS, anchor='w')
        self._status.pack(fill=tk.X, padx=12)

    def _build_left(self, parent):
        tk.Label(parent, text="ИЗОБРАЖЕНИЯ", bg=T.BG1, fg=T.FG4, font=T.FONT_XS).pack(anchor=tk.W, padx=8, pady=(8, 4))
        lf = tk.Frame(parent, bg=T.BG)
        lf.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))
        sb = tk.Scrollbar(lf, troughcolor=T.BG2, bg=T.BG3, highlightthickness=0, bd=0)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(lf, yscrollcommand=sb.set, bg=T.BG, fg=T.FG, font=T.FONT_SM,
                                  selectmode=tk.SINGLE, selectbackground=T.ACCENT_DIM,
                                  selectforeground=T.ACCENT, highlightthickness=0, bd=0,
                                  activestyle='none', relief='flat')
        self.listbox.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.listbox.yview)
        self.listbox.bind('<<ListboxSelect>>', self._on_select)
        self.listbox.bind('<Double-Button-1>', lambda e: self._rename_image())
        bf = tk.Frame(parent, bg=T.BG1)
        bf.pack(fill=tk.X, padx=6, pady=2)
        DarkButton(bf, "+ Файлы", self._add_images, 'secondary', small=True).pack(side=tk.LEFT, padx=(0, 2), fill=tk.X, expand=True)
        DarkButton(bf, "+ Папка", self._add_folder, 'secondary', small=True).pack(side=tk.LEFT, padx=(0, 2), fill=tk.X, expand=True)
        DarkButton(bf, "- Удал.", self._remove_image, 'ghost', small=True).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Frame(parent, bg=T.BORDER, height=1).pack(fill=tk.X, padx=6, pady=6)
        DarkButton(parent, "ОБРАБОТАТЬ ВСЕ", self._process_all, 'primary').pack(fill=tk.X, padx=6, pady=2)
        tk.Frame(parent, bg=T.BORDER, height=1).pack(fill=tk.X, padx=6, pady=6)
        tk.Label(parent, text="РЕЗУЛЬТАТЫ", bg=T.BG1, fg=T.FG4, font=T.FONT_XS).pack(anchor=tk.W, padx=8, pady=(0, 2))
        self._res_text = tk.Text(parent, bg=T.BG, fg=T.FG3, font=T.FONT_SM, height=10,
                                 state=tk.DISABLED, highlightthickness=0, bd=0, relief='flat')
        self._res_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

    def _build_center(self, parent):
        tab_bar = tk.Frame(parent, bg=T.BG)
        tab_bar.pack(fill=tk.X, pady=(0, 4))
        self._tabs = {}
        for tid, label in [('original', 'Оригинал'), ('result', 'Результат'), ('compare', 'Сравнение')]:
            b = tk.Label(tab_bar, text=label, bg=T.BG, fg=T.FG4, font=T.FONT_SM, padx=12, pady=4, cursor='hand2')
            b.pack(side=tk.LEFT)
            b.bind('<Button-1>', lambda e, t=tid: self._switch_tab(t))
            self._tabs[tid] = b
        tb = tk.Frame(tab_bar, bg=T.BG)
        tb.pack(side=tk.RIGHT)
        for txt, cmd in [("Зум x1", self._reset_zoom), ("Отмена", self._undo), ("Очист.", self._clear_manual), ("Восст.", self._restore_auto)]:
            DarkButton(tb, txt, cmd, 'ghost', small=True).pack(side=tk.LEFT, padx=1)
        tk.Frame(tb, bg=T.BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)
        self._dish_btn = DarkButton(tb, "Границы", self._toggle_dish_edit, 'ghost', small=True)
        self._dish_btn.pack(side=tk.LEFT, padx=1)
        DarkButton(tb, "Сброс", self._reset_dish_overrides, 'ghost', small=True).pack(side=tk.LEFT, padx=1)

        cf = tk.Frame(parent, bg=T.CANVAS_BG, highlightthickness=1, highlightbackground=T.BORDER)
        cf.pack(fill=tk.BOTH, expand=True)
        self.canvas_orig = tk.Canvas(cf, bg=T.CANVAS_BG, highlightthickness=0, cursor='crosshair')
        self.canvas_proc = tk.Canvas(cf, bg=T.CANVAS_BG, highlightthickness=0, cursor='tcross')
        # Compare tab — two-image side-by-side
        self._compare_frame = tk.Frame(cf, bg=T.CANVAS_BG)
        self._compare_sel = tk.Frame(self._compare_frame, bg=T.BG1)
        self._compare_sel.pack(fill=tk.X, padx=4, pady=4)
        tk.Label(self._compare_sel, text="Левое:", bg=T.BG1, fg=T.FG3, font=T.FONT_XS).pack(side=tk.LEFT, padx=(0, 4))
        self._cmp_var_l = tk.StringVar()
        self._cmp_menu_l = tk.OptionMenu(self._compare_sel, self._cmp_var_l, "")
        self._cmp_menu_l.config(bg=T.BG2, fg=T.FG, font=T.FONT_XS, highlightthickness=0)
        self._cmp_menu_l.pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(self._compare_sel, text="Правое:", bg=T.BG1, fg=T.FG3, font=T.FONT_XS).pack(side=tk.LEFT, padx=(0, 4))
        self._cmp_var_r = tk.StringVar()
        self._cmp_menu_r = tk.OptionMenu(self._compare_sel, self._cmp_var_r, "")
        self._cmp_menu_r.config(bg=T.BG2, fg=T.FG, font=T.FONT_XS, highlightthickness=0)
        self._cmp_menu_r.pack(side=tk.LEFT)
        DarkButton(self._compare_sel, "Показать", self._draw_comparison, 'primary', small=True).pack(side=tk.LEFT, padx=8)
        self.canvas_compare = tk.Canvas(self._compare_frame, bg=T.CANVAS_BG, highlightthickness=0)
        self.canvas_compare.pack(fill=tk.BOTH, expand=True)
        self._switch_tab('result')

        self.canvas_orig.bind("<Configure>", lambda e: self._redraw(self.canvas_orig, self._pil_orig))
        self.canvas_proc.bind("<Configure>", lambda e: self._refresh_proc_canvas())
        self.canvas_proc.bind("<Button-1>", self._on_lmb_down)
        self.canvas_proc.bind("<B1-Motion>", self._on_lmb_motion)
        self.canvas_proc.bind("<ButtonRelease-1>", self._on_lmb_up)
        self.canvas_proc.bind("<Button-3>", self._on_proc_rclick)
        self.canvas_proc.bind("<MouseWheel>", self._on_proc_wheel)
        self.canvas_proc.bind("<ButtonPress-2>", self._on_proc_pan_start)
        self.canvas_proc.bind("<B2-Motion>", self._on_proc_pan_move)
        self.canvas_proc.bind("<ButtonRelease-2>", lambda e: setattr(self, '_pan_drag', None))
        self.canvas_compare.bind("<Configure>", lambda e: self._draw_comparison())

        tk.Label(parent, text="ЛКМ-метка  ПКМ-убрать  Колесо-зум  СКМ-пан  F1-F4 пресеты",
                 bg=T.BG, fg=T.FG4, font=T.FONT_XS).pack(anchor=tk.W, pady=(4, 0))

    def _build_right(self, parent):
        # Tabbed notebook for right panel
        nb = tk.Frame(parent, bg=T.BG1)
        nb.pack(fill=tk.BOTH, expand=True)
        # Tab buttons
        tab_row = tk.Frame(nb, bg=T.BG1)
        tab_row.pack(fill=tk.X, padx=4, pady=(4, 0))
        self._rtabs = {}
        self._rtab_frames = {}
        for tid, label in [('detect', 'Обнар.'), ('advanced', 'Доп.'), ('calib', 'Калибр.'), ('actions', 'Действ.')]:
            f = tk.Frame(nb, bg=T.BG1)
            self._rtab_frames[tid] = f
            b = tk.Label(tab_row, text=label, bg=T.BG1, fg=T.FG4, font=T.FONT_XS, padx=8, pady=3, cursor='hand2')
            b.pack(side=tk.LEFT, expand=True, fill=tk.X)
            b.bind('<Button-1>', lambda e, t=tid: self._switch_rtab(t))
            self._rtabs[tid] = b
        self._build_tab_detect(self._rtab_frames['detect'])
        self._build_tab_advanced(self._rtab_frames['advanced'])
        self._build_tab_calib(self._rtab_frames['calib'])
        self._build_tab_actions(self._rtab_frames['actions'])
        self._switch_rtab('detect')

    def _switch_rtab(self, tid):
        for t, w in self._rtabs.items():
            w.config(bg=T.ACCENT_DIM if t == tid else T.BG1, fg=T.ACCENT if t == tid else T.FG4)
        for t, f in self._rtab_frames.items():
            f.pack_forget()
        self._rtab_frames[tid].pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _build_tab_detect(self, parent):
        s1 = DarkSection(parent, "Обнаружение")
        s1.pack(fill=tk.X, pady=4)
        DarkSlider(s1.body, "Мин. диаметр (мм)", self.p['min_diam_mm'], 0.1, 5.0, 0.1).pack(fill=tk.X, pady=2)
        DarkSlider(s1.body, "Макс. диаметр (мм)", self.p['max_diam_mm'], 0.5, 20.0, 0.1).pack(fill=tk.X, pady=2)
        DarkSlider(s1.body, "Порог", self.p['threshold'], 5, 100, 1).pack(fill=tk.X, pady=2)
        s2 = DarkSection(parent, "Фильтры")
        s2.pack(fill=tk.X, pady=4)
        for txt, var in [("Пузыри воздуха", self.p['filter_bubbles']),
                         ("Watershed кластеры", self.p['use_watershed']),
                         ("Числа на картинке", self.p['show_numbers']),
                         ("Вытянутые объекты", self.p['filter_elongated']),
                         ("Невыпуклые объекты", self.p['filter_nonconvex']),
                         ("Маскировать этикетку", self.p['detect_label'])]:
            DarkCheck(s2.body, txt, var).pack(fill=tk.X, pady=1)

    def _build_tab_advanced(self, parent):
        s3 = DarkSection(parent, "v2.0")
        s3.pack(fill=tk.X, pady=4)
        DarkCheck(s3.body, "Светлые этикетки", self.p['detect_light_label']).pack(fill=tk.X, pady=1)
        DarkCheck(s3.body, "Otsu per-image", self.p['use_otsu']).pack(fill=tk.X, pady=1)
        DarkCheck(s3.body, "Цветовой фильтр HSV", self.p['use_color_filter']).pack(fill=tk.X, pady=1)
        s4 = DarkSection(parent, "Авто-обучение")
        s4.pack(fill=tk.X, pady=4)
        DarkCheck(s4.body, "Обучаться по правкам", self.p['auto_learn']).pack(fill=tk.X, pady=1)
        self._learn_lbl = tk.Label(s4.body, text="", bg=T.BG1, fg=T.FG4, font=T.FONT_XS, anchor='w')
        self._learn_lbl.pack(fill=tk.X, pady=(2, 0))
        lr = tk.Frame(s4.body, bg=T.BG1)
        lr.pack(fill=tk.X, pady=(4, 0))
        DarkButton(lr, "Применить", self._apply_learned, 'secondary', small=True).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        DarkButton(lr, "Сброс", self._reset_learning, 'danger', small=True).pack(side=tk.LEFT, expand=True, fill=tk.X)
        self._refresh_learn_label()

    def _build_tab_calib(self, parent):
        sc = DarkSection(parent, "Калибровка")
        sc.pack(fill=tk.X, pady=4)
        DarkSlider(sc.body, "Диаметр чашки (мм)", self.p['dish_diameter_mm'], 30, 150, 1).pack(fill=tk.X, pady=2)
        sf = DarkSection(parent, "CFU/мл")
        sf.pack(fill=tk.X, pady=4)
        DarkSlider(sf.body, "Объём посева (мл)", self.p['plating_volume_ml'], 0.01, 1.0, 0.01).pack(fill=tk.X, pady=2)
        # Dilution as text entry (1, 10, 100, 1000, etc.)
        dr = tk.Frame(sf.body, bg=T.BG1)
        dr.pack(fill=tk.X, pady=2)
        tk.Label(dr, text="Разведение 1:", bg=T.BG1, fg=T.FG3, font=T.FONT_XS).pack(side=tk.LEFT)
        tk.Entry(dr, textvariable=self.p['dilution_factor'], bg=T.BG2, fg=T.FG,
                 font=T.FONT_SM, insertbackground=T.FG, highlightthickness=1,
                 highlightbackground=T.BORDER, bd=0, width=10).pack(side=tk.LEFT, padx=4)
        tk.Label(dr, text="(1=чистая)", bg=T.BG1, fg=T.FG4, font=T.FONT_XS).pack(side=tk.LEFT)
        sdg = DarkSection(parent, "Серия разведений")
        sdg.pack(fill=tk.X, pady=4)
        gr = tk.Frame(sdg.body, bg=T.BG1)
        gr.pack(fill=tk.X)
        tk.Label(gr, text="Группа:", bg=T.BG1, fg=T.FG3, font=T.FONT_XS).pack(side=tk.LEFT)
        tk.Entry(gr, textvariable=self.p['dilution_group'], bg=T.BG2, fg=T.FG, font=T.FONT_SM,
                 insertbackground=T.FG, highlightthickness=1, highlightbackground=T.BORDER, bd=0, width=10).pack(side=tk.LEFT, padx=4)

    def _build_tab_actions(self, parent):
        # Presets
        sp = DarkSection(parent, "Пресеты F1-F4")
        sp.pack(fill=tk.X, pady=4)
        pr = tk.Frame(sp.body, bg=T.BG1)
        pr.pack(fill=tk.X)
        for key, label in [('default', 'F1 Стд'), ('sensitive', 'F2 Чувст'), ('strict', 'F3 Строг'), ('large', 'F4 Крупн')]:
            DarkButton(pr, label, lambda k=key: self._apply_preset(k), 'ghost', small=True).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        # Buttons
        DarkButton(parent, "ОБРАБОТАТЬ ТЕКУЩЕЕ", self._process_current, 'primary').pack(fill=tk.X, pady=4)
        DarkButton(parent, "Авто-порог (Otsu)", self._auto_threshold, 'secondary', small=True).pack(fill=tk.X, pady=1)
        DarkButton(parent, "Аннотация", self._add_annotation, 'ghost', small=True).pack(fill=tk.X, pady=1)
        if HAS_MPL:
            r = tk.Frame(parent, bg=T.BG1)
            r.pack(fill=tk.X, pady=1)
            DarkButton(r, "Стат.", self._show_statistics, 'ghost', small=True).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 1))
            DarkButton(r, "Heatmap", self._show_heatmap, 'ghost', small=True).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
            DarkButton(r, "Воспр.", self._show_reproducibility, 'ghost', small=True).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(1, 0))
        # Stats
        s5 = DarkSection(parent, "Статистика")
        s5.pack(fill=tk.X, pady=4)
        self._stats_text = tk.Text(s5.body, bg=T.BG, fg=T.FG3, font=T.FONT_SM, height=12,
                                   state=tk.DISABLED, highlightthickness=0, bd=0, relief='flat')
        self._stats_text.pack(fill=tk.X)
        # Legend
        s6 = DarkSection(parent, "Легенда")
        s6.pack(fill=tk.X, pady=4)
        for color, label in [(T.ACCENT, "одиночная"), ('#3b82f6', "кластер"),
                             (T.YELLOW, "ручная"), (T.RED, "исключённая")]:
            r = tk.Frame(s6.body, bg=T.BG1)
            r.pack(fill=tk.X, pady=1)
            tk.Canvas(r, width=10, height=10, bg=color, highlightthickness=0).pack(side=tk.LEFT, padx=(0, 6))
            tk.Label(r, text=label, bg=T.BG1, fg=T.FG4, font=T.FONT_XS).pack(side=tk.LEFT)

    # ═══════════ TAB SWITCHING ═══════════════════════════════════════════

    def _switch_tab(self, tid):
        for t, w in self._tabs.items():
            w.config(bg=T.ACCENT_DIM if t == tid else T.BG, fg=T.ACCENT if t == tid else T.FG4)
        for c in (self.canvas_orig, self.canvas_proc, self._compare_frame):
            c.pack_forget()
        if tid == 'original':
            self.canvas_orig.pack(fill=tk.BOTH, expand=True)
            if self._pil_orig:
                self.root.after(30, lambda: self._blit(self._pil_orig, self.canvas_orig))
        elif tid == 'result':
            self.canvas_proc.pack(fill=tk.BOTH, expand=True)
            self.root.after(30, self._refresh_proc_canvas)
        else:
            self._compare_frame.pack(fill=tk.BOTH, expand=True)
            self._update_compare_menus()
            self.root.after(30, self._draw_comparison)

    # ═══════════ FILE OPS ════════════════════════════════════════════════

    def _add_images(self):
        paths = filedialog.askopenfilenames(
            title="Изображения",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"), ("Все", "*.*")])
        self._add_paths(list(paths))

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Папка")
        if not folder:
            return
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        paths = [str(e) for e in sorted(Path(folder).iterdir()) if e.is_file() and e.suffix.lower() in exts]
        self._add_paths(paths)

    def _add_paths(self, paths):
        added = 0
        for path in paths:
            nf = count_tiff_frames(path)
            if nf > 1 and path.lower().endswith(('.tiff', '.tif')):
                for fi in range(nf):
                    vp = f"{path}::frame{fi}"
                    if vp not in self.image_paths:
                        self.image_paths.append(vp)
                        self.image_data[vp] = None
                        nm = f"{Path(path).name} [#{fi + 1}]"
                        self.display_names[vp] = nm
                        self.listbox.insert(tk.END, nm)
                        added += 1
            else:
                if path not in self.image_paths:
                    self.image_paths.append(path)
                    self.image_data[path] = None
                    self.display_names[path] = Path(path).name
                    self.listbox.insert(tk.END, Path(path).name)
                    added += 1
        if added:
            self._set_status(f"Добавлено {added} изображений")

    def _remove_image(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        path = self.image_paths.pop(idx)
        self.image_data.pop(path, None)
        self.display_names.pop(path, None)
        self._cache.remove(f"{path}_a")
        self._cache.remove(f"{path}_c")
        self.listbox.delete(idx)
        if self.current_path == path:
            self.current_path = None
            self._pil_orig = None
            self._pil_proc = None
            self.canvas_orig.delete("all")
            self.canvas_proc.delete("all")
        self._refresh_results()

    def _rename_image(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        path = self.image_paths[idx]
        nm = simpledialog.askstring("Переименовать", "Имя:",
                                    initialvalue=self.display_names.get(path, ''), parent=self.root)
        if not nm or not nm.strip():
            return
        self.display_names[path] = nm.strip()
        self.listbox.delete(idx)
        self.listbox.insert(idx, nm.strip())
        self.listbox.selection_set(idx)
        self._refresh_results()

    def _navigate(self, d):
        if not self.image_paths:
            return
        sel = self.listbox.curselection()
        idx = sel[0] if sel else 0
        ni = max(0, min(len(self.image_paths) - 1, idx + d))
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(ni)
        self.listbox.see(ni)
        self._on_select()

    def _on_select(self, _e=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        path = self.image_paths[sel[0]]
        if self._prev_path and self._prev_path != path:
            self._do_learning(self._prev_path)
        self._prev_path = path
        self.current_path = path
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        result = self.image_data.get(path)
        if result:
            self._refresh_stats(result)
            self._switch_tab('result')
            self.root.after(60, self._refresh_proc_canvas)
        else:
            self._pil_proc = None
            self.canvas_proc.delete("all")
            self._switch_tab('original')
        self.root.after(60, lambda p=path: self._show_file(p, self.canvas_orig))

    # ═══════════ PROCESSING ══════════════════════════════════════════════

    def _get_params(self):
        """Collect all params. mm→px conversion happens inside processor
        after real dish radius is known (two-pass calibration)."""
        return {k: v.get() for k, v in self.p.items()}

    def _process_image(self, path, params, progress_cb=None):
        """Unified: handles TIFF frames, temp files, cleanup."""
        if '::frame' in path:
            fi = load_tiff_frame(path)
            if fi is None:
                raise ValueError(f"Не удалось загрузить: {path}")
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            try:
                cv_imwrite(tmp.name, fi)
                return self.processor.process(tmp.name, params, progress_cb)
            finally:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
        return self.processor.process(path, params, progress_cb)

    def _process_current(self):
        if not self.current_path or self._processing:
            return
        self._run_threaded([self.current_path])

    def _process_all(self):
        if not self.image_paths or self._processing:
            return
        self._run_threaded(list(self.image_paths))

    def _run_threaded(self, paths):
        self._processing = True
        # Snapshot all params and overrides BEFORE thread starts (thread safety)
        frozen_params = self._get_params()
        frozen_overrides = {p: list(v) for p, v in self.dish_overrides.items()}

        def worker():
            n = len(paths)
            for i, path in enumerate(paths):
                try:
                    params = dict(frozen_params)
                    ov = frozen_overrides.get(path)
                    if ov:
                        params['dish_overrides'] = ov
                    _path = path

                    def pcb(frac, msg, _i=i, _n=n, _p=_path):
                        overall = (_i + frac) / _n
                        self.root.after(0, lambda v=overall, m=msg, pp=_p: (
                            self._prog_bar.place_configure(relwidth=v),
                            self._set_status(f"[{_i + 1}/{_n}] {Path(pp.split('::')[0]).name}: {m}")))

                    result = self._process_image(path, params, pcb)
                    self._cache.store(f"{path}_a", result['annotated'])
                    self._cache.store(f"{path}_c", result['img_clean'])
                    rl = {k: v for k, v in result.items() if k not in ('annotated', 'img_clean')}
                    rl['_cached'] = True
                    self.root.after(0, lambda p=path, r=rl: self._on_done(p, r))
                except Exception as exc:
                    self.root.after(0, lambda e=exc: self._set_status(f"Ошибка: {e}"))
            self.root.after(0, self._on_all_done)

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, path, result):
        with self._lock:
            self.image_data[path] = result
        if self.current_path == path:
            self._refresh_stats(result)
            self._switch_tab('result')
            self.root.after(60, self._refresh_proc_canvas)

    def _on_all_done(self):
        self._processing = False
        self._prog_bar.place_configure(relwidth=1.0)
        self.root.after(1500, lambda: self._prog_bar.place_configure(relwidth=0))
        self._refresh_results()
        n = sum(1 for r in self.image_data.values() if r)
        self._set_status(f"Готово. Обработано: {n}")

    def _run_sync(self, path, silent=False):
        try:
            params = self._get_params()
            ov = self.dish_overrides.get(path)
            if ov:
                params['dish_overrides'] = ov
            result = self._process_image(path, params)
            self._cache.store(f"{path}_a", result['annotated'])
            self._cache.store(f"{path}_c", result['img_clean'])
            rl = {k: v for k, v in result.items() if k not in ('annotated', 'img_clean')}
            rl['_cached'] = True
            with self._lock:
                self.image_data[path] = rl
            if self.current_path == path:
                self._refresh_stats(rl)
                self.root.after(60, self._refresh_proc_canvas)
        except Exception as exc:
            if not silent:
                messagebox.showerror("Ошибка", str(exc))

    def _auto_threshold(self):
        if not self.current_path:
            return
        try:
            if '::frame' in self.current_path:
                img = load_tiff_frame(self.current_path)
            else:
                img = cv_imread(self.current_path)
            if img is None:
                return
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cx, cy, r = self.processor.detect_dish(gray)
            h, w = gray.shape
            mask = np.zeros((h, w), np.uint8)
            cv2.circle(mask, (cx, cy), int(r * C.DISH_MASK_RATIO), 255, -1)
            norm = self.processor.normalize_background(gray, mask)
            enh = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP, tileGridSize=C.CLAHE_TILE).apply(norm)
            ov, _ = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            s = max(5, int(ov * 0.48))
            self.p['threshold'].set(s)
            self._set_status(f"Otsu: {s}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    # ═══════════ DISPLAY ═════════════════════════════════════════════════

    def _set_status(self, text):
        self._status.config(text=text)

    def _show_file(self, path, canvas):
        try:
            if '::frame' in path:
                ci = load_tiff_frame(path)
                if ci is None:
                    return
                pil = Image.fromarray(cv2.cvtColor(ci, cv2.COLOR_BGR2RGB))
            else:
                pil = Image.open(path)
            if canvas is self.canvas_orig:
                self._pil_orig = pil
            else:
                self._pil_proc = pil
            self._blit(pil, canvas)
        except Exception as e:
            self._set_status(f"Ошибка: {e}")

    def _redraw(self, canvas, pil):
        if pil:
            self._blit(pil, canvas)

    def _blit(self, pil, canvas):
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 100)
        ch = max(canvas.winfo_height(), 100)
        iw, ih = pil.size
        fit = min(cw / iw, ch / ih)
        if canvas is self.canvas_proc:
            ds = fit * self._zoom
            px, py = self._pan
        else:
            ds, px, py = fit, 0.0, 0.0
        ox = cw / 2 + px - iw * ds / 2
        oy = ch / 2 + py - ih * ds / 2
        if canvas is self.canvas_proc:
            self._proc_transform = (ds, ox, oy, iw, ih)
        sx0 = max(0, int(-ox / ds))
        sy0 = max(0, int(-oy / ds))
        sx1 = min(iw, int((cw - ox) / ds) + 1)
        sy1 = min(ih, int((ch - oy) / ds) + 1)
        if sx1 <= sx0 or sy1 <= sy0:
            canvas.delete("all")
            return
        crop = pil.crop((sx0, sy0, sx1, sy1))
        ow = max(1, round((sx1 - sx0) * ds))
        oh = max(1, round((sy1 - sy0) * ds))
        resized = crop.resize((ow, oh), Image.Resampling.BILINEAR)
        photo = ImageTk.PhotoImage(resized)
        canvas.delete("all")
        canvas.create_image(round(ox + sx0 * ds), round(oy + sy0 * ds), anchor=tk.NW, image=photo)
        canvas.image = photo

    def _update_compare_menus(self):
        """Rebuild dropdown menus with processed image names."""
        processed = [p for p in self.image_paths if self.image_data.get(p)]
        names = [self.display_names.get(p, Path(p).name) for p in processed]
        if not names:
            return
        for menu_w, var in [(self._cmp_menu_l, self._cmp_var_l),
                            (self._cmp_menu_r, self._cmp_var_r)]:
            m = menu_w['menu']
            m.delete(0, tk.END)
            for nm in names:
                m.add_command(label=nm, command=lambda v=var, n=nm: v.set(n))
        if not self._cmp_var_l.get() or self._cmp_var_l.get() not in names:
            self._cmp_var_l.set(names[0])
        if not self._cmp_var_r.get() or self._cmp_var_r.get() not in names:
            self._cmp_var_r.set(names[min(1, len(names) - 1)])

    def _draw_comparison(self):
        """Draw two processed images side by side."""
        canvas = self.canvas_compare
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 200)
        ch = max(canvas.winfo_height(), 200)

        name_l = self._cmp_var_l.get()
        name_r = self._cmp_var_r.get()
        path_l = next((p for p in self.image_paths
                       if self.display_names.get(p, Path(p).name) == name_l
                       and self.image_data.get(p)), None)
        path_r = next((p for p in self.image_paths
                       if self.display_names.get(p, Path(p).name) == name_r
                       and self.image_data.get(p)), None)

        canvas.delete("all")
        if not path_l or not path_r:
            canvas.create_text(cw // 2, ch // 2, text="Обработайте 2+ изображения",
                               fill=T.FG4, font=T.FONT_SM)
            return

        gap = 8
        half_w = (cw - gap) // 2

        for side, path, x_off in [("left", path_l, 0), ("right", path_r, half_w + gap)]:
            ann = self._get_annotated(path)
            if ann is None:
                continue
            pil = Image.fromarray(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
            iw, ih = pil.size
            fit = min(half_w / iw, ch / ih)
            nw, nh = max(1, int(iw * fit)), max(1, int(ih * fit))
            resized = pil.resize((nw, nh), Image.Resampling.BILINEAR)
            photo = ImageTk.PhotoImage(resized)
            ox = x_off + (half_w - nw) // 2
            oy = (ch - nh) // 2
            canvas.create_image(ox, oy, anchor=tk.NW, image=photo)
            # Store reference to prevent GC
            if side == "left":
                canvas._photo_l = photo
            else:
                canvas._photo_r = photo
            # Label with count
            aa, mn, _ = self._grand_total(path)
            nm = self.display_names.get(path, Path(path).name)
            if len(nm) > 25:
                nm = nm[:22] + "..."
            label = f"{nm}  |  {aa + mn} колоний"
            canvas.create_text(x_off + half_w // 2, 12, text=label,
                               fill=T.ACCENT, font=T.FONT_XS)

        # Divider line
        canvas.create_line(half_w + gap // 2, 0, half_w + gap // 2, ch,
                           fill=T.BORDER, width=2)

    # ═══════════ MANUAL EDITING ══════════════════════════════════════════

    def _c2i(self, cx, cy):
        if not self._proc_transform:
            return None
        s, ox, oy, iw, ih = self._proc_transform
        ix, iy = int((cx - ox) / s), int((cy - oy) / s)
        return (ix, iy) if 0 <= ix < iw and 0 <= iy < ih else None

    def _i2c(self, ix, iy):
        if not self._proc_transform:
            return None
        s, ox, oy, _, _ = self._proc_transform
        return (ox + ix * s, oy + iy * s)

    def _ensure_transform(self):
        if not self._pil_proc:
            return
        cw = max(self.canvas_proc.winfo_width(), 100)
        ch = max(self.canvas_proc.winfo_height(), 100)
        iw, ih = self._pil_proc.size
        fit = min(cw / iw, ch / ih)
        ds = fit * self._zoom
        px, py = self._pan
        self._proc_transform = (ds, cw / 2 + px - iw * ds / 2, ch / 2 + py - ih * ds / 2, iw, ih)

    def _push_undo(self):
        p = self.current_path
        if not p:
            return
        self._undo_stack.append((p,
            [tuple(m) for m in self.manual_marks.get(p, [])],
            [tuple(e) for e in self.excluded_auto.get(p, [])]))
        self._redo_stack.clear()
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)

    def _undo(self):
        if not self._undo_stack:
            return
        p, marks, excl = self._undo_stack.pop()
        self._redo_stack.append((p,
            [tuple(m) for m in self.manual_marks.get(p, [])],
            [tuple(e) for e in self.excluded_auto.get(p, [])]))
        self.manual_marks[p] = list(marks)
        self.excluded_auto[p] = list(excl)
        self._refresh_proc_canvas()
        self._refresh_stats_cur()
        self._refresh_results()

    def _redo(self):
        if not self._redo_stack:
            return
        p, marks, excl = self._redo_stack.pop()
        self._undo_stack.append((p,
            [tuple(m) for m in self.manual_marks.get(p, [])],
            [tuple(e) for e in self.excluded_auto.get(p, [])]))
        self.manual_marks[p] = list(marks)
        self.excluded_auto[p] = list(excl)
        self._refresh_proc_canvas()
        self._refresh_stats_cur()
        self._refresh_results()

    def _on_lmb_down(self, e):
        if self._dish_edit:
            self.canvas_proc.config(cursor='fleur')
            self._dish_drag_start(e)
        else:
            self._on_proc_click(e)

    def _on_lmb_motion(self, e):
        if self._dish_edit and self._dish_drag:
            self._dish_drag_move(e)

    def _on_lmb_up(self, e):
        if self._dish_edit and self._dish_drag:
            self._dish_drag_end(e)
            self.canvas_proc.config(cursor='hand2' if self._dish_edit else 'tcross')

    def _on_proc_click(self, e):
        if not self.current_path:
            return
        self._ensure_transform()
        pt = self._c2i(e.x, e.y)
        if pt is None:
            return
        self._push_undo()
        self.manual_marks.setdefault(self.current_path, []).append(pt)
        self._refresh_proc_canvas()
        self._refresh_stats_cur()
        self._refresh_results()

    def _on_proc_rclick(self, e):
        if not self.current_path or self._dish_edit:
            return
        self._ensure_transform()
        pt = self._c2i(e.x, e.y)
        if pt is None:
            return
        path = self.current_path
        result = self.image_data.get(path)
        cands = []
        for i, m in enumerate(self.manual_marks.get(path, [])):
            cands.append(('m', i, m))
        if result:
            excl = set(self.excluded_auto.get(path, []))
            for col in result['colonies']:
                for cx_c, cy_c in col['ws_centers']:
                    if (cx_c, cy_c) not in excl:
                        cands.append(('a', (cx_c, cy_c), (cx_c, cy_c)))
        if not cands:
            return
        dists = [(abs(c[2][0] - pt[0]) + abs(c[2][1] - pt[1]), i) for i, c in enumerate(cands)]
        _, ni = min(dists)
        kind, ref, _ = cands[ni]
        self._push_undo()
        if kind == 'm':
            self.manual_marks[path].pop(ref)
        else:
            self.excluded_auto.setdefault(path, []).append(ref)
        self._refresh_proc_canvas()
        self._refresh_stats_cur()
        self._refresh_results()

    def _clear_manual(self):
        if not self.current_path:
            return
        if self.manual_marks.get(self.current_path):
            self._push_undo()
            self.manual_marks[self.current_path] = []
            self._refresh_proc_canvas()
            self._refresh_stats_cur()
            self._refresh_results()

    def _restore_auto(self):
        if not self.current_path:
            return
        if self.excluded_auto.get(self.current_path):
            self._push_undo()
            self.excluded_auto[self.current_path] = []
            self._refresh_proc_canvas()
            self._refresh_stats_cur()
            self._refresh_results()

    # ═══════════ ZOOM / PAN ══════════════════════════════════════════════

    def _on_proc_wheel(self, e):
        if not self._pil_proc:
            return "break"
        f = C.ZOOM_FACTOR if e.delta > 0 else 1.0 / C.ZOOM_FACTOR
        oz = self._zoom
        nz = max(C.ZOOM_MIN, min(C.ZOOM_MAX, oz * f))
        if nz == oz:
            return "break"
        cw = max(self.canvas_proc.winfo_width(), 100)
        ch = max(self.canvas_proc.winfo_height(), 100)
        iw, ih = self._pil_proc.size
        fit = min(cw / iw, ch / ih)
        oox = cw / 2 + self._pan[0] - iw * fit * oz / 2
        ooy = ch / 2 + self._pan[1] - ih * fit * oz / 2
        ix = (e.x - oox) / (fit * oz)
        iy = (e.y - ooy) / (fit * oz)
        nox = e.x - ix * fit * nz
        noy = e.y - iy * fit * nz
        self._pan[0] = nox - (cw / 2 - iw * fit * nz / 2)
        self._pan[1] = noy - (ch / 2 - ih * fit * nz / 2)
        self._zoom = nz
        self._refresh_proc_canvas()
        return "break"

    def _on_proc_pan_start(self, e):
        self._pan_drag = (e.x, e.y, self._pan[0], self._pan[1])

    def _on_proc_pan_move(self, e):
        if not self._pan_drag:
            return
        sx, sy, px, py = self._pan_drag
        self._pan[0] = px + (e.x - sx)
        self._pan[1] = py + (e.y - sy)
        self._refresh_proc_canvas()

    def _reset_zoom(self):
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self._refresh_proc_canvas()

    # ═══════════ IMAGE HELPERS ═══════════════════════════════════════════

    def _get_annotated(self, path):
        r = self.image_data.get(path)
        if not r:
            return None
        return self._cache.load(f"{path}_a") if r.get('_cached') else r.get('annotated')

    def _get_clean(self, path):
        r = self.image_data.get(path)
        if not r:
            return None
        return self._cache.load(f"{path}_c") if r.get('_cached') else r.get('img_clean')

    def _make_proc(self, path):
        result = self.image_data.get(path)
        if not result:
            return None
        excl = set(self.excluded_auto.get(path, []))
        marks = self.manual_marks.get(path, [])
        annotations = self._annotations.get(path, [])
        dish_ov = self.dish_overrides.get(path)
        if not excl and not marks and not annotations:
            return self._get_annotated(path)
        clean = self._get_clean(path)
        if clean is None:
            return self._get_annotated(path)
        return calc.make_annotated_image(result, excl, marks, annotations, dish_ov, clean)

    def _refresh_proc_canvas(self):
        if not self.current_path:
            return
        cv_img = self._make_proc(self.current_path)
        if cv_img is not None:
            self._pil_proc = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            self._blit(self._pil_proc, self.canvas_proc)
        elif self._pil_proc:
            self._blit(self._pil_proc, self.canvas_proc)
        if self._dish_edit:
            self._draw_dish_handles()

    # ═══════════ DISH EDITING ════════════════════════════════════════════

    def _toggle_dish_edit(self):
        self._dish_edit = not self._dish_edit
        self._dish_btn.set_text("Границы [ON]" if self._dish_edit else "Границы")
        self.canvas_proc.config(cursor='hand2' if self._dish_edit else 'tcross')
        self._refresh_proc_canvas()

    def _reset_dish_overrides(self):
        if not self.current_path:
            return
        if self.current_path in self.dish_overrides:
            del self.dish_overrides[self.current_path]
            self._run_sync(self.current_path)
            self._refresh_results()

    def _get_dishes(self):
        if not self.current_path:
            return []
        ov = self.dish_overrides.get(self.current_path)
        if ov:
            return [tuple(d) for d in ov]
        r = self.image_data.get(self.current_path)
        return r.get('dishes', [r['dish']]) if r else []

    def _draw_dish_handles(self):
        self.canvas_proc.delete('dh')
        self._ensure_transform()
        if not self._proc_transform:
            return
        s = self._proc_transform[0]
        for di, (dcx, dcy, dr) in enumerate(self._get_dishes()):
            cc = self._i2c(dcx, dcy)
            if not cc:
                continue
            ccx, ccy = cc
            cr = dr * s
            self.canvas_proc.create_oval(ccx - cr, ccy - cr, ccx + cr, ccy + cr,
                                         outline=T.YELLOW, width=2, dash=(8, 4), tags='dh')
            hw = 7
            self.canvas_proc.create_rectangle(ccx - hw, ccy - hw, ccx + hw, ccy + hw,
                                              fill=T.YELLOW, outline=T.BG, tags='dh')
            for ex, ey in [(dcx, dcy - dr), (dcx + dr, dcy), (dcx, dcy + dr), (dcx - dr, dcy)]:
                ec = self._i2c(ex, ey)
                if ec:
                    self.canvas_proc.create_rectangle(ec[0] - 5, ec[1] - 5, ec[0] + 5, ec[1] + 5,
                                                     fill='cyan', outline=T.BG, tags='dh')

    def _dish_drag_start(self, e):
        self._ensure_transform()
        pt = self._c2i(e.x, e.y)
        if not pt:
            return
        px, py = pt
        for di, (dcx, dcy, dr) in enumerate(self._get_dishes()):
            dc = ((px - dcx)**2 + (py - dcy)**2)**0.5
            if dc <= max(15, dr * 0.08):
                self._dish_drag = dict(type='move', di=di, s=(px, py), o=(dcx, dcy, dr))
                return
            if abs(dc - dr) <= max(14, dr * 0.07):
                self._dish_drag = dict(type='resize', di=di, s=(px, py), o=(dcx, dcy, dr))
                return

    def _dish_drag_move(self, e):
        self._ensure_transform()
        pt = self._c2i(e.x, e.y)
        if not pt or not self._dish_drag:
            return
        d = self._dish_drag
        px, py = pt
        dcx, dcy, dr = d['o']
        sx, sy = d['s']
        if d['type'] == 'move':
            self._set_dish(d['di'], int(dcx + px - sx), int(dcy + py - sy), dr)
        else:
            self._set_dish(d['di'], dcx, dcy, max(20, int(((px - dcx)**2 + (py - dcy)**2)**0.5)))
        self._refresh_proc_canvas()

    def _dish_drag_end(self, _e=None):
        if not self._dish_drag:
            return
        self._dish_drag = None
        if self.current_path:
            self._run_sync(self.current_path)

    def _set_dish(self, di, cx, cy, r):
        if not self.current_path:
            return
        dishes = [[int(d[0]), int(d[1]), int(d[2])] for d in self._get_dishes()]
        while len(dishes) <= di:
            dishes.append(dishes[-1][:] if dishes else [cx, cy, r])
        dishes[di] = [int(cx), int(cy), int(r)]
        self.dish_overrides[self.current_path] = dishes

    # ═══════════ STATS ═══════════════════════════════════════════════════

    def _grand_total(self, path):
        r = self.image_data.get(path)
        excl = set(self.excluded_auto.get(path, []))
        marks = self.manual_marks.get(path, [])
        return calc.grand_total(r, excl, marks)

    def _refresh_stats_cur(self):
        if self.current_path and self.image_data.get(self.current_path):
            self._refresh_stats(self.image_data[self.current_path])

    def _refresh_stats(self, result):
        path = self.current_path
        aa, mn, en = self._grand_total(path) if path else (result['total'], 0, 0)
        singles = result['colony_count'] - result['cluster_count']
        grand = aa + mn
        hid = result.get('hidden_estimate', 0)
        dr = result.get('dish_results', [])
        dl = ""
        if len(dr) > 1:
            for i, d in enumerate(dr, 1):
                dl += f"  #{i}:       {d['total']:>6}\n"
        ppm = calc.px_per_mm(result, self.p['dish_diameter_mm'].get())
        area_str = f"{result['avg_colony_area']:>5.0f} px"
        if ppm and ppm > 0.01:
            area_str += f" ({result['avg_colony_area'] / (ppm * ppm):.3f} мм\u00b2)"
        cfu = calc.calc_cfu_ml(grand, self.p['plating_volume_ml'].get(), self.p['dilution_factor'].get())
        cfu_str = f"\nCFU/мл:     {cfu:>.0f}\n" if cfu and cfu > 0 else ""
        morph = calc.classify_morphology(result)
        morph_str = ""
        if morph:
            morph_str = (f"{'─' * 22}\nМелкие:     {morph['small']:>6}\n"
                         f"Средние:    {morph['medium']:>6}\nКрупные:    {morph['large']:>6}\n"
                         f"Круглые:    {morph['round']:>6}\nНеправ.:    {morph['irregular']:>6}\n")
        text = (f"Авто:       {result['total']:>6}\n" + dl +
                f"  одиноч.:  {singles:>6}\n  кластер.: {result['cluster_count']:>6}\n"
                f"  исключ.:  {en:>6}\nРучных:     {mn:>6}\n"
                f"{'─' * 22}\nИТОГО:      {grand:>6}\n" + cfu_str
                + (f"+скрытых:   {hid:>6}\n" if hid else "")
                + f"{'─' * 22}\nСр.площ.:   {area_str}\n"
                + (f"Этикетка:   да\n" if result.get('has_label') else "")
                + morph_str)
        self._stats_text.config(state=tk.NORMAL)
        self._stats_text.delete("1.0", tk.END)
        self._stats_text.insert(tk.END, text)
        self._stats_text.config(state=tk.DISABLED)

    def _refresh_results(self):
        lines = []
        ts = 0
        for path in self.image_paths:
            nm = self.display_names.get(path, Path(path).name)
            short = (nm[:18] + "...") if len(nm) > 19 else nm
            r = self.image_data.get(path)
            if r:
                a, m, _ = self._grand_total(path)
                c = a + m
                ts += c
                lines.append(f"{short:<19} {c:>4}")
            else:
                lines.append(f"{short:<19}  ---")
        lines.append(f"{'─' * 24}")
        lines.append(f"{'ИТОГО':<19} {ts:>4}")
        self._res_text.config(state=tk.NORMAL)
        self._res_text.delete("1.0", tk.END)
        self._res_text.insert(tk.END, "\n".join(lines))
        self._res_text.config(state=tk.DISABLED)

    # ═══════════ LEARNING ════════════════════════════════════════════════

    def _do_learning(self, path):
        if not self.p['auto_learn'].get():
            return
        result = self.image_data.get(path)
        if not result:
            return
        excl = len(self.excluded_auto.get(path, []))
        added = len(self.manual_marks.get(path, []))
        if excl == 0 and added == 0:
            return
        nt = self.learner.update(result['total'], excl, added, self.p['threshold'].get())
        if nt is not None:
            self.p['threshold'].set(nt)
            self._set_status(f"Обучение: порог -> {nt}")
            self._refresh_learn_label()

    def _apply_learned(self, silent=False):
        s = self.learner.suggestion
        if s is not None:
            self.p['threshold'].set(s)
            if not silent:
                self._set_status(f"Порог = {s}")
        elif not silent:
            self._set_status("Нет данных обучения")

    def _refresh_learn_label(self):
        s = self.learner.suggestion
        n = self.learner.samples
        self._learn_lbl.config(text=f"Модель: {n} сес. | ~{s}" if s else "Нет данных")

    def _reset_learning(self):
        self.learner.reset()
        self._refresh_learn_label()
        self._set_status("Модель сброшена")

    # ═══════════ PRESETS ═════════════════════════════════════════════════

    def _apply_preset(self, name):
        presets = {
            'default': dict(min_diam_mm=0.3, max_diam_mm=3.0, threshold=25),
            'sensitive': dict(min_diam_mm=0.2, max_diam_mm=5.0, threshold=15),
            'strict': dict(min_diam_mm=0.5, max_diam_mm=2.0, threshold=40),
            'large': dict(min_diam_mm=0.8, max_diam_mm=10.0, threshold=30),
        }
        p = presets.get(name)
        if p:
            for k, v in p.items():
                if k in self.p:
                    self.p[k].set(v)
            self._set_status(f"Пресет: {name}")

    # ═══════════ ANNOTATIONS ═════════════════════════════════════════════

    def _add_annotation(self):
        if not self.current_path:
            return
        text = simpledialog.askstring("Аннотация", "Текст:", parent=self.root)
        if not text or not text.strip():
            return
        self._annotations.setdefault(self.current_path, []).append(text.strip())
        self._refresh_proc_canvas()

    # ═══════════ EXPORT ══════════════════════════════════════════════════

    def _build_export_rows(self):
        rows = []
        for path in self.image_paths:
            result = self.image_data.get(path)
            if not result:
                continue
            aa, mn, en = self._grand_total(path)
            ppm = calc.px_per_mm(result, self.p['dish_diameter_mm'].get())
            cfu = calc.calc_cfu_ml(aa + mn, self.p['plating_volume_ml'].get(), self.p['dilution_factor'].get())
            rows.append(calc.format_result_row(
                path, result, self.display_names.get(path, Path(path).name),
                aa, mn, en, ppm, cfu,
                self.p['dilution_group'].get(), self.p['dilution_factor'].get()))
        return rows

    def _export_excel(self):
        rows = self._build_export_rows()
        if not rows:
            messagebox.showwarning("Нет данных", "Обработайте изображения.")
            return
        sp = filedialog.asksaveasfilename(defaultextension=".xlsx",
            initialfile=f"colonies_{datetime.date.today()}.xlsx", filetypes=[("Excel", "*.xlsx")])
        if sp:
            try:
                excel_export.export_excel(sp, rows)
                self._set_status(f"Excel: {sp}")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def _export_csv(self):
        rows = self._build_export_rows()
        if not rows:
            messagebox.showwarning("Нет данных", "Обработайте изображения.")
            return
        sp = filedialog.asksaveasfilename(defaultextension=".csv",
            initialfile=f"colonies_{datetime.date.today()}.csv", filetypes=[("CSV", "*.csv")])
        if sp:
            try:
                csv_export.export_csv(sp, rows)
                self._set_status(f"CSV: {sp}")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def _export_pdf(self):
        if not HAS_MPL:
            return
        rows = self._build_export_rows()
        if not rows:
            messagebox.showwarning("Нет данных", "Обработайте изображения.")
            return
        sp = filedialog.asksaveasfilename(defaultextension=".pdf",
            initialfile=f"colonies_{datetime.date.today()}.pdf", filetypes=[("PDF", "*.pdf")])
        if sp:
            try:
                results_dict = {r['path']: self.image_data.get(p)
                                for p in self.image_paths
                                for r in [calc.format_result_row(p, self.image_data[p], '', 0, 0, 0)]
                                if self.image_data.get(p)}
                pdf_export.export_pdf(sp, rows, self._get_annotated,
                                      {p: self.image_data[p] for p in self.image_paths if self.image_data.get(p)})
                self._set_status(f"PDF: {sp}")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def _export_image(self):
        if not self.current_path:
            return
        ci = self._make_proc(self.current_path)
        if ci is None:
            messagebox.showwarning("Нет", "Обработайте.")
            return
        sp = filedialog.asksaveasfilename(defaultextension=".png",
            initialfile=f"{Path(self.current_path).stem}_result.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
        if sp:
            image_export.export_image(sp, ci)
            self._set_status(f"Сохранено: {Path(sp).name}")

    # ═══════════ SESSION ═════════════════════════════════════════════════

    def _save_session(self):
        sp = filedialog.asksaveasfilename(defaultextension=".colsession",
            initialfile=f"session_{datetime.date.today()}.colsession",
            filetypes=[("Session", "*.colsession"), ("JSON", "*.json")])
        if not sp:
            return
        try:
            sess.save_session(sp, self.image_paths, self.display_names,
                              self.manual_marks, self.excluded_auto, self.dish_overrides,
                              {k: v.get() for k, v in self.p.items()}, self.current_path)
            self._set_status(f"Сессия: {Path(sp).name}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _load_session(self):
        lp = filedialog.askopenfilename(
            filetypes=[("Session", "*.colsession"), ("JSON", "*.json")])
        if not lp:
            return
        try:
            data = sess.load_session(lp)
            self.image_paths.clear()
            self.image_data.clear()
            self.manual_marks.clear()
            self.excluded_auto.clear()
            self.display_names.clear()
            self.dish_overrides.clear()
            self._cache.cleanup()
            self.listbox.delete(0, tk.END)
            self.current_path = None
            self._prev_path = None
            self._pil_orig = None
            self._pil_proc = None
            self.canvas_orig.delete("all")
            self.canvas_proc.delete("all")
            for k, v in data['params'].items():
                if k in self.p:
                    self.p[k].set(v)
            for img in data['images']:
                path = img['path']
                self.image_paths.append(path)
                self.image_data[path] = None
                self.display_names[path] = img['name']
                self.listbox.insert(tk.END, img['name'])
                if img['marks']:
                    self.manual_marks[path] = img['marks']
                if img['excl']:
                    self.excluded_auto[path] = img['excl']
                if img['dish_ov']:
                    self.dish_overrides[path] = img['dish_ov']
            cur = data.get('current_path')
            if cur and cur in self.image_paths:
                idx = self.image_paths.index(cur)
                self.listbox.selection_set(idx)
                self.listbox.see(idx)
            msg = f"Сессия: {len(self.image_paths)} изобр."
            if data['missing']:
                msg += f" Не найдено: {len(data['missing'])}"
            self._set_status(msg + "  Нажмите ОБРАБОТАТЬ ВСЕ")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    # ═══════════ ANALYSIS DIALOGS ════════════════════════════════════════

    def _show_statistics(self):
        if not HAS_MPL:
            return
        processed = [(p, r) for p in self.image_paths if (r := self.image_data.get(p))]
        if not processed:
            return
        win = tk.Toplevel(self.root)
        win.title("Статистика")
        win.geometry("1000x700")
        win.config(bg=T.BG)
        fig = Figure(figsize=(12, 8), dpi=100, facecolor=T.BG1)
        names = [self.display_names.get(p, Path(p).name)[:14] for p, _ in processed]
        counts = [sum(self._grand_total(p)[:2]) for p, _ in processed]
        ax1 = fig.add_subplot(2, 2, 1, facecolor=T.BG)
        ax1.bar(range(len(names)), counts, color=T.ACCENT)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=7, color=T.FG3)
        ax1.set_title("Колоний", color=T.FG, fontsize=10)
        ax1.tick_params(colors=T.FG3)
        ax2 = fig.add_subplot(2, 2, 2, facecolor=T.BG)
        all_a = [c['feat']['area'] for _, r in processed for c in r['colonies']]
        if all_a:
            ax2.hist(all_a, bins=min(50, max(10, len(all_a) // 5)), color=T.ACCENT, edgecolor=T.BG, alpha=0.8)
            ax2.axvline(np.median(all_a), color=T.RED, linestyle='--')
        ax2.set_title("Площади", color=T.FG, fontsize=10)
        ax2.tick_params(colors=T.FG3)
        ax3 = fig.add_subplot(2, 2, 3, facecolor=T.BG)
        sl = [r['colony_count'] - r['cluster_count'] for _, r in processed]
        cl = [r['cluster_count'] for _, r in processed]
        ax3.bar(range(len(names)), sl, label='Одиноч.', color=T.ACCENT)
        ax3.bar(range(len(names)), cl, bottom=sl, label='Кластер.', color=T.ORANGE)
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=7, color=T.FG3)
        ax3.legend(fontsize=8)
        ax3.set_title("Типы", color=T.FG, fontsize=10)
        ax3.tick_params(colors=T.FG3)
        ax4 = fig.add_subplot(2, 2, 4, facecolor=T.BG)
        ax4.bar(range(len(names)), [r['avg_colony_area'] for _, r in processed], color='#8b5cf6')
        ax4.set_xticks(range(len(names)))
        ax4.set_xticklabels(names, rotation=45, ha='right', fontsize=7, color=T.FG3)
        ax4.set_title("Ср. площадь", color=T.FG, fontsize=10)
        ax4.tick_params(colors=T.FG3)
        fig.tight_layout()
        c = FigureCanvasTkAgg(fig, master=win)
        c.draw()
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _show_heatmap(self):
        if not HAS_MPL or not self.current_path:
            return
        result = self.image_data.get(self.current_path)
        if not result or not result['colonies']:
            return
        win = tk.Toplevel(self.root)
        win.title("Heatmap")
        win.geometry("700x700")
        win.config(bg=T.BG)
        fig = Figure(figsize=(7, 7), dpi=100, facecolor=T.BG1)
        ax = fig.add_subplot(111, facecolor=T.BG)
        xs = [c['center'][0] for c in result['colonies']]
        ys = [c['center'][1] for c in result['colonies']]
        ann = self._get_annotated(self.current_path)
        if ann is not None:
            ax.imshow(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), alpha=0.4)
        if xs and ys:
            ax.hexbin(xs, ys, gridsize=20, cmap='YlOrRd', alpha=0.6, mincnt=1)
            ax.scatter(xs, ys, c=T.ACCENT, s=8, alpha=0.5, edgecolors='none')
        ax.set_title("Плотность колоний", color=T.FG, fontsize=11)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.tick_params(colors=T.FG3)
        fig.tight_layout()
        c = FigureCanvasTkAgg(fig, master=win)
        c.draw()
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _show_reproducibility(self):
        if not HAS_MPL or not self.current_path:
            return
        if not self.image_data.get(self.current_path):
            return
        path = self.current_path
        base_thresh = self.p['threshold'].get()
        results = []
        for delta in [-10, -5, 0, 5, 10]:
            t = max(5, min(100, base_thresh + delta))
            params = self._get_params()
            params['threshold'] = t
            try:
                r = self._process_image(path, params)
                results.append((t, r['total']))
            except Exception:
                continue
        if not results:
            return
        win = tk.Toplevel(self.root)
        win.title("Воспроизводимость")
        win.geometry("700x400")
        win.config(bg=T.BG)
        fig = Figure(figsize=(7, 4), dpi=100, facecolor=T.BG1)
        ax = fig.add_subplot(111, facecolor=T.BG)
        thresholds = [r[0] for r in results]
        totals = [r[1] for r in results]
        ax.plot(thresholds, totals, 'o-', color=T.ACCENT, markersize=10, linewidth=2)
        ax.axvline(base_thresh, color=T.RED, linestyle='--', label=f"Текущий: {base_thresh}")
        for t, tot in zip(thresholds, totals):
            ax.annotate(str(tot), (t, tot), textcoords="offset points",
                        xytext=(0, 12), ha='center', color=T.FG, fontsize=10)
        ax.set_xlabel("Порог", color=T.FG3)
        ax.set_ylabel("Колоний", color=T.FG3)
        ax.set_title("Чувствительность к порогу", color=T.FG)
        ax.tick_params(colors=T.FG3)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        c = FigureCanvasTkAgg(fig, master=win)
        c.draw()
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
