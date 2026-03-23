"""Custom dark-themed widgets."""
import tkinter as tk
from colony_counter.ui.theme import T


class DarkButton(tk.Frame):
    """Flat button with hover effect."""

    def __init__(self, parent, text, command=None, variant='primary', small=False, **kw):
        super().__init__(parent, **kw)
        if T.is_dark():
            colors = {
                'primary': (T.ACCENT_BTN, T.ACCENT_HOV, '#ffffff'),
                'secondary': (T.BG3, '#52525b', T.FG2),
                'danger': (T.RED_DIM, '#991b1b', '#fca5a5'),
                'ghost': (T.BG1, T.BG2, T.FG3),
            }
        else:
            colors = {
                'primary': (T.ACCENT_BTN, T.ACCENT_HOV, '#ffffff'),
                'secondary': (T.BG2, T.BG3, T.FG2),
                'danger': (T.RED_DIM, '#fca5a5', T.RED),
                'ghost': (T.BG1, T.BG2, T.FG3),
            }
        self._bg, self._hover, self._fg = colors.get(variant, colors['primary'])
        self._cmd = command
        pad_x = 6 if small else 10
        pad_y = 2 if small else 5
        font = T.FONT_XS if small else T.FONT_SM

        self.config(bg=self._bg, cursor='hand2')
        self.lbl = tk.Label(self, text=text, bg=self._bg, fg=self._fg,
                            font=font, padx=pad_x, pady=pad_y)
        self.lbl.pack()
        for w in (self, self.lbl):
            w.bind('<Enter>', self._on_enter)
            w.bind('<Leave>', self._on_leave)
            w.bind('<Button-1>', self._on_click)

    def _on_enter(self, _e):
        self.config(bg=self._hover)
        self.lbl.config(bg=self._hover)

    def _on_leave(self, _e):
        self.config(bg=self._bg)
        self.lbl.config(bg=self._bg)

    def _on_click(self, _e):
        if self._cmd:
            self._cmd()

    def set_text(self, text):
        self.lbl.config(text=text)


class DarkCheck(tk.Frame):
    """Minimal checkbox with emerald accent."""

    def __init__(self, parent, text, variable, **kw):
        super().__init__(parent, bg=T.BG1, **kw)
        self._var = variable
        self._box = tk.Canvas(self, width=14, height=14, bg=T.BG2,
                              highlightthickness=1, highlightbackground=T.BORDER,
                              cursor='hand2')
        self._box.pack(side=tk.LEFT, padx=(0, 6))
        self._lbl = tk.Label(self, text=text, bg=T.BG1, fg=T.FG3,
                             font=T.FONT_SM, cursor='hand2')
        self._lbl.pack(side=tk.LEFT)
        self._box.bind('<Button-1>', self._toggle)
        self._lbl.bind('<Button-1>', self._toggle)
        self._var.trace_add('write', lambda *_: self._draw())
        self._draw()

    def _toggle(self, _e=None):
        self._var.set(not self._var.get())

    def _draw(self):
        self._box.delete('check')
        if self._var.get():
            self._box.config(highlightbackground=T.ACCENT)
            self._box.create_line(3, 7, 6, 11, fill=T.ACCENT, width=2, tags='check')
            self._box.create_line(6, 11, 12, 3, fill=T.ACCENT, width=2, tags='check')
        else:
            self._box.config(highlightbackground=T.BORDER)


class DarkSlider(tk.Frame):
    """Labeled slider with value display."""

    def __init__(self, parent, label, variable, from_, to, resolution=1, **kw):
        super().__init__(parent, bg=T.BG1, **kw)
        self._var = variable
        self._res = resolution
        tk.Label(self, text=label, bg=T.BG1, fg=T.FG3,
                 font=T.FONT_XS, anchor='w').pack(fill=tk.X)
        row = tk.Frame(self, bg=T.BG1)
        row.pack(fill=tk.X, pady=(2, 0))
        tk.Scale(row, from_=from_, to=to, orient=tk.HORIZONTAL,
                 variable=variable, resolution=resolution,
                 bg=T.BG2, fg=T.FG, troughcolor=T.BG,
                 highlightthickness=0, sliderrelief='flat',
                 activebackground=T.ACCENT, font=T.FONT_XS,
                 showvalue=False, bd=0, length=160).pack(
                     side=tk.LEFT, fill=tk.X, expand=True)
        self._val_lbl = tk.Label(row, text='', bg=T.BG1, fg=T.ACCENT,
                                 font=T.FONT_SM, width=6, anchor='e')
        self._val_lbl.pack(side=tk.RIGHT, padx=(4, 0))
        variable.trace_add('write', self._update_val)
        self._update_val()

    def _update_val(self, *_):
        v = self._var.get()
        # Show decimals for float values, integers for int values
        if self._res < 1:
            self._val_lbl.config(text=f"{v:.1f}")
        else:
            self._val_lbl.config(text=str(int(v)))


class DarkSection(tk.Frame):
    """Bordered section with uppercase header."""

    def __init__(self, parent, title, **kw):
        super().__init__(parent, bg=T.BG1, highlightthickness=1,
                         highlightbackground=T.BORDER, **kw)
        hdr = tk.Frame(self, bg=T.BG1)
        hdr.pack(fill=tk.X, padx=8, pady=(6, 2))
        tk.Label(hdr, text=title.upper(), bg=T.BG1, fg=T.FG4,
                 font=T.FONT_XS).pack(anchor=tk.W)
        self.body = tk.Frame(self, bg=T.BG1)
        self.body.pack(fill=tk.X, padx=8, pady=(0, 8))
