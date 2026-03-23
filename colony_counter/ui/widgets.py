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


class ToolTip:
    """Delayed tooltip on hover for any widget."""

    def __init__(self, widget, text: str, delay: int = 500):
        self._widget = widget
        self._text = text
        self._delay = delay
        self._tip = None
        self._after_id = None
        widget.bind('<Enter>', self._schedule, add='+')
        widget.bind('<Leave>', self._cancel, add='+')
        widget.bind('<Button>', self._cancel, add='+')

    def _schedule(self, _e=None):
        self._cancel()
        self._after_id = self._widget.after(self._delay, self._show)

    def _cancel(self, _e=None):
        if self._after_id:
            self._widget.after_cancel(self._after_id)
            self._after_id = None
        self._hide()

    def _show(self):
        if self._tip:
            return
        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self._text, bg='#1e1e2e', fg='#cdd6f4',
                 font=('Consolas', 9), padx=8, pady=4,
                 relief='solid', borderwidth=1).pack()

    def _hide(self):
        if self._tip:
            self._tip.destroy()
            self._tip = None
