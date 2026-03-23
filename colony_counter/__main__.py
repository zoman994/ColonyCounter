"""Entry point: python -m colony_counter"""
import sys
import ctypes
import tkinter as tk

from colony_counter.app import App


def _enable_dpi_awareness():
    """Make Windows render the app at native resolution instead of upscaling."""
    if sys.platform != 'win32':
        return
    try:
        # Windows 10 1703+ — Per-Monitor V2 (best quality)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            # Windows 8.1+ — System DPI aware
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                # Windows Vista+ fallback
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def main():
    _enable_dpi_awareness()
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
