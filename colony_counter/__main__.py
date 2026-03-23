"""Entry point: python -m colony_counter"""
import tkinter as tk
from colony_counter.app import App


def main():
    root = tk.Tk()
    try:
        root.tk.call('tk', 'scaling', 1.25)
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
