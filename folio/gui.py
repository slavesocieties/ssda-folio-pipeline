"""Drag-and-drop desktop app for the folio pipeline.

    python -m folio.gui          (or, once installed:  folio-gui)

Drop SSDA scans (images or folders) onto the window; the tool writes upright,
single-folio crops to an output folder and shows them as thumbnails, with
review-flagged pages highlighted. Pure-stdlib tkinter UI; uses tkinterdnd2 for
native drag-and-drop when available, and a file-picker fallback otherwise.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

from . import process as P

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _HAS_DND = True
except Exception:  # pragma: no cover - optional
    _HAS_DND = False

BG = "#1e1f24"; CARD = "#2a2c33"; ACCENT = "#4a8cff"; FG = "#e7e9ee"
MUTED = "#9aa0ab"; REVIEW = "#ff5a5a"; OK = "#46c46a"
THUMB = 150


class FolioGUI:
    def __init__(self, root):
        self.root = root
        root.title("Folio Processor — SSDA page pre-processing")
        root.geometry("960x680")
        root.configure(bg=BG)
        self.inputs: list[Path] = []
        self.out_dir = Path.home() / "folio_out"
        self._out_user_set = False
        self._thumbs = []           # keep PhotoImage refs alive
        self._q: queue.Queue = queue.Queue()
        self._busy = False
        self._build()

    # ----------------------------------------------------------------- layout
    def _build(self):
        head = tk.Frame(self.root, bg=BG)
        head.pack(fill="x", padx=18, pady=(16, 6))
        tk.Label(head, text="Folio Processor", bg=BG, fg=FG,
                 font=("Segoe UI Semibold", 20)).pack(anchor="w")
        tk.Label(head, text="Drop archival scans below → upright, single-folio, cropped pages",
                 bg=BG, fg=MUTED, font=("Segoe UI", 11)).pack(anchor="w")

        # drop zone
        self.drop = tk.Label(
            self.root, text="⬇  Drag images or a folder here", bg=CARD, fg=MUTED,
            font=("Segoe UI", 14), height=4, relief="ridge", bd=2)
        self.drop.pack(fill="x", padx=18, pady=10)
        if _HAS_DND:
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind("<<Drop>>", self._on_drop)
        else:
            self.drop.configure(text="Click “Add images” or “Add folder” below")

        # controls row
        row = tk.Frame(self.root, bg=BG)
        row.pack(fill="x", padx=18)
        self._btn(row, "Add images…", self._pick_files).pack(side="left")
        self._btn(row, "Add folder…", self._pick_folder).pack(side="left", padx=(8, 0))
        self._btn(row, "Clear", self._clear).pack(side="left", padx=(8, 0))
        self.process_btn = self._btn(row, "Process", self._start, primary=True)
        self.process_btn.pack(side="right")
        self.process_btn.configure(state="disabled")

        # output row
        orow = tk.Frame(self.root, bg=BG)
        orow.pack(fill="x", padx=18, pady=(10, 4))
        tk.Label(orow, text="Output →", bg=BG, fg=MUTED, font=("Segoe UI", 10)).pack(side="left")
        self.out_lbl = tk.Label(orow, text=str(self.out_dir), bg=BG, fg=FG, font=("Segoe UI", 10))
        self.out_lbl.pack(side="left", padx=(6, 0))
        self._btn(orow, "Change…", self._pick_out).pack(side="left", padx=(8, 0))
        self.open_btn = self._btn(orow, "Open output folder", self._open_out)
        self.open_btn.pack(side="right")
        self.open_btn.configure(state="disabled")

        # progress + status
        self.prog = ttk.Progressbar(self.root, mode="determinate")
        self.prog.pack(fill="x", padx=18, pady=(8, 2))
        self.status = tk.Label(self.root, text="Ready.", bg=BG, fg=MUTED,
                               font=("Segoe UI", 10), anchor="w")
        self.status.pack(fill="x", padx=18)

        # results (scrollable thumbnails)
        wrap = tk.Frame(self.root, bg=BG)
        wrap.pack(fill="both", expand=True, padx=14, pady=10)
        self.canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.canvas.yview)
        self.grid = tk.Frame(self.canvas, bg=BG)
        self.grid.bind("<Configure>",
                       lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.grid, anchor="nw")
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.canvas.bind_all("<MouseWheel>",
                             lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))

    def _btn(self, parent, text, cmd, primary=False):
        return tk.Button(parent, text=text, command=cmd, relief="flat",
                         bg=(ACCENT if primary else CARD), fg="white" if primary else FG,
                         activebackground=ACCENT, activeforeground="white",
                         font=("Segoe UI", 10), padx=14, pady=6, bd=0, cursor="hand2")

    # ------------------------------------------------------------------ inputs
    def _on_drop(self, event):
        for item in self.root.tk.splitlist(event.data):
            self._add(Path(item))
        self._refresh_inputs()

    def _pick_files(self):
        fs = filedialog.askopenfilenames(
            title="Choose scans",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff"), ("All", "*.*")])
        for f in fs:
            self._add(Path(f))
        self._refresh_inputs()

    def _pick_folder(self):
        d = filedialog.askdirectory(title="Choose a folder of scans")
        if d:
            self._add(Path(d))
        self._refresh_inputs()

    def _pick_out(self):
        d = filedialog.askdirectory(title="Choose output folder")
        if d:
            self.out_dir = Path(d); self._out_user_set = True
            self.out_lbl.configure(text=str(self.out_dir))

    def _add(self, path: Path):
        if path not in self.inputs and (path.is_dir() or
                                        path.suffix.lower() in P.IMAGE_EXTS):
            self.inputs.append(path)

    def _refresh_inputs(self):
        n = len(P.list_images(self.inputs))
        self.drop.configure(text=f"{n} image(s) ready  —  drop more, or press Process",
                            fg=FG if n else MUTED)
        self.process_btn.configure(state=("normal" if n and not self._busy else "disabled"))
        if self.inputs and not self._out_user_set:
            base = self.inputs[0] if self.inputs[0].is_dir() else self.inputs[0].parent
            self.out_dir = base / "folio_out"
            self.out_lbl.configure(text=str(self.out_dir))

    def _clear(self):
        if self._busy:
            return
        self.inputs.clear(); self._thumbs.clear()
        for w in self.grid.winfo_children():
            w.destroy()
        self.prog["value"] = 0
        self.status.configure(text="Ready.")
        self.open_btn.configure(state="disabled")
        self._refresh_inputs()

    # --------------------------------------------------------------- processing
    def _start(self):
        files = P.list_images(self.inputs)
        if not files or self._busy:
            return
        self._busy = True
        self.process_btn.configure(state="disabled")
        for w in self.grid.winfo_children():
            w.destroy()
        self._thumbs.clear()
        self.prog["value"] = 0
        threading.Thread(target=self._work, args=(files, self.out_dir), daemon=True).start()
        self.root.after(80, self._poll)

    def _work(self, files, out):
        def on_start(n, mode, device):
            self._q.put(("start", n, mode, device))

        def on_item(i, n, name, res):
            rev = bool(res and any(f.needs_review for f in res.folios))
            err = (res is None) or bool(getattr(res, "error", None) and not res.folios)
            self._q.put(("item", i, n, name, rev, err))
        try:
            stats, mode = P.run_local(files, out, on_start=on_start, on_item=on_item)
            self._q.put(("done", stats, str(out)))
        except Exception as e:  # surface failures in the UI
            self._q.put(("error", f"{type(e).__name__}: {e}"))

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg[0]
                if kind == "start":
                    _, n, mode, device = msg
                    self.prog.configure(maximum=n, value=0)
                    self.status.configure(text=f"{mode}  ·  device={device}  ·  {n} image(s)…")
                elif kind == "item":
                    _, i, n, name, rev, err = msg
                    self.prog["value"] = i
                    tag = "  ⚠ review" if rev else ("  ✗ error" if err else "")
                    self.status.configure(text=f"[{i}/{n}] {name}{tag}")
                elif kind == "done":
                    _, stats, out = msg
                    self._finish(stats, out)
                    return
                elif kind == "error":
                    self._busy = False
                    self._refresh_inputs()
                    messagebox.showerror("Folio Processor", msg[1])
                    self.status.configure(text="Failed.")
                    return
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def _finish(self, stats, out):
        self._busy = False
        self.open_btn.configure(state="normal")
        self._refresh_inputs()
        self.status.configure(
            text=f"Done: {stats.folios} crop(s) from {stats.images} image(s)  ·  "
                 f"{stats.review} flagged for review  ·  {stats.errors} error(s)")
        self._show_thumbs(Path(out))

    def _show_thumbs(self, out: Path):
        folios = sorted((out / "folios").glob("*.jpg"))
        review = {p.name for p in (out / "review").glob("*.jpg")}
        cols = 5
        shown = folios[:80]
        for idx, p in enumerate(shown):
            flagged = p.name in review
            cell = tk.Frame(self.grid, bg=CARD, bd=2,
                            highlightthickness=2,
                            highlightbackground=(REVIEW if flagged else CARD))
            r, c = divmod(idx, cols)
            cell.grid(row=r, column=c, padx=6, pady=6, sticky="n")
            try:
                im = Image.open(p); im.thumbnail((THUMB, THUMB))
                ph = ImageTk.PhotoImage(im); self._thumbs.append(ph)
                lbl = tk.Label(cell, image=ph, bg=CARD, cursor="hand2")
                lbl.pack()
                lbl.bind("<Button-1>", lambda e, path=p: self._open(path))
            except Exception:
                tk.Label(cell, text="(preview\nfailed)", bg=CARD, fg=MUTED).pack()
            cap = ("⚠ " if flagged else "") + p.name
            tk.Label(cell, text=cap, bg=CARD, fg=(REVIEW if flagged else FG),
                     font=("Segoe UI", 8), wraplength=THUMB).pack(pady=(2, 4))
        if len(folios) > len(shown):
            tk.Label(self.grid, text=f"… and {len(folios)-len(shown)} more in the output folder",
                     bg=BG, fg=MUTED, font=("Segoe UI", 9)).grid(
                         row=(len(shown) // cols) + 1, column=0, columnspan=cols, pady=8)

    # --------------------------------------------------------------------- open
    def _open(self, path: Path):
        try:
            os.startfile(str(path))  # Windows
        except AttributeError:
            import subprocess
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(path)])

    def _open_out(self):
        self._open(self.out_dir)


def main():
    root = TkinterDnD.Tk() if _HAS_DND else tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    FolioGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
