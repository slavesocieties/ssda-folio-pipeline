"""folio-web — a tiny local web app for the folio crop pipeline.

Drop archival scans in a browser, get clean single-folio crops back (plus a zip).
No LLM, no command line: `folio-web` starts a local server and opens your browser.
Self-contained (one file, one template) so a non-technical user can just run it.

    folio-web                 # opens http://127.0.0.1:8000
    folio-web --port 9000 --host 0.0.0.0   # serve on the network / a server

Only extra dependency is Flask (see requirements-web.txt). The heavy pipeline
models load once at startup; each upload is processed with approach B (default).
"""
from __future__ import annotations

import argparse
import io
import os
import tempfile
import threading
import time
import uuid
import webbrowser
import zipfile
from pathlib import Path

import cv2
import numpy as np

try:
    from flask import Flask, request, send_file, send_from_directory, abort, Response
except ImportError:  # pragma: no cover
    raise SystemExit("Flask is required for the web app:  pip install -r requirements-web.txt")

from .process import make_config, build_pipeline, find_legacy_weights

app = Flask(__name__)
_WORK = Path(tempfile.gettempdir()) / "folio_web"
_WORK.mkdir(exist_ok=True)
_PIPE = None
_MODE = ""
_PIPE_LOCK = threading.Lock()


def _pipeline():
    """Build the pipeline once, thread-safely. Preloaded at startup (see main) so the
    first request never races on a cold model load."""
    global _PIPE, _MODE
    if _PIPE is None:
        with _PIPE_LOCK:
            if _PIPE is None:                     # double-checked: only one thread builds
                cfg = make_config()
                lw = find_legacy_weights(None, str(Path(__file__).resolve().parent.parent))
                _PIPE, _MODE = build_pipeline(cfg, lw)
    return _PIPE


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Folio Processor</title><style>
:root{color-scheme:light dark}
body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.5}
h1{margin-bottom:.2rem}.sub{color:#888;margin-top:0}
#drop{border:2px dashed #999;border-radius:12px;padding:3rem;text-align:center;cursor:pointer;transition:.15s}
#drop.over{border-color:#3b82f6;background:rgba(59,130,246,.08)}
button{background:#3b82f6;color:#fff;border:0;border-radius:8px;padding:.6rem 1.2rem;font-size:1rem;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-top:1.5rem}
.card{border:1px solid #ccc3;border-radius:8px;overflow:hidden}.card img{width:100%;display:block;background:#eee}
.card .cap{font-size:.75rem;padding:.3rem .4rem;color:#666}.flag{color:#d97706}
.bar{display:flex;gap:1rem;align-items:center;margin:1rem 0;flex-wrap:wrap}
a.dl{display:inline-block}.mode{font-size:.75rem;color:#999}
#skipped{margin-top:1rem}#skipped .s{background:rgba(217,119,6,.1);border:1px solid rgba(217,119,6,.4);
 border-radius:8px;padding:.5rem .8rem;margin:.4rem 0;font-size:.85rem}#skipped b{color:#d97706}
</style></head><body>
<h1>Folio Processor</h1>
<p class=sub>Drop archival page scans &rarr; clean single-folio, upright, cropped images.</p>
<div id=drop>Drop images here, or click to choose<input id=file type=file accept="image/*" multiple hidden></div>
<div class=bar><button id=go disabled>Process</button><span id=status></span></div>
<div class=bar id=actions style=display:none><a class=dl id=zip href=#><button>Download all (.zip)</button></a>
<span class=mode id=mode></span></div>
<div id=skipped></div>
<div class=grid id=grid></div>
<script>
const drop=document.getElementById('drop'),file=document.getElementById('file'),go=document.getElementById('go'),
 status=document.getElementById('status'),grid=document.getElementById('grid'),actions=document.getElementById('actions'),
 zip=document.getElementById('zip'),mode=document.getElementById('mode'),skipped=document.getElementById('skipped');let files=[];
drop.onclick=()=>file.click();
file.onchange=()=>{files=[...file.files];status.textContent=files.length+' file(s) selected';go.disabled=!files.length};
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',ev=>{files=[...ev.dataTransfer.files].filter(f=>f.type.startsWith('image'));
 status.textContent=files.length+' file(s) selected';go.disabled=!files.length});
go.onclick=async()=>{go.disabled=true;grid.innerHTML='';skipped.innerHTML='';actions.style.display='none';
 status.textContent='Processing '+files.length+' image(s)… (first run loads the models, ~20s)';
 const fd=new FormData();files.forEach(f=>fd.append('images',f));
 let r;try{r=await fetch('/process',{method:'POST',body:fd})}catch(e){status.textContent='Error: '+e;go.disabled=false;return}
 if(!r.ok){status.textContent='Error: '+(await r.text());go.disabled=false;return}
 const d=await r.json();
 status.textContent=d.folios.length+' folio crop(s) from '+d.images+' image(s)'+(d.skipped&&d.skipped.length?', '+d.skipped.length+' skipped':'');
 mode.textContent=d.mode;
 if(d.folios.length){zip.href='/zip/'+d.session;actions.style.display='flex'}
 (d.skipped||[]).forEach(s=>{const e=document.createElement('div');e.className='s';
  e.innerHTML='<b>&#9888; '+s.name+'</b> — '+s.reason;skipped.appendChild(e)});
 if(!d.folios.length&&(!d.skipped||!d.skipped.length))status.textContent='No crops produced. Try a photo of a book/document page (JPG/PNG).';
 d.folios.forEach(f=>{const c=document.createElement('div');c.className='card';
  c.innerHTML='<img loading=lazy src="/file/'+d.session+'/'+encodeURIComponent(f.name)+'">'+
   '<div class=cap>'+f.name+(f.review?' <span class=flag>&#9873; review</span>':'')+'</div>';grid.appendChild(c)});
 go.disabled=false};
</script></body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/process", methods=["POST"])
def process():
    ups = request.files.getlist("images")
    if not ups:
        return ("No images uploaded", 400)
    pipe = _pipeline()
    sid = uuid.uuid4().hex[:12]
    outdir = _WORK / sid
    outdir.mkdir(parents=True, exist_ok=True)
    folios = []
    skipped = []                     # files that produced no crop, with the reason
    n_img = 0
    for up in ups:
        data = np.frombuffer(up.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            skipped.append({"name": up.filename,
                            "reason": "could not read this file as an image "
                                      "(unsupported format — use JPG, PNG or TIFF; "
                                      "phone HEIC and PDF are not supported)"})
            continue
        n_img += 1
        stem = Path(up.filename).stem
        try:
            res = pipe.process_image(up.filename, img)
        except Exception as e:  # keep the app alive on a bad image
            app.logger.warning("failed on %s: %s", up.filename, e)
            skipped.append({"name": up.filename, "reason": f"processing error: {type(e).__name__}: {e}"})
            continue
        before = len(folios)
        for f in res.folios:
            if f.crop is None:
                continue
            name = f"{stem}{('-' + f.label) if f.label else ''}.jpg"
            cv2.imwrite(str(outdir / name), f.crop)
            folios.append({"name": name, "review": bool(getattr(f, "needs_review", False))})
        if len(folios) == before:
            pc = getattr(res.page_count, "value", "?")
            reason = res.error or (f"no folio detected (page count: {pc}) — "
                                   "is this a photo of a book/document page?")
            skipped.append({"name": up.filename, "reason": reason})
    return {"session": sid, "images": n_img, "mode": _MODE,
            "folios": folios, "skipped": skipped}


@app.route("/file/<sid>/<path:name>")
def file(sid, name):
    d = _WORK / sid
    if not d.is_dir() or "/" in name or ".." in name:
        abort(404)
    return send_from_directory(d, name)


@app.route("/zip/<sid>")
def zip_all(sid):
    d = _WORK / sid
    if not d.is_dir():
        abort(404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(d.glob("*.jpg")):
            z.write(p, arcname=p.name)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name="folio_crops.zip")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Local web app for the folio crop pipeline.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    args = ap.parse_args(argv)
    url = f"http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{args.port}"
    print(f"Folio Processor web app -> {url}")
    print("Loading models…  (one-time, ~20s — please wait for 'ready' before uploading)")
    # Preload the pipeline BEFORE serving, so the very first upload can't fail on a cold
    # model load (which is what made an early run produce no crops). Also warms EasyOCR's
    # own model download if present, so nothing downloads mid-request.
    try:
        pipe = _pipeline()
        ocr = getattr(pipe, "ocr_verifier", None)
        if ocr is not None:
            _ = ocr.available          # build EasyOCR reader now (downloads its models once)
        print(f"ready: {_MODE}")
    except Exception as e:
        print(f"[!] model load failed: {type(e).__name__}: {e}\n"
              "    Check that the weights are present (tools/fetch_weights.py) and re-run.")
        return
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
