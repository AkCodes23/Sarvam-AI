"""Human listening-audit harness. Cannot be faked by an AI: a person listens and fills it.

  python scripts/human_audit.py sample   # draw 20 en + 20 te clips -> CSV + audit.html
  python scripts/human_audit.py score    # read the filled CSV -> audit statistics table

The audit page (data/review_app/audit.html) plays each clip and asks three yes/no
questions: transcript correct, emotion correct, audio quality ok. Export writes
human_audit.csv; `score` turns it into the per-language audit table for the report.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

from ttsds.config import MANIFEST_DIR, PROJECT_ROOT, REVIEW_DIR
from ttsds.build_dataset import FINAL_SELECTION

CSV_PATH = MANIFEST_DIR / "human_audit.csv"
N_PER_LANG = 20


def sample():
    recs = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))
    rows = []
    for cfg, rs in recs.items():
        ordered = sorted(rs, key=lambda r: r["audio"])
        k = max(1, len(ordered) // N_PER_LANG)
        for r in ordered[::k][:N_PER_LANG]:
            rows.append(r)
    # blank audit sheet for offline filling
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "language", "emotion", "transcript",
                    "transcript_correct", "emotion_correct", "audio_ok", "notes"])
        for r in rows:
            sid = os.path.splitext(os.path.basename(r["audio"]))[0]
            w.writerow([sid, r["language"], r["emotion"], r["text"], "", "", "", ""])
    # interactive audit page
    items = []
    for r in rows:
        sid = os.path.splitext(os.path.basename(r["audio"]))[0]
        wav = os.path.relpath(PROJECT_ROOT / r["audio"], REVIEW_DIR).replace("\\", "/")
        items.append({"id": sid, "lang": r["language"], "emotion": r["emotion"],
                      "text": r["text"], "wav": wav})
    (REVIEW_DIR / "audit.html").write_text(_HTML.replace("__DATA__", json.dumps(items, ensure_ascii=False)),
                                           encoding="utf-8")
    print(f"wrote {CSV_PATH} ({len(rows)} clips) and {REVIEW_DIR/'audit.html'}")
    print("Open audit.html, listen, answer y/n, Export -> save as human_audit.csv, then run: score")


def score():
    if not CSV_PATH.exists():
        print(f"{CSV_PATH} not found. Fill the audit first."); return
    by = defaultdict(lambda: defaultdict(list))
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k in ("transcript_correct", "emotion_correct", "audio_ok"):
                v = (row.get(k) or "").strip().lower()
                if v in ("y", "yes", "1", "true"):
                    by[row["language"]][k].append(1)
                elif v in ("n", "no", "0", "false"):
                    by[row["language"]][k].append(0)
    print(f"{'metric':<20}{'English':>10}{'Telugu':>10}")
    for k in ("transcript_correct", "emotion_correct", "audio_ok"):
        def pct(l):
            v = by.get(l, {}).get(k, [])
            return f"{round(sum(v)/len(v)*100)}% (n={len(v)})" if v else "pending"
        print(f"{k:<20}{pct('en'):>10}{pct('te'):>10}")


_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Human audit</title>
<style>body{font:14px system-ui;margin:0;background:#0f1115;color:#e6e6e6}
header{position:sticky;top:0;background:#161922;padding:10px 14px;border-bottom:1px solid #2a2f3a}
.card{margin:10px 14px;padding:10px 12px;background:#161922;border:1px solid #2a2f3a;border-radius:8px}
audio{width:320px}button{background:#0f1115;color:#e6e6e6;border:1px solid #333a48;border-radius:6px;padding:4px 9px;cursor:pointer;margin-right:6px}
.q{margin-top:6px}.y.on{background:#16a34a;border-color:#16a34a}.n.on{background:#dc2626;border-color:#dc2626}
.lang{font-size:11px;background:#334155;padding:1px 6px;border-radius:4px;margin-right:6px}.meta{color:#9aa4b2;font-size:12px}</style></head>
<body><header><b>Human listening audit</b> <span id="c" class="meta"></span>
<button onclick="exp()">Export human_audit.csv</button>
<span class="meta">listen, answer the 3 questions, export, save to data/manifests/human_audit.csv, run: python scripts/human_audit.py score</span></header>
<div id="g"></div><script>
const D=__DATA__,S={};D.forEach(d=>S[d.id]={transcript_correct:"",emotion_correct:"",audio_ok:""});
const QS=[["transcript_correct","Transcript correct?"],["emotion_correct","Emotion correct?"],["audio_ok","Audio quality OK?"]];
function set(id,q,v){S[id][q]=v;render()}
function render(){const g=document.getElementById('g');g.innerHTML='';let done=0;
 D.forEach(d=>{const s=S[d.id];if(s.transcript_correct&&s.emotion_correct&&s.audio_ok)done++;
  const qhtml=QS.map(([k,lbl])=>`<div class="q">${lbl}
    <button class="y ${s[k]==='y'?'on':''}" onclick="set('${d.id}','${k}','y')">yes</button>
    <button class="n ${s[k]==='n'?'on':''}" onclick="set('${d.id}','${k}','n')">no</button></div>`).join('');
  g.insertAdjacentHTML('beforeend',`<div class="card"><span class="lang">${d.lang}</span><b>${d.id}</b> <span class="meta">[${d.emotion}]</span>
    <div><audio controls preload="none" src="${d.wav}"></audio></div>
    <div class="meta" style="margin:4px 0">${d.text}</div>${qhtml}</div>`)});
 document.getElementById('c').textContent=`(${done}/${D.length} done)`}
function exp(){let o="id,language,emotion,transcript,transcript_correct,emotion_correct,audio_ok,notes\n";
 const esc=s=>'"'+String(s==null?'':s).replace(/"/g,'""')+'"';
 D.forEach(d=>{const s=S[d.id];o+=[d.id,d.lang,d.emotion,esc(d.text),s.transcript_correct,s.emotion_correct,s.audio_ok,''].join(',')+'\n'});
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([o],{type:'text/csv'}));a.download='human_audit.csv';a.click()}
render();</script></body></html>"""


if __name__ == "__main__":
    (score if len(sys.argv) > 1 and sys.argv[1] == "score" else sample)()
