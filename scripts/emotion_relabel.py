"""#1 Human emotion-relabel harness (a person picks the true emotion by ear).

  python scripts/emotion_relabel.py sample   # stratified clips -> CSV + emotion_relabel.html
  python scripts/emotion_relabel.py score    # filled CSV -> human-vs-LLM agreement + confusion

Stratified across all 8 emotion tags per language (so every label, especially the
weakly-corroborated `calm`, gets reviewed). The page shows the CURRENT tag and the
audio; the reviewer confirms or corrects it. A model cannot fill this.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict

from ttsds.config import MANIFEST_DIR, PROJECT_ROOT, REVIEW_DIR
from ttsds.build_dataset import FINAL_SELECTION

CSV_PATH = MANIFEST_DIR / "emotion_relabel.csv"
EMOTIONS = ["neutral", "happy", "sad", "angry", "excited", "calm", "fearful", "surprised"]
PER_EMOTION_PER_LANG = 4


def _sample_rows(recs: dict) -> list[dict]:
    rows = []
    for cfg, rs in recs.items():
        by_emo: dict[str, list] = defaultdict(list)
        for r in sorted(rs, key=lambda r: r["audio"]):
            by_emo[r.get("emotion")].append(r)
        for emo in EMOTIONS:
            bucket = by_emo.get(emo, [])
            k = max(1, len(bucket) // PER_EMOTION_PER_LANG) if bucket else 1
            rows.extend(bucket[::k][:PER_EMOTION_PER_LANG])
    return rows


def sample() -> None:
    recs = json.loads(FINAL_SELECTION.read_text(encoding="utf-8"))
    rows = _sample_rows(recs)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "language", "current_emotion", "transcript", "true_emotion", "notes"])
        for r in rows:
            sid = os.path.splitext(os.path.basename(r["audio"]))[0]
            w.writerow([sid, r["language"], r["emotion"], r["text"], "", ""])
    items = []
    for r in rows:
        sid = os.path.splitext(os.path.basename(r["audio"]))[0]
        wav = os.path.relpath(PROJECT_ROOT / r["audio"], REVIEW_DIR).replace("\\", "/")
        items.append({"id": sid, "lang": r["language"], "cur": r["emotion"], "text": r["text"], "wav": wav})
    html = _HTML.replace("__DATA__", json.dumps(items, ensure_ascii=False)).replace(
        "__EMO__", json.dumps(EMOTIONS))
    (REVIEW_DIR / "emotion_relabel.html").write_text(html, encoding="utf-8")
    print(f"wrote {CSV_PATH} ({len(rows)} clips) and {REVIEW_DIR/'emotion_relabel.html'}")
    print("Open emotion_relabel.html, listen, pick the true emotion, Export, save over the CSV, then: score")


def score() -> None:
    if not CSV_PATH.exists():
        print(f"{CSV_PATH} not found. Run sample and fill it first."); return
    agree = total = 0
    conf = defaultdict(Counter)
    by_lang = defaultdict(lambda: [0, 0])
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cur, true = (row.get("current_emotion") or "").strip(), (row.get("true_emotion") or "").strip()
            if not true:
                continue
            total += 1
            by_lang[row["language"]][1] += 1
            if cur == true:
                agree += 1; by_lang[row["language"]][0] += 1
            conf[cur][true] += 1
    if not total:
        print("No rows filled yet."); return
    print(f"human-vs-LLM emotion agreement: {agree}/{total} = {round(100*agree/total)}%")
    for lang, (a, n) in by_lang.items():
        print(f"  {lang}: {a}/{n} = {round(100*a/n)}%")
    print("confusion (LLM tag -> human true):")
    for cur, c in conf.items():
        print(f"  {cur:10} -> {dict(c.most_common())}")


_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Emotion relabel</title>
<style>body{font:14px system-ui;margin:0;background:#0f1115;color:#e6e6e6}
header{position:sticky;top:0;background:#161922;padding:10px 14px;border-bottom:1px solid #2a2f3a}
.card{margin:10px 14px;padding:10px 12px;background:#161922;border:1px solid #2a2f3a;border-radius:8px}
audio{width:320px}button{background:#0f1115;color:#e6e6e6;border:1px solid #333a48;border-radius:6px;padding:4px 9px;cursor:pointer;margin:2px}
button.on{background:#2563eb;border-color:#2563eb}.cur{color:#f59e0b}
.lang{font-size:11px;background:#334155;padding:1px 6px;border-radius:4px;margin-right:6px}.meta{color:#9aa4b2;font-size:12px}</style></head>
<body><header><b>Emotion relabel</b> <span id="c" class="meta"></span>
<button onclick="exp()">Export emotion_relabel.csv</button>
<span class="meta">listen, pick the TRUE emotion (orange = current tag), export, save to data/manifests/emotion_relabel.csv, run: python scripts/emotion_relabel.py score</span></header>
<div id="g"></div><script>
const D=__DATA__,EMO=__EMO__,S={};D.forEach(d=>S[d.id]="");
function set(id,v){S[id]=v;render()}
function render(){const g=document.getElementById('g');g.innerHTML='';let done=0;
 D.forEach(d=>{if(S[d.id])done++;
  const btns=EMO.map(e=>`<button class="${S[d.id]===e?'on':''}" onclick="set('${d.id}','${e}')">${e}${e===d.cur?' ★':''}</button>`).join('');
  g.insertAdjacentHTML('beforeend',`<div class="card"><span class="lang">${d.lang}</span><b>${d.id}</b> <span class="meta">current tag: <span class="cur">${d.cur}</span></span>
   <div><audio controls preload="none" src="${d.wav}"></audio></div>
   <div class="meta" style="margin:4px 0">${d.text}</div><div>${btns}</div></div>`)});
 document.getElementById('c').textContent=`(${done}/${D.length} done)`}
function exp(){let o="id,language,current_emotion,transcript,true_emotion,notes\n";
 const esc=s=>'"'+String(s==null?'':s).replace(/"/g,'""')+'"';
 D.forEach(d=>{o+=[d.id,d.lang,d.cur,esc(d.text),S[d.id],''].join(',')+'\n'});
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([o],{type:'text/csv'}));a.download='emotion_relabel.csv';a.click()}
render();</script></body></html>"""


if __name__ == "__main__":
    (score if len(sys.argv) > 1 and sys.argv[1] == "score" else sample)()
