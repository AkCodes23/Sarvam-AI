"""Human-in-the-loop review. Builds a self-contained static HTML app (open in a
browser) to listen, fix transcripts, relabel emotion/style, and accept/reject.
Exports a decisions CSV; `merge_decisions` applies it — human edits always win."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, REVIEW_DIR, Config
from .models import Segment, load_all_segments, save_segments

REVIEW_CSV = PROJECT_ROOT / "data" / "manifests" / "review.csv"


def build_review_app(cfg: Config, segments: list[Segment] | None = None) -> Path:
    segments = segments if segments is not None else load_all_segments()
    rows = []
    for s in segments:
        if s.status == "reject":
            continue
        wav_abs = PROJECT_ROOT / s.wav_path
        rows.append({
            "id": s.id, "language": s.language, "source_id": s.source_id,
            "speaker_id": s.speaker_id,
            "wav": os.path.relpath(wav_abs, REVIEW_DIR).replace("\\", "/"),
            "transcript": s.transcript, "emotion": s.emotion or "neutral",
            "style": s.style or "narrative",
            "confidence": s.emotion_confidence, "rationale": s.emotion_rationale or "",
            "status": s.status, "flags": ", ".join(s.flags),
            "dur": round(s.duration_s, 1), "snr": s.metrics.get("snr_db"),
            "gap": s.metrics.get("gap_energy_ratio"),
            "url": f"{s.source_url}&t={int(s.start_s)}" if s.source_url else "",
        })

    html = _HTML.replace("__DATA__", json.dumps(rows, ensure_ascii=False)) \
               .replace("__EMOTIONS__", json.dumps(cfg.emotion.emotions)) \
               .replace("__STYLES__", json.dumps(cfg.emotion.styles))
    out = REVIEW_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


def merge_decisions(csv_path: Path = REVIEW_CSV) -> dict:
    """Apply human decisions. Reject -> dropped; changed emotion/style/transcript
    -> overrides with tag_source='human'."""
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    decisions = {r["id"]: r for _, r in df.iterrows()}

    segs = load_all_segments()
    stats = {"accepted": 0, "rejected": 0, "relabeled": 0, "transcript_fixed": 0}
    for s in segs:
        d = decisions.get(s.id)
        if d is None:
            continue
        decision = (d.get("decision") or "").strip().lower()
        if decision == "reject":
            s.review_decision = "reject"
            stats["rejected"] += 1
            continue
        if decision == "accept":
            s.review_decision = "accept"
            stats["accepted"] += 1

        new_e = (d.get("emotion") or "").strip().lower()
        new_s = (d.get("style") or "").strip().lower()
        new_t = (d.get("transcript") or "").strip()
        changed = False
        if new_e and new_e != (s.emotion or ""):
            s.emotion = new_e; changed = True
        if new_s and new_s != (s.style or ""):
            s.style = new_s; changed = True
        if new_t and new_t != s.transcript:
            s.transcript = new_t; stats["transcript_fixed"] += 1; changed = True
        if changed:
            s.tag_source = "human"
            s.emotion_confidence = 1.0
            stats["relabeled"] += 1

    by_source: dict[str, list[Segment]] = defaultdict(list)
    for s in segs:
        by_source[s.source_id].append(s)
    for sid, group in by_source.items():
        save_segments(sid, group)
    return stats


_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>TTS Dataset Review</title>
<style>
 body{font:14px/1.4 system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 header{position:sticky;top:0;background:#161922;padding:10px 16px;border-bottom:1px solid #2a2f3a;z-index:5}
 header h1{font-size:16px;margin:0 0 6px} header .ctl{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 select,input,button{background:#0f1115;color:#e6e6e6;border:1px solid #333a48;border-radius:6px;padding:5px 8px}
 button{cursor:pointer;background:#2563eb;border-color:#2563eb} button.sec{background:#0f1115}
 .grid{padding:12px 16px;display:grid;gap:10px}
 .card{background:#161922;border:1px solid #2a2f3a;border-radius:10px;padding:10px 12px;display:grid;
   grid-template-columns:280px 1fr;gap:12px;align-items:start}
 .card.rej{opacity:.45} .card.acc{border-color:#16a34a}
 .meta{font:12px ui-monospace,monospace;color:#9aa4b2} .flag{color:#f59e0b}
 audio{width:260px} textarea{width:100%;min-height:48px;background:#0f1115;color:#e6e6e6;border:1px solid #333a48;border-radius:6px;padding:6px}
 .lang{display:inline-block;font-size:11px;padding:1px 6px;border-radius:4px;background:#334155;margin-right:6px}
 .row{display:flex;gap:8px;align-items:center;margin-top:6px;flex-wrap:wrap}
 a{color:#60a5fa}
</style></head><body>
<header>
 <h1>TTS Dataset Review <span id="count" class="meta"></span></h1>
 <div class="ctl">
  <label>Lang <select id="fl"><option value="">all</option><option>en</option><option>te</option></select></label>
  <label>Show <select id="fs"><option value="">all</option><option>flag</option><option>pass</option></select></label>
  <label><input type="checkbox" id="fneed"> only undecided</label>
  <button onclick="exportCsv()">⬇ Export review.csv</button>
  <span class="meta">edits autosave in this page; export when done → save to data/manifests/review.csv → run <code>ttsds review-merge</code></span>
 </div>
</header>
<div id="grid" class="grid"></div>
<script>
const DATA=__DATA__, EMOTIONS=__EMOTIONS__, STYLES=__STYLES__;
const state={}; DATA.forEach(d=>state[d.id]={decision:"",emotion:d.emotion,style:d.style,transcript:d.transcript});
function opts(list,sel){return list.map(o=>`<option ${o===sel?'selected':''}>${o}</option>`).join('')}
function render(){
 const fl=fl_.value, fs=fs_.value, need=fneed_.checked;
 const g=document.getElementById('grid'); g.innerHTML='';
 let shown=0;
 DATA.forEach(d=>{
  if(fl&&d.language!==fl)return; if(fs&&d.status!==fs)return;
  if(need&&state[d.id].decision)return; shown++;
  const st=state[d.id];
  const card=document.createElement('div');
  card.className='card'+(st.decision==='reject'?' rej':st.decision==='accept'?' acc':'');
  card.innerHTML=`<div>
     <div><span class="lang">${d.language}</span><b>${d.id}</b></div>
     <audio controls preload="none" src="${d.wav}"></audio>
     <div class="meta">spk:${d.speaker_id} · ${d.dur}s · snr:${d.snr} · gap:${d.gap}
       ${d.flags?`<div class="flag">⚑ ${d.flags}</div>`:''}
       <div>conf:${d.confidence} — ${d.rationale}</div>
       ${d.url?`<a href="${d.url}" target="_blank">source ↗</a>`:''}</div>
   </div>
   <div>
     <textarea oninput="upd('${d.id}','transcript',this.value)">${st.transcript}</textarea>
     <div class="row">
      <label>emotion <select onchange="upd('${d.id}','emotion',this.value)">${opts(EMOTIONS,st.emotion)}</select></label>
      <label>style <select onchange="upd('${d.id}','style',this.value)">${opts(STYLES,st.style)}</select></label>
      <button class="sec" onclick="dec('${d.id}','accept')">✓ accept</button>
      <button class="sec" onclick="dec('${d.id}','reject')">✗ reject</button>
      <span class="meta" id="d_${d.id}">${st.decision||''}</span>
     </div>
   </div>`;
  g.appendChild(card);
 });
 document.getElementById('count').textContent=`(${shown} shown / ${DATA.length} total)`;
}
function upd(id,k,v){state[id][k]=v}
function dec(id,v){state[id].decision=v; render()}
function exportCsv(){
 const esc=s=>'"'+String(s==null?'':s).replace(/"/g,'""')+'"';
 let out='id,decision,emotion,style,transcript\n';
 DATA.forEach(d=>{const s=state[d.id];out+=[d.id,s.decision,s.emotion,s.style,esc(s.transcript)].map((v,i)=>i===4?v:esc(v)).join(',')+'\n'});
 const blob=new Blob([out],{type:'text/csv'});const a=document.createElement('a');
 a.href=URL.createObjectURL(blob);a.download='review.csv';a.click();
}
const fl_=document.getElementById('fl'),fs_=document.getElementById('fs'),fneed_=document.getElementById('fneed');
[fl_,fs_,fneed_].forEach(el=>el.addEventListener('change',render)); render();
</script></body></html>"""
