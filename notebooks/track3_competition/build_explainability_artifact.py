# -*- coding: utf-8 -*-
import numpy as np, json, os

BASE = "/Users/danieleuras/Documents/GitHub/miralis-hypergraph-imagined-speech/.claude/worktrees/elegant-neumann/notebooks/track3_competition"
OUT  = "/private/tmp/claude-501/-Users-danieleuras-Documents-GitHub-miralis-hypergraph-imagined-speech--claude-worktrees-elegant-neumann/00440521-d691-44e0-8aef-6a512ea2cd1c/scratchpad/hypertempnet_explainability.html"

POS = {"Fp1":[-0.0294367,0.0839171],"Fp2":[0.0298723,0.0848959],"F7":[-0.0702629,0.0424743],"F3":[-0.0502438,0.0531112],"Fz":[0.0003122,0.058512],"F4":[0.0518362,0.0543048],"F8":[0.0730431,0.0444217],"FC5":[-0.0772149,0.0186433],"FC1":[-0.0340619,0.0260111],"FC2":[0.0347841,0.0264379],"FC6":[0.0795341,0.0199357],"T7":[-0.0841611,-0.0160187],"C3":[-0.0653581,-0.0116317],"Cz":[0.0004009,-0.009167],"C4":[0.0671179,-0.0109003],"T8":[0.0850799,-0.0150203],"TP9":[-0.0856192,-0.0465147],"CP5":[-0.0795922,-0.0465507],"CP1":[-0.0355131,-0.0472919],"CP2":[0.0383838,-0.0470731],"CP6":[0.0833218,-0.0461013],"TP10":[0.0861618,-0.0470353],"P7":[-0.0724343,-0.0734527],"P3":[-0.0530073,-0.0787878],"Pz":[0.0003247,-0.0811149],"P4":[0.0556667,-0.0785602],"P8":[0.0730557,-0.0730683],"PO9":[-0.0549104,-0.0980448],"O1":[-0.0294134,-0.112449],"Oz":[0.0001076,-0.114892],"O2":[0.0298426,-0.112156],"PO10":[0.0549876,-0.0980911],"AF7":[-0.0548397,0.0685722],"AF3":[-0.0337007,0.0768371],"AF4":[0.0357123,0.0777259],"AF8":[0.0557433,0.0696568],"F5":[-0.0644658,0.0480353],"F1":[-0.0274958,0.0569311],"F2":[0.0295142,0.0576019],"F6":[0.0679142,0.0498297],"FT9":[-0.0840759,0.0145673],"FT7":[-0.080775,0.0141203],"FC3":[-0.0601819,0.0227162],"FC4":[0.0622931,0.0237228],"FT8":[0.0818151,0.0154167],"FT10":[0.0841131,0.0143647],"C5":[-0.0802801,-0.0137597],"C1":[-0.036158,-0.0099839],"C2":[0.037672,-0.0096241],"C6":[0.0834559,-0.0127763],"TP7":[-0.0848302,-0.0460217],"CP3":[-0.0635562,-0.0470088],"CPz":[0.0003858,-0.047318],"CP4":[0.0666118,-0.0466372],"TP8":[0.0855488,-0.0455453],"P5":[-0.0672723,-0.0762907],"P1":[-0.0286203,-0.0805249],"P2":[0.0319197,-0.0804871],"P6":[0.0678877,-0.0759043],"PO7":[-0.0548404,-0.0975279],"PO3":[-0.0365114,-0.1008529],"POz":[0.0002156,-0.1021779],"PO4":[0.0367816,-0.1008491],"PO8":[0.0556666,-0.0976251]}

d = np.load(os.path.join(BASE, "results/ig_hypertempnet.npz"), allow_pickle=True)
clab = [str(c) for c in d["clab"]]; cls = [str(c) for c in d["class_names"]]
sp = d["spatial"].astype(float)      # (5,64)
tp = d["temporal"].astype(float)     # (5,795)
acc = d["acc"].astype(float); fs = int(d["fs"]); nT = int(d["n_times"])

# posizioni normalizzate (naso in alto = +y). raggio unitario
maxr = max((POS[c][0]**2 + POS[c][1]**2) ** 0.5 for c in clab)
pos_n = {c: [POS[c][0]/maxr, POS[c][1]/maxr] for c in clab}

gmin, gmax = sp.min(), sp.max()
def nrm(v): return (v - gmin) / (gmax - gmin)

words = []
for ci, w in enumerate(cls):
    elos = [{"ch": clab[j], "x": round(pos_n[clab[j]][0], 4), "y": round(pos_n[clab[j]][1], 4),
             "v": round(float(nrm(sp[ci, j])), 4)} for j in range(len(clab))]
    order = np.argsort(sp[ci])[::-1]
    top = [clab[j] for j in order[:4]]
    words.append({"word": w, "acc": round(float(acc[ci]), 3), "electrodes": elos, "top": top})

# profilo temporale: downsample a 160 punti, mean + per-parola, normalizzato 0..1 globale
def ds(a, n=160):
    idx = np.linspace(0, len(a)-1, n).astype(int); return a[idx]
tmean = tp.mean(0); tallmin, tallmax = tp.min(), tp.max()
def tnrm(a): return [(round(float((x - tallmin)/(tallmax - tallmin)), 4)) for x in ds(a)]
temporal = {"t_ms": [int(round(x)) for x in np.linspace(0, nT/fs*1000, 160)],
            "mean": tnrm(tmean), "per_word": {cls[i]: tnrm(tp[i]) for i in range(len(cls))}}

DATA = {"words": words, "temporal": temporal,
        "disc_s": round(float(d["disc_s"]), 4), "disc_t": round(float(d["disc_t"]), 4),
        "test_bacc": 0.556}

html = """<h2 class="sr-only">Mappe di explainability (integrated gradients) di HyperTempNet: per ognuna delle cinque parole immaginate, la topografia sullo scalpo dei canali EEG che guidano la predizione, piu il profilo temporale della saliency.</h2>
<main>
  <header class="hd">
    <div class="eyebrow">EXPLAINABILITY &middot; INTEGRATED GRADIENTS</div>
    <h1>Cosa guarda HyperTempNet<span class="thin"> per riconoscere ogni parola immaginata</span></h1>
    <p class="sub">Attribuzione sull'EEG grezzo per le 5 parole del Track&#8203;#3 (BCIC2020-3). Ogni topografia mostra <em>quali canali</em> spingono la predizione di quella parola &mdash; calcolata dal modello addestrato (subject-mixed, bacc&nbsp;0.556). Colore piu caldo = piu importante.</p>
  </header>

  <div class="legendbar">
    <span class="lg-lo">meno</span>
    <div id="cbar" class="cbar" aria-hidden="true"></div>
    <span class="lg-hi">piu importante</span>
    <div class="tip">passa il mouse su un elettrodo</div>
  </div>

  <section class="heads" id="heads" aria-label="Topografie per parola"></section>

  <section class="tpanel">
    <div class="tp-head">
      <h2>Profilo temporale della saliency</h2>
      <p>Dove nel trial (~3.1&nbsp;s) il modello concentra l'attenzione. Le 5 parole condividono quasi lo stesso profilo (disc.&nbsp;temporale 0.02) &mdash; la differenza tra parole e <strong>spaziale</strong> (disc.&nbsp;0.12), non temporale.</p>
    </div>
    <div class="chartwrap"><canvas id="tchart" height="150"></canvas></div>
  </section>

  <footer class="note">
    <div class="ntitle">Metodo &amp; onesta</div>
    <ul>
      <li><strong>Integrated gradients</strong> (32 step, baseline&nbsp;=&nbsp;0 = media del segnale z-scored), attribuiti alla classe vera, |IG| mediato su tutti i trial di test di ogni parola. Ogni parola e pronunciata da tutti i 15 soggetti &rarr; mediando, l'identita del soggetto si annulla: le differenze tra parole sono segnale <em>di parola</em>.</li>
      <li>La struttura discriminante e <strong>spaziale</strong> (lateralizzazione: Hello/Helpme a destra temporo-parietale, Stop/Yes a sinistra occipito-parietale, Thankyou fronto-centrale), non temporale.</li>
      <li><strong>Nota critica</strong>: l'incidenza degli iperarchi <code>H</code> del modello e risultata quasi <em>uniforme</em> (entropia 0.998) &mdash; il ramo ipergrafo agisce come pooling temporale regolarizzato, non come selezione di motivi. Questa explainability viene dall'attribuzione sull'input, che e fedele, non da <code>H</code>.</li>
    </ul>
  </footer>
</main>

<div id="tooltip" class="tooltip" role="status"></div>

<style>
:root{
  --bg:#fbfcfe; --panel:#ffffff; --ink:#18202c; --muted:#5c6775; --line:#e4e9f0;
  --accent:#21918c; --eyebrow:#3b528b; --shadow:0 1px 2px rgba(20,30,50,.04),0 8px 24px rgba(20,30,50,.06);
  --head:#eef1f6; --headline:#c3ccd8;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0c1016; --panel:#141a22; --ink:#e6edf5; --muted:#8b97a7; --line:#222c38;
  --accent:#3fb6ab; --eyebrow:#8aa0d8; --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
  --head:#1b222c; --headline:#2c3744;
}}
:root[data-theme="light"]{--bg:#fbfcfe;--panel:#ffffff;--ink:#18202c;--muted:#5c6775;--line:#e4e9f0;--accent:#21918c;--eyebrow:#3b528b;--shadow:0 1px 2px rgba(20,30,50,.04),0 8px 24px rgba(20,30,50,.06);--head:#eef1f6;--headline:#c3ccd8;}
:root[data-theme="dark"]{--bg:#0c1016;--panel:#141a22;--ink:#e6edf5;--muted:#8b97a7;--line:#222c38;--accent:#3fb6ab;--eyebrow:#8aa0d8;--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);--head:#1b222c;--headline:#2c3744;}
*{box-sizing:border-box}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap;border:0}
main{max-width:960px;margin:0 auto;padding:34px 22px 46px;color:var(--ink);
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-feature-settings:"ss01";line-height:1.5;background:var(--bg)}
.eyebrow{font-size:11.5px;letter-spacing:.14em;font-weight:650;color:var(--eyebrow);margin-bottom:12px}
h1{font-size:29px;line-height:1.14;font-weight:680;letter-spacing:-.018em;margin:0 0 12px;text-wrap:balance}
h1 .thin{font-weight:430;color:var(--muted)}
.sub{max-width:63ch;color:var(--muted);font-size:15px;margin:0}
.sub em{color:var(--ink);font-style:normal;font-weight:600}
.legendbar{display:flex;align-items:center;gap:11px;margin:26px 0 18px;flex-wrap:wrap}
.cbar{height:11px;width:230px;border-radius:6px;border:1px solid var(--line)}
.lg-lo,.lg-hi{font-size:12px;color:var(--muted)}
.tip{margin-left:auto;font-size:12px;color:var(--muted);font-style:italic}
.heads{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
@media(max-width:760px){.heads{grid-template-columns:repeat(2,1fr)}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:13px 11px 11px;box-shadow:var(--shadow)}
.card .wname{font-size:15px;font-weight:650;letter-spacing:-.01em;display:flex;align-items:baseline;justify-content:space-between}
.card .acc{font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums;font-weight:500}
.card svg{width:100%;height:auto;display:block;margin:4px 0 7px}
.card .top{font-size:11px;color:var(--muted);line-height:1.5}
.card .top b{color:var(--accent);font-weight:600}
.elec{stroke:rgba(120,130,145,.5);stroke-width:.6;cursor:default;transition:r .1s}
.elec:hover{stroke:var(--ink);stroke-width:1.4}
.headline{fill:none;stroke:var(--headline);stroke-width:1.5}
.tpanel{margin-top:30px;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px 20px 16px;box-shadow:var(--shadow)}
.tp-head h2{font-size:16px;margin:0 0 4px;font-weight:640}
.tp-head p{margin:0 0 6px;color:var(--muted);font-size:13.5px;max-width:70ch}
.tp-head strong{color:var(--ink)}
.chartwrap{width:100%;overflow-x:auto}
canvas{width:100%;display:block}
.note{margin-top:26px;border-top:1px solid var(--line);padding-top:18px}
.ntitle{font-size:11.5px;letter-spacing:.13em;font-weight:650;color:var(--eyebrow);margin-bottom:10px}
.note ul{margin:0;padding-left:18px;color:var(--muted);font-size:13px;line-height:1.62}
.note li{margin-bottom:7px}
.note strong{color:var(--ink)}
.note em{color:var(--ink);font-style:normal}
.note code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;background:var(--head);padding:1px 5px;border-radius:4px}
.tooltip{position:fixed;pointer-events:none;opacity:0;transform:translate(-50%,-130%);
  background:var(--ink);color:var(--bg);font-size:12px;font-weight:550;padding:5px 9px;border-radius:7px;
  white-space:nowrap;z-index:9;transition:opacity .1s;font-variant-numeric:tabular-nums}
</style>

<script>
const DATA = __DATA__;
// viridis (5 stop) -> interpolazione
const VIR = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
function viridis(t){ t=Math.max(0,Math.min(1,t)); const s=t*4, i=Math.min(3,Math.floor(s)), f=s-i;
  const a=VIR[i], b=VIR[i+1]; return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},${Math.round(a[2]+(b[2]-a[2])*f)})`; }

// legenda colormap
(function(){ let s='linear-gradient(90deg,'; for(let i=0;i<=10;i++){ s+=viridis(i/10)+(i<10?',':''); } s+=')';
  document.getElementById('cbar').style.background=s; })();

// topomap SVG per parola
const VB=100, R=40, CX=50, CY=52;
function head(w){
  const el=w.electrodes.map(e=>{
    const x=CX+e.x*R, y=CY-e.y*R;   // naso in alto
    const r=2.0+e.v*2.6;
    return `<circle class="elec" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="${viridis(e.v)}" data-ch="${e.ch}" data-v="${e.v}"/>`;
  }).join('');
  const nose=`<path class="headline" d="M ${CX-5} ${CY-R+1} Q ${CX} ${CY-R-6} ${CX+5} ${CY-R+1}"/>`;
  const ears=`<path class="headline" d="M ${CX-R+1} ${CY-6} q -5 6 0 12"/><path class="headline" d="M ${CX+R-1} ${CY-6} q 5 6 0 12"/>`;
  return `<svg viewBox="0 0 ${VB} ${VB-4}" role="img" aria-label="topografia ${w.word}">
    <circle class="headline" cx="${CX}" cy="${CY}" r="${R}"/>${nose}${ears}${el}</svg>`;
}
const topHtml=w=>w.top.map((c,i)=>i===0?`<b>${c}</b>`:c).join(' &middot; ');
document.getElementById('heads').innerHTML = DATA.words.map(w=>`
  <div class="card">
    <div class="wname">${w.word}<span class="acc">${(w.acc*100).toFixed(0)}%</span></div>
    ${head(w)}
    <div class="top">${topHtml(w)}</div>
  </div>`).join('');

// tooltip
const tt=document.getElementById('tooltip');
document.getElementById('heads').addEventListener('mousemove',e=>{
  const t=e.target;
  if(t.classList&&t.classList.contains('elec')){
    tt.textContent=`${t.dataset.ch}  ${(t.dataset.v*100|0)}%`;
    tt.style.left=e.clientX+'px'; tt.style.top=e.clientY+'px'; tt.style.opacity=1;
  } else tt.style.opacity=0;
});
document.getElementById('heads').addEventListener('mouseleave',()=>tt.style.opacity=0);

// striscia temporale
function drawT(){
  const c=document.getElementById('tchart'), dpr=window.devicePixelRatio||1;
  const W=c.clientWidth, H=150; c.width=W*dpr; c.height=H*dpr; const g=c.getContext('2d'); g.scale(dpr,dpr);
  g.clearRect(0,0,W,H);
  const cs=getComputedStyle(document.documentElement);
  const line=cs.getPropertyValue('--line'), muted=cs.getPropertyValue('--muted'), ink=cs.getPropertyValue('--ink');
  const T=DATA.temporal, n=T.mean.length, padL=8,padR=8,padT=12,padB=22;
  const x=i=>padL+(W-padL-padR)*i/(n-1), y=v=>padT+(H-padT-padB)*(1-v);
  // griglia secondi
  g.strokeStyle=line; g.lineWidth=1; g.fillStyle=muted; g.font='11px ui-sans-serif,system-ui'; g.textAlign='center';
  for(let s=0;s<=3;s++){ const xi=padL+(W-padL-padR)*(s*1000)/T.t_ms[n-1]; g.beginPath();g.moveTo(xi,padT);g.lineTo(xi,H-padB);g.stroke(); g.fillText(s+'s',xi,H-6); }
  // per-parola (tenui)
  Object.values(T.per_word).forEach(arr=>{ g.strokeStyle='rgba(33,145,140,.22)'; g.lineWidth=1; g.beginPath();
    arr.forEach((v,i)=>{ i?g.lineTo(x(i),y(v)):g.moveTo(x(i),y(v)); }); g.stroke(); });
  // media (bold, viridis-teal)
  g.strokeStyle=cs.getPropertyValue('--accent'); g.lineWidth=2.4; g.beginPath();
  T.mean.forEach((v,i)=>{ i?g.lineTo(x(i),y(v)):g.moveTo(x(i),y(v)); }); g.stroke();
  g.fillStyle=muted; g.textAlign='left'; g.fillText('|IG| medio nel tempo',padL+2,padT+2);
}
drawT(); addEventListener('resize',drawT);
const mo=new MutationObserver(drawT); mo.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
</script>"""

html = html.replace("__DATA__", json.dumps(DATA))
with open(OUT, "w") as f:
    f.write(html)
print("written", OUT, len(html), "bytes")
print("disc_s", DATA["disc_s"], "disc_t", DATA["disc_t"])
