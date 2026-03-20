const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Daniele Uras";
pres.title = "Decodifica Imagined Speech da EEG con GNN";

// ── PALETTE ──────────────────────────────────────────────────────────────────
const C = {
  navy:    "0D1B2A",   // sfondo dark
  teal:    "00A896",   // accento principale
  tealDk:  "028090",   // accento scuro
  white:   "FFFFFF",
  offWhite:"F4F7F9",
  gray:    "64748B",
  lightBg: "EEF3F7",
  text:    "1E293B",
  muted:   "94A3B8",
  green:   "22C55E",
  amber:   "F59E0B",
  red:     "EF4444",
};

const makeShadow = () => ({
  type: "outer", color: "000000", opacity: 0.12, blur: 8, offset: 3, angle: 135
});

// ── SLIDE 1 — TITLE ──────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  // Left teal bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.25, h: 5.625, fill: { color: C.teal }, line: { color: C.teal }
  });

  // Decorative teal rectangle top-right
  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.5, y: 0, w: 2.5, h: 1.2, fill: { color: C.tealDk }, line: { color: C.tealDk }
  });

  // Institution tag
  s.addText("Politecnico di Milano — DEIB", {
    x: 0.5, y: 0.35, w: 7, h: 0.4,
    fontSize: 11, color: C.teal, fontFace: "Calibri Light", margin: 0
  });

  // Main title
  s.addText("Decodifica dell'Imagined Speech\nda Segnali EEG", {
    x: 0.5, y: 1.1, w: 9, h: 1.8,
    fontSize: 40, bold: true, color: C.white, fontFace: "Calibri",
    lineSpacingMultiple: 1.2
  });

  // Subtitle
  s.addText("con Graph Neural Networks e Hypergraph Learning", {
    x: 0.5, y: 2.9, w: 9, h: 0.55,
    fontSize: 20, color: C.teal, fontFace: "Calibri Light", margin: 0
  });

  // Horizontal divider
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.55, w: 5, h: 0.04, fill: { color: C.tealDk }, line: { color: C.tealDk }
  });

  // Author / role
  s.addText([
    { text: "Daniele Uras", options: { bold: true, breakLine: true } },
    { text: "Tesi Magistrale — 2025/2026", options: { breakLine: true } },
    { text: "Marzo 2026", options: {} }
  ], {
    x: 0.5, y: 3.75, w: 6, h: 1.3,
    fontSize: 14, color: C.muted, fontFace: "Calibri Light"
  });

  // Brain/EEG icon placeholder (colored circle)
  s.addShape(pres.shapes.OVAL, {
    x: 7.8, y: 2.0, w: 1.8, h: 1.8,
    fill: { color: C.teal, transparency: 80 }, line: { color: C.teal, width: 2 }
  });
  s.addText("🧠", { x: 7.8, y: 2.0, w: 1.8, h: 1.8, fontSize: 40, align: "center", valign: "middle" });
}

// ── SLIDE 2 — OBIETTIVO ───────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };

  // Header bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 1.0, fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("Obiettivo della Tesi", {
    x: 0.4, y: 0, w: 9, h: 1.0,
    fontSize: 26, bold: true, color: C.white, fontFace: "Calibri", valign: "middle", margin: 0
  });

  // Left col
  s.addText("Il Task", {
    x: 0.4, y: 1.2, w: 4.3, h: 0.45,
    fontSize: 16, bold: true, color: C.tealDk, fontFace: "Calibri", margin: 0
  });
  s.addText([
    { text: "Classificare parole immaginate (imagined speech)\ndall'attività cerebrale EEG", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "110 parole italiane → 4-6 categorie semantiche", options: { breakLine: true } },
    { text: "70 soggetti, 5 sessioni, 59 canali, 256 Hz", options: {} },
  ], {
    x: 0.4, y: 1.7, w: 4.3, h: 2.2,
    fontSize: 14, color: C.text, fontFace: "Calibri", lineSpacingMultiple: 1.4
  });

  // Right col
  s.addText("Approccio", {
    x: 5.3, y: 1.2, w: 4.3, h: 0.45,
    fontSize: 16, bold: true, color: C.tealDk, fontFace: "Calibri", margin: 0
  });
  const items = [
    "Segnale EEG grezzo come input (end-to-end)",
    "Graph Neural Networks per struttura spaziale",
    "Hypergraph Learning per relazioni di ordine superiore",
    "Approccio subject-independent (generalizzazione)",
  ];
  items.forEach((txt, i) => {
    s.addShape(pres.shapes.OVAL, {
      x: 5.3, y: 1.75 + i * 0.7, w: 0.3, h: 0.3,
      fill: { color: C.teal }, line: { color: C.teal }
    });
    s.addText(txt, {
      x: 5.7, y: 1.72 + i * 0.7, w: 3.9, h: 0.38,
      fontSize: 13, color: C.text, fontFace: "Calibri", valign: "middle", margin: 0
    });
  });

  // Bottom callout
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 4.2, w: 9.2, h: 0.95,
    fill: { color: C.navy }, line: { color: C.navy }, shadow: makeShadow()
  });
  s.addText("Obiettivo finale: un dizionario neurale semantico — mappatura EEG → categoria concettuale", {
    x: 0.4, y: 4.2, w: 9.2, h: 0.95,
    fontSize: 14, color: C.white, fontFace: "Calibri", align: "center", valign: "middle", margin: 0
  });
}

// ── SLIDE 3 — DATASET ─────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 1.0, fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("Dataset EEG", {
    x: 0.4, y: 0, w: 9, h: 1.0,
    fontSize: 26, bold: true, color: C.white, fontFace: "Calibri", valign: "middle", margin: 0
  });

  // Stats cards
  const stats = [
    { val: "70", lbl: "Soggetti" },
    { val: "110", lbl: "Parole" },
    { val: "59", lbl: "Canali EEG" },
    { val: "256 Hz", lbl: "Sample Rate" },
    { val: "~1.5s", lbl: "Durata Epoca" },
    { val: "5", lbl: "Sessioni/Sogg." },
  ];
  stats.forEach((st, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 0.4 + col * 3.1;
    const y = 1.15 + row * 1.4;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 2.8, h: 1.15,
      fill: { color: C.white }, line: { color: C.teal, width: 1.5 }, shadow: makeShadow()
    });
    s.addText(st.val, {
      x, y: y + 0.08, w: 2.8, h: 0.6,
      fontSize: 30, bold: true, color: C.tealDk, fontFace: "Calibri", align: "center", margin: 0
    });
    s.addText(st.lbl, {
      x, y: y + 0.65, w: 2.8, h: 0.4,
      fontSize: 12, color: C.gray, fontFace: "Calibri Light", align: "center", margin: 0
    });
  });

  // Note
  s.addText("Format: HDF5 preprocessati → tensori PyTorch (59 × 384) per trial | Canali: A1/A2 esclusi (riferimento)", {
    x: 0.4, y: 5.05, w: 9.2, h: 0.4,
    fontSize: 11, color: C.muted, fontFace: "Calibri Light", align: "center", margin: 0
  });
}

// ── SLIDE 4 — PIPELINE ────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 1.0, fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("Pipeline del Progetto", {
    x: 0.4, y: 0, w: 9, h: 1.0,
    fontSize: 26, bold: true, color: C.white, fontFace: "Calibri", valign: "middle", margin: 0
  });

  const steps = [
    { icon: "📁", title: "File H5", desc: "Dati grezzi preprocessati\n(MATLAB/EEGLAB)" },
    { icon: "⚙️", title: "Tensori", desc: "EEG_02: tensori (59×384)\nper ogni trial/soggetto" },
    { icon: "🏷️", title: "Label", desc: "EEG_00: schemi clustering\n110 parole → 4-6 classi" },
    { icon: "🧠", title: "Modelli", desc: "EEGNet, Conformer, GNN\nend-to-end su segnale raw" },
    { icon: "📊", title: "Risultati", desc: "TensorBoard: accuracy,\nbalanced acc, loss" },
  ];

  steps.forEach((st, i) => {
    const x = 0.3 + i * 1.9;
    // Card
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.2, w: 1.7, h: 3.5,
      fill: { color: C.white }, line: { color: C.teal, width: 1 }, shadow: makeShadow()
    });
    // Top teal
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.2, w: 1.7, h: 0.6,
      fill: { color: C.teal }, line: { color: C.teal }
    });
    s.addText(st.icon, {
      x, y: 1.2, w: 1.7, h: 0.6,
      fontSize: 20, align: "center", valign: "middle", margin: 0
    });
    s.addText(st.title, {
      x, y: 1.85, w: 1.7, h: 0.45,
      fontSize: 13, bold: true, color: C.navy, fontFace: "Calibri", align: "center", margin: 0
    });
    s.addText(st.desc, {
      x: x + 0.05, y: 2.35, w: 1.6, h: 2.1,
      fontSize: 10.5, color: C.text, fontFace: "Calibri Light", align: "center", lineSpacingMultiple: 1.3
    });
    // Arrow between cards
    if (i < steps.length - 1) {
      s.addShape(pres.shapes.LINE, {
        x: x + 1.75, y: 2.95, w: 0.15, h: 0,
        line: { color: C.tealDk, width: 2 }
      });
    }
  });

  s.addText("Tutti i passi sono riproducibili: paths portabili, configs/ git-tracked, environment YAMLs Linux/macOS", {
    x: 0.4, y: 5.05, w: 9.2, h: 0.4,
    fontSize: 11, color: C.muted, fontFace: "Calibri Light", align: "center", margin: 0
  });
}

// ── SLIDE 5 — FINDING CRITICO ─────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  // Section label
  s.addText("FINDING CRITICO", {
    x: 0.5, y: 0.3, w: 9, h: 0.4,
    fontSize: 11, color: C.teal, fontFace: "Calibri", charSpacing: 4, margin: 0
  });

  s.addText("La variabilità inter-soggetto domina il segnale EEG", {
    x: 0.5, y: 0.75, w: 9, h: 1.0,
    fontSize: 30, bold: true, color: C.white, fontFace: "Calibri"
  });

  // Two big stat boxes
  const boxes = [
    { val: "ε² = 0.85", sub: "Soggetto", desc: "L'85% della varianza\ndel segnale è spiegato\ndall'identità del soggetto", color: C.red },
    { val: "ε² = 0.03", sub: "Parola", desc: "Solo il 3% della varianza\nè legato alla parola\nimmaginata", color: C.amber },
    { val: "r = 0.001", sub: "RSA cross-subj", desc: "Correlazione tra matriche\nRDM soggetti diversi:\nnessuna struttura comune", color: C.teal },
  ];
  boxes.forEach((b, i) => {
    const x = 0.4 + i * 3.15;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 2.0, w: 2.9, h: 3.1,
      fill: { color: "FFFFFF", transparency: 92 }, line: { color: b.color, width: 2 }
    });
    s.addText(b.val, {
      x, y: 2.1, w: 2.9, h: 0.85,
      fontSize: 32, bold: true, color: b.color, fontFace: "Calibri", align: "center", margin: 0
    });
    s.addText(b.sub, {
      x, y: 2.9, w: 2.9, h: 0.35,
      fontSize: 12, color: C.muted, fontFace: "Calibri Light", align: "center", margin: 0
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.5, y: 3.3, w: 1.9, h: 0.03,
      fill: { color: b.color }, line: { color: b.color }
    });
    s.addText(b.desc, {
      x: x + 0.05, y: 3.45, w: 2.8, h: 1.4,
      fontSize: 11, color: C.muted, fontFace: "Calibri Light", align: "center", lineSpacingMultiple: 1.3
    });
  });

  s.addText("Conclusione: i trial si raggruppano per soggetto, non per parola. Ogni modello deve affrontare il domain shift inter-soggetto.", {
    x: 0.4, y: 5.2, w: 9.2, h: 0.35,
    fontSize: 11, color: C.muted, fontFace: "Calibri Light", align: "center", margin: 0
  });
}

// ── SLIDE 6 — SCHEMI CLUSTERING ───────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 1.0, fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("Schemi di Clustering — Riduzione del Task", {
    x: 0.4, y: 0, w: 9, h: 1.0,
    fontSize: 26, bold: true, color: C.white, fontFace: "Calibri", valign: "middle", margin: 0
  });

  // Two columns: word-based vs eeg-based
  // Left: word-based (good)
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 1.1, w: 4.5, h: 0.5,
    fill: { color: C.teal }, line: { color: C.teal }
  });
  s.addText("✓  Word-Based (affidabili)", {
    x: 0.3, y: 1.1, w: 4.5, h: 0.5,
    fontSize: 13, bold: true, color: C.white, fontFace: "Calibri", valign: "middle", align: "center", margin: 0
  });

  const wbSchemes = [
    { name: "concr4", k: 4, desc: "CONCR/AZIONE/STATO/ASTRATTO\n(Binder 2011)", imb: "2.4x" },
    { name: "phon4",  k: 4, desc: "VOC/LAB/COR/DOR onset\n(Cooney 2020)", imb: "1.8x" },
    { name: "sem5",   k: 5, desc: "Substrati neurali\n(limbico, motorio...)", imb: "4.8x" },
    { name: "ward4/5/6", k: "4-6", desc: "Ward gerarchico\nsu SBERT embeddings", imb: "3-34x" },
  ];
  wbSchemes.forEach((sc, i) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.3, y: 1.65 + i * 0.8, w: 4.5, h: 0.72,
      fill: { color: C.white }, line: { color: C.teal, width: 0.8 }
    });
    s.addText(sc.name, {
      x: 0.4, y: 1.68 + i * 0.8, w: 1.0, h: 0.35,
      fontSize: 12, bold: true, color: C.tealDk, fontFace: "Calibri", margin: 0
    });
    s.addText(`k=${sc.k} | imb: ${sc.imb}`, {
      x: 0.4, y: 2.02 + i * 0.8, w: 1.6, h: 0.28,
      fontSize: 9.5, color: C.muted, fontFace: "Calibri Light", margin: 0
    });
    s.addText(sc.desc, {
      x: 2.05, y: 1.68 + i * 0.8, w: 2.6, h: 0.66,
      fontSize: 10.5, color: C.text, fontFace: "Calibri Light", lineSpacingMultiple: 1.2
    });
  });

  // Right: eeg-based (bad)
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.1, w: 4.5, h: 0.5,
    fill: { color: C.gray }, line: { color: C.gray }
  });
  s.addText("✗  EEG-Based (instabili)", {
    x: 5.2, y: 1.1, w: 4.5, h: 0.5,
    fontSize: 13, bold: true, color: C.white, fontFace: "Calibri", valign: "middle", align: "center", margin: 0
  });

  s.addText([
    { text: "eeg_4, eeg_5, eeg_z4, eeg_z5, eeg_hdb2", options: { bold: true, breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "ARI cross-soggetto ≈ 0 — i cluster dipendono\ndal soggetto specifico, non dalle parole.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Silhouette nello spazio EEG ≈ 0.10\n(solo marginalmente sopra chance)\n\nRSA r = 0.001 → nessuna struttura comune\ntra soggetti diversi", options: {} },
  ], {
    x: 5.3, y: 1.7, w: 4.3, h: 3.6,
    fontSize: 12, color: C.text, fontFace: "Calibri Light", lineSpacingMultiple: 1.4
  });

  s.addText("La classificazione usa SOLO schemi word-based come target — label stabili e neuroscientificamente motivate", {
    x: 0.3, y: 5.1, w: 9.4, h: 0.35,
    fontSize: 10.5, color: C.muted, fontFace: "Calibri Light", align: "center", margin: 0
  });
}

// ── SLIDE 7 — BASELINE VETTORIALI E GNN ──────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 1.0, fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("Baseline: Modelli Vettoriali e GNN (Feature-Based)", {
    x: 0.4, y: 0, w: 9, h: 1.0,
    fontSize: 24, bold: true, color: C.white, fontFace: "Calibri", valign: "middle", margin: 0
  });

  // Table header
  const colX = [0.3, 2.5, 4.5, 6.0, 7.6];
  const headers = ["Modello", "Rappresentazione", "Dimensione", "Subj-Spec.", "Subj-Indep."];
  headers.forEach((h, i) => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: colX[i], y: 1.1, w: i === 0 ? 2.1 : i === 1 ? 1.9 : i === 2 ? 1.4 : 1.5, h: 0.45,
      fill: { color: C.tealDk }, line: { color: C.tealDk }
    });
    s.addText(h, {
      x: colX[i] + 0.05, y: 1.1, w: i === 0 ? 2.0 : i === 1 ? 1.8 : i === 2 ? 1.3 : 1.4, h: 0.45,
      fontSize: 11, bold: true, color: C.white, fontFace: "Calibri", valign: "middle", margin: 0
    });
  });

  const rows = [
    ["Logistic Reg.", "Aggregato (media canali)", "(40,)", "~Chance", "~Chance"],
    ["MLP", "Aggregato (media canali)", "(40,)", "~Chance", "~Chance"],
    ["MLP", "Full flattened", "(2.360,)", "~Chance", "~Chance"],
    ["MLP", "Time-resolved", "(11.800,)", "~Chance", "~Chance"],
    ["GCN Statico", "Grafo k-NN spaziale", "59 nodi", "~Chance", "~Chance"],
    ["GCN Adattivo", "Feature-Similarity", "59 nodi", "~2x Chance", "~Chance"],
  ];

  rows.forEach((row, ri) => {
    const bg = ri % 2 === 0 ? C.white : C.lightBg;
    row.forEach((cell, ci) => {
      const w = ci === 0 ? 2.1 : ci === 1 ? 1.9 : ci === 2 ? 1.4 : 1.5;
      s.addShape(pres.shapes.RECTANGLE, {
        x: colX[ci], y: 1.6 + ri * 0.48, w, h: 0.46,
        fill: { color: bg }, line: { color: "E2E8F0", width: 0.5 }
      });
      const isHighlight = (ci === 3 && cell === "~2x Chance");
      s.addText(cell, {
        x: colX[ci] + 0.05, y: 1.6 + ri * 0.48, w: w - 0.1, h: 0.46,
        fontSize: 11, color: isHighlight ? C.amber : C.text, fontFace: "Calibri",
        bold: isHighlight, valign: "middle", margin: 0
      });
    });
  });

  s.addText("Insight: i grafi adattivi (feature-similarity) catturano struttura intra-soggetto (~2x chance) ma non generalizzano cross-soggetto.", {
    x: 0.3, y: 5.1, w: 9.4, h: 0.38,
    fontSize: 11, color: C.gray, fontFace: "Calibri Light", align: "center", margin: 0
  });
}

// ── SLIDE 8 — DEEP LEARNING END-TO-END ───────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 1.0, fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("Deep Learning End-to-End su Segnale EEG Grezzo", {
    x: 0.4, y: 0, w: 9, h: 1.0,
    fontSize: 24, bold: true, color: C.white, fontFace: "Calibri", valign: "middle", margin: 0
  });

  // Input description
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 1.1, w: 2.2, h: 1.5,
    fill: { color: C.teal }, line: { color: C.teal }
  });
  s.addText([
    { text: "Input\n", options: { bold: true, breakLine: false } },
    { text: "Tensor\n(59 × 384)\nper trial\n\n256 Hz\n1.5s", options: {} }
  ], {
    x: 0.3, y: 1.1, w: 2.2, h: 1.5,
    fontSize: 12, color: C.white, fontFace: "Calibri", align: "center", valign: "middle"
  });

  // 6 model cards
  const models = [
    { name: "EEGNet", note: "Compact CNN\nDepthwise conv" },
    { name: "ShallowFBCSP", note: "Filterbank +\nSpatial filter" },
    { name: "Deep4Net", note: "Deep CNN\nMulti-scale" },
    { name: "EEGConformer", note: "CNN + Transformer\nSelf-attention" },
    { name: "ATCNet", note: "Attention +\nTemporal conv" },
    { name: "Labram", note: "Large brain\nfoundation model" },
  ];
  models.forEach((m, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 2.8 + col * 2.35;
    const y = 1.1 + row * 1.05;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 2.15, h: 0.95,
      fill: { color: C.white }, line: { color: C.tealDk, width: 1.2 }, shadow: makeShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 0.08, h: 0.95,
      fill: { color: C.teal }, line: { color: C.teal }
    });
    s.addText(m.name, {
      x: x + 0.12, y: y + 0.05, w: 2.0, h: 0.38,
      fontSize: 12, bold: true, color: C.navy, fontFace: "Calibri", margin: 0
    });
    s.addText(m.note, {
      x: x + 0.12, y: y + 0.45, w: 2.0, h: 0.46,
      fontSize: 9.5, color: C.gray, fontFace: "Calibri Light", lineSpacingMultiple: 1.2, margin: 0
    });
  });

  // Config box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 3.3, w: 9.4, h: 1.0,
    fill: { color: C.lightBg }, line: { color: C.muted, width: 0.5 }
  });
  s.addText([
    { text: "Split subject-independent: ", options: { bold: true } },
    { text: "train 01-50, val 51-60, test 61-74  |  ", options: {} },
    { text: "Schema attivo: ", options: { bold: true } },
    { text: "concr4/phon4 (4 classi, chance 25%)  |  ", options: {} },
    { text: "Metrica: ", options: { bold: true } },
    { text: "Balanced Accuracy (robusta a squilibrio classi)", options: {} },
  ], {
    x: 0.5, y: 3.3, w: 9.0, h: 1.0,
    fontSize: 12, color: C.text, fontFace: "Calibri", valign: "middle"
  });

  // TensorBoard note
  s.addText("TensorBoard: runs/sweep_si/ — monitoraggio live loss, accuracy, balanced accuracy per ogni combinazione modello/schema", {
    x: 0.3, y: 5.08, w: 9.4, h: 0.38,
    fontSize: 11, color: C.muted, fontFace: "Calibri Light", align: "center", margin: 0
  });
}

// ── SLIDE 9 — RISULTATI DL ────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 1.0, fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("Risultati — Baseline Subject-Independent (EEG_05)", {
    x: 0.4, y: 0, w: 9, h: 1.0,
    fontSize: 24, bold: true, color: C.white, fontFace: "Calibri", valign: "middle", margin: 0
  });

  // Big centered result
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 1.1, w: 4.4, h: 1.6,
    fill: { color: C.white }, line: { color: C.teal, width: 2 }, shadow: makeShadow()
  });
  s.addText("~25%", {
    x: 0.3, y: 1.2, w: 4.4, h: 0.9,
    fontSize: 52, bold: true, color: C.tealDk, fontFace: "Calibri", align: "center", margin: 0
  });
  s.addText("Balanced Accuracy  (tutti i modelli, entrambi gli schemi)", {
    x: 0.3, y: 2.1, w: 4.4, h: 0.5,
    fontSize: 11, color: C.gray, fontFace: "Calibri Light", align: "center", margin: 0
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.0, y: 1.1, w: 4.7, h: 1.6,
    fill: { color: C.white }, line: { color: C.amber, width: 2 }, shadow: makeShadow()
  });
  s.addText("= Chance", {
    x: 5.0, y: 1.2, w: 4.7, h: 0.9,
    fontSize: 52, bold: true, color: C.amber, fontFace: "Calibri", align: "center", margin: 0
  });
  s.addText("Chance level per 4 classi = 25%  —  ratio ≈ 1.0x", {
    x: 5.0, y: 2.1, w: 4.7, h: 0.5,
    fontSize: 11, color: C.gray, fontFace: "Calibri Light", align: "center", margin: 0
  });

  // Observations
  s.addText([
    { text: "Osservazioni chiave:", options: { bold: true, breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "• ", options: {} },
    { text: "Le curve val_bacc sono piatte dall'inizio", options: { bold: true } },
    { text: " — i modelli non imparano rappresentazioni cross-subject", options: { breakLine: true } },
    { text: "•  110 classi (chance 0.9%) e 4 classi (chance 25%) sono ugualmente difficili in proporzione — il task non è il bottleneck", options: { breakLine: true } },
    { text: "•  EEGConformer: early stopping a epoca 7 (30 min) — val_bacc mai migliorata", options: { breakLine: true } },
    { text: "•  Deep4Net: 41 epoche (2.7h) per restare a 0.2503 — il training converge ma non generalizza", options: {} },
  ], {
    x: 0.3, y: 2.85, w: 9.4, h: 2.3,
    fontSize: 12.5, color: C.text, fontFace: "Calibri", lineSpacingMultiple: 1.45
  });
}

// ── SLIDE 10 — PERCHE' CHANCE / INSTANCE NORM ─────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  s.addText("INTERPRETAZIONE E PROSSIMO PASSO", {
    x: 0.5, y: 0.3, w: 9, h: 0.35,
    fontSize: 11, color: C.teal, fontFace: "Calibri", charSpacing: 4, margin: 0
  });
  s.addText("Perché tutto è a chance?\nE come si risolve.", {
    x: 0.5, y: 0.7, w: 9, h: 1.2,
    fontSize: 30, bold: true, color: C.white, fontFace: "Calibri"
  });

  // Problem box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 2.0, w: 4.4, h: 2.8,
    fill: { color: "FFFFFF", transparency: 90 }, line: { color: C.red, width: 2 }
  });
  s.addText("Il Problema", {
    x: 0.3, y: 2.05, w: 4.4, h: 0.45,
    fontSize: 14, bold: true, color: C.red, fontFace: "Calibri", align: "center", margin: 0
  });
  s.addText([
    { text: "La normalizzazione globale per-canale\n(media/std dal training set) non elimina\nil bias per-soggetto.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "I soggetti nel test set hanno distribuzioni\ndel segnale diverse da quelli nel training\nset → domain shift.", options: {} }
  ], {
    x: 0.45, y: 2.55, w: 4.1, h: 2.1,
    fontSize: 12, color: C.muted, fontFace: "Calibri Light", lineSpacingMultiple: 1.4
  });

  // Solution box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.3, y: 2.0, w: 4.4, h: 2.8,
    fill: { color: "FFFFFF", transparency: 90 }, line: { color: C.green, width: 2 }
  });
  s.addText("Instance Normalization", {
    x: 5.3, y: 2.05, w: 4.4, h: 0.45,
    fontSize: 14, bold: true, color: C.green, fontFace: "Calibri", align: "center", margin: 0
  });
  s.addText([
    { text: "Bomatter et al. (2024)", options: { bold: true, breakLine: true } },
    { text: "1 riga di codice in __getitem__:", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Normalizza ogni trial individualmente\nindipendentemente dal soggetto.\nElimina bias per-trial senza usare\nstatistiche del training set.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Toggle: USE_INSTANCE_NORM = True", options: { bold: true } }
  ], {
    x: 5.45, y: 2.55, w: 4.1, h: 2.1,
    fontSize: 12, color: C.muted, fontFace: "Calibri Light", lineSpacingMultiple: 1.35
  });

  s.addText("In esecuzione su spinlabs01: runs/sweep_si_instNorm/ — risultati attesi a breve", {
    x: 0.3, y: 5.2, w: 9.4, h: 0.3,
    fontSize: 10.5, color: C.muted, fontFace: "Calibri Light", align: "center", margin: 0
  });
}

// ── SLIDE 11 — ROADMAP ────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 1.0, fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("Roadmap — Prossimi Passi", {
    x: 0.4, y: 0, w: 9, h: 1.0,
    fontSize: 26, bold: true, color: C.white, fontFace: "Calibri", valign: "middle", margin: 0
  });

  const steps = [
    {
      phase: "IMMEDIATO",
      color: C.teal,
      items: ["Instance Normalization (Bomatter 2024) — in corso su WSL", "Subject-specific baseline con concr4 — validare struttura intra-soggetto"],
    },
    {
      phase: "BREVE",
      color: C.amber,
      items: ["Graph Attention Networks (GAT) — struttura spaziale adattiva", "Sweep schemi: concr4, phon4, sem5 con e senza Instance Norm"],
    },
    {
      phase: "MEDIO",
      color: "#7C3AED",
      items: ["Hypergraph Neural Networks (DHSLP/DHSLF — Li et al. 2025)", "Target: 78% accuracy cross-subject"],
    },
    {
      phase: "AVANZATO",
      color: C.red,
      items: ["Domain Adaptation (MMD, CORAL, adversarial)", "Contrastive Learning subject-invariant (Shen 2022)"],
    },
  ];

  steps.forEach((st, i) => {
    const x = 0.3 + (i % 2) * 4.85;
    const y = 1.15 + Math.floor(i / 2) * 2.05;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.5, h: 1.9,
      fill: { color: C.white }, line: { color: st.color, width: 1.5 }, shadow: makeShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.5, h: 0.45,
      fill: { color: st.color }, line: { color: st.color }
    });
    s.addText(st.phase, {
      x: x + 0.1, y, w: 4.3, h: 0.45,
      fontSize: 12, bold: true, color: C.white, fontFace: "Calibri", valign: "middle", charSpacing: 2, margin: 0
    });
    st.items.forEach((item, j) => {
      s.addText(`• ${item}`, {
        x: x + 0.1, y: y + 0.5 + j * 0.62, w: 4.3, h: 0.6,
        fontSize: 11.5, color: C.text, fontFace: "Calibri Light", lineSpacingMultiple: 1.2, margin: 0
      });
    });
  });
}

// ── SLIDE 12 — CONCLUSIONI ────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  // Right teal accent
  s.addShape(pres.shapes.RECTANGLE, {
    x: 9.75, y: 0, w: 0.25, h: 5.625, fill: { color: C.teal }, line: { color: C.teal }
  });

  s.addText("STATO ATTUALE", {
    x: 0.5, y: 0.5, w: 9, h: 0.35,
    fontSize: 11, color: C.teal, fontFace: "Calibri", charSpacing: 4, margin: 0
  });
  s.addText("Conclusioni e Stato del Progetto", {
    x: 0.5, y: 0.9, w: 9, h: 0.85,
    fontSize: 30, bold: true, color: C.white, fontFace: "Calibri"
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.8, w: 4, h: 0.04, fill: { color: C.teal }, line: { color: C.teal }
  });

  const conclusions = [
    { icon: "✓", text: "Pipeline EEG completa e riproducibile (EEG_00 → EEG_05)" },
    { icon: "✓", text: "7 schemi di clustering word-based documentati e git-tracked" },
    { icon: "✓", text: "6 modelli DL end-to-end configurati e testati" },
    { icon: "✓", text: "Finding critico: variabilità inter-soggetto (ε²=0.85) è il bottleneck" },
    { icon: "✓", text: "Instance Normalization implementata — in esecuzione" },
    { icon: "→", text: "Hypergraph Learning: passo successivo verso la tesi" },
  ];

  conclusions.forEach((c, i) => {
    const isArrow = c.icon === "→";
    s.addShape(pres.shapes.OVAL, {
      x: 0.5, y: 2.0 + i * 0.52, w: 0.3, h: 0.3,
      fill: { color: isArrow ? C.amber : C.teal }, line: { color: isArrow ? C.amber : C.teal }
    });
    s.addText(c.icon, {
      x: 0.5, y: 2.0 + i * 0.52, w: 0.3, h: 0.3,
      fontSize: 11, color: C.white, fontFace: "Calibri", align: "center", valign: "middle", margin: 0
    });
    s.addText(c.text, {
      x: 0.9, y: 1.98 + i * 0.52, w: 8.6, h: 0.35,
      fontSize: 13, color: isArrow ? C.amber : C.muted, fontFace: "Calibri Light", valign: "middle", margin: 0
    });
  });

  // Footer
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.25, w: 10, h: 0.375, fill: { color: C.tealDk }, line: { color: C.tealDk }
  });
  s.addText("Daniele Uras — Politecnico di Milano, DEIB — Tesi Magistrale 2025/2026 — Marzo 2026", {
    x: 0, y: 5.25, w: 10, h: 0.375,
    fontSize: 10, color: C.white, fontFace: "Calibri Light", align: "center", valign: "middle", margin: 0
  });
}

// ── WRITE FILE ────────────────────────────────────────────────────────────────
const outPath = "/Users/danieleuras/Documents/GitHub/miralis-hypergraph-imagined-speech/docs/EEG_Tesi_Stato_Progetto_Mar2026.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("✓ Presentazione salvata:", outPath);
});
