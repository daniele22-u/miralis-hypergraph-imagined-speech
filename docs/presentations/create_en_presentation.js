const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Daniele Uras";
pres.title = "Decoding Imagined Speech from EEG with GNNs — Project Status March 2026";

// ── Palette: Deep Blue + White + Cyan accent ──────────────────────────────
const C = {
  navy:    "1A2A4A",
  blue:    "1E5FAD",
  cyan:    "00B4D8",
  white:   "FFFFFF",
  light:   "F0F4FA",
  gray:    "6B7A99",
  dark:    "0D1B2A",
  green:   "2EC4B6",
  red:     "E63946",
  yellow:  "FFB703",
};

function addSlideHeader(slide, title, subtitle) {
  // Top accent bar
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.cyan }, line: { color: C.cyan }
  });
  slide.addText(title, {
    x: 0.4, y: 0.18, w: 9.2, h: 0.55,
    fontSize: 26, bold: true, color: C.navy, fontFace: "Calibri",
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.4, y: 0.72, w: 9.2, h: 0.3,
      fontSize: 13, color: C.gray, fontFace: "Calibri", margin: 0,
    });
  }
  // Separator line
  slide.addShape(pres.shapes.LINE, {
    x: 0.4, y: 1.05, w: 9.2, h: 0,
    line: { color: C.cyan, width: 1.5 }
  });
}

function card(slide, x, y, w, h, color) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h,
    fill: { color },
    line: { color, width: 0 },
    rectRadius: 0.08,
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 1 — Title
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.12, fill: { color: C.cyan }, line: { color: C.cyan }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.505, w: 10, h: 0.12, fill: { color: C.cyan }, line: { color: C.cyan }
  });

  s.addText("Decoding Imagined Speech\nfrom EEG with Graph Neural Networks", {
    x: 0.6, y: 1.2, w: 8.8, h: 1.6,
    fontSize: 34, bold: true, color: C.white, fontFace: "Calibri",
    align: "center", valign: "middle",
  });

  s.addText("Project Status — March 2026", {
    x: 0.6, y: 3.0, w: 8.8, h: 0.4,
    fontSize: 18, color: C.cyan, fontFace: "Calibri", align: "center",
  });

  s.addText("Daniele Uras\nPolitecnico di Milano, DEIB\nM.Sc. Thesis 2025–2026", {
    x: 0.6, y: 3.6, w: 8.8, h: 0.9,
    fontSize: 14, color: C.gray, fontFace: "Calibri", align: "center",
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 2 — Project Objective
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addSlideHeader(s, "Project Objective", "What are we building?");

  // Main objective box
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.4, y: 1.2, w: 9.2, h: 0.9,
    fill: { color: C.navy }, line: { color: C.navy }, rectRadius: 0.08,
  });
  s.addText("Build a neural semantic dictionary mapping EEG patterns to conceptual categories\nthrough imagined speech decoding using Graph Neural Networks", {
    x: 0.6, y: 1.25, w: 8.8, h: 0.8,
    fontSize: 14, color: C.white, fontFace: "Calibri", align: "center", valign: "middle",
  });

  // 4 info cards
  const cards = [
    { label: "Task", val: "Classify 110 imagined Italian words from 59-channel EEG", icon: "🧠" },
    { label: "Dataset", val: "70 subjects × 5 sessions × ~220 trials ≈ 77,000 trials total", icon: "📊" },
    { label: "Approach", val: "End-to-end deep learning on raw EEG signal — no manual features", icon: "⚡" },
    { label: "Target", val: "Hypergraph Neural Networks — Li et al. 2025 (78% on similar dataset)", icon: "🎯" },
  ];
  cards.forEach((c, i) => {
    const x = 0.4 + (i % 2) * 4.65;
    const y = 2.3 + Math.floor(i / 2) * 1.25;
    card(s, x, y, 4.45, 1.05, i % 2 === 0 ? C.light : "EBF5FB");
    s.addText(c.label.toUpperCase(), {
      x: x + 0.15, y: y + 0.08, w: 4.15, h: 0.25,
      fontSize: 10, bold: true, color: C.blue, fontFace: "Calibri", margin: 0,
    });
    s.addText(c.val, {
      x: x + 0.15, y: y + 0.3, w: 4.15, h: 0.65,
      fontSize: 12, color: C.dark, fontFace: "Calibri", margin: 0,
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 3 — EEG Dataset
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addSlideHeader(s, "EEG Dataset", "Signal properties and key finding");

  // Left: specs
  s.addText("Technical Specs", {
    x: 0.4, y: 1.2, w: 4.5, h: 0.35,
    fontSize: 15, bold: true, color: C.navy, fontFace: "Calibri", margin: 0,
  });
  const specs = [
    "59 valid EEG channels (A1, A2, Pz, POz removed)",
    "Sampling rate: 256 Hz",
    "Epoch duration: ~1.5s (~384 samples)",
    "350 H5 files (70 subjects × 5 sessions)",
    "PyTorch tensors: (n_trials, 59, 384)",
  ];
  specs.forEach((txt, i) => {
    s.addShape(pres.shapes.OVAL, {
      x: 0.4, y: 1.65 + i * 0.55, w: 0.22, h: 0.22,
      fill: { color: C.cyan }, line: { color: C.cyan },
    });
    s.addText(txt, {
      x: 0.72, y: 1.63 + i * 0.55, w: 4.2, h: 0.28,
      fontSize: 12, color: C.dark, fontFace: "Calibri", margin: 0,
    });
  });

  // Right: key finding big stats
  card(s, 5.2, 1.2, 4.4, 4.05, C.navy);
  s.addText("KEY FINDING", {
    x: 5.4, y: 1.35, w: 4.0, h: 0.3,
    fontSize: 11, bold: true, color: C.cyan, fontFace: "Calibri",
    align: "center", charSpacing: 3, margin: 0,
  });

  s.addText("0.85", {
    x: 5.4, y: 1.75, w: 4.0, h: 0.8,
    fontSize: 60, bold: true, color: C.white, fontFace: "Calibri", align: "center", margin: 0,
  });
  s.addText("Inter-subject variance (ε²)", {
    x: 5.4, y: 2.5, w: 4.0, h: 0.3,
    fontSize: 12, color: C.cyan, fontFace: "Calibri", align: "center", margin: 0,
  });

  s.addShape(pres.shapes.LINE, {
    x: 5.8, y: 3.0, w: 3.0, h: 0,
    line: { color: C.gray, width: 0.75, dashType: "dash" }
  });

  s.addText("0.03", {
    x: 5.4, y: 3.15, w: 4.0, h: 0.7,
    fontSize: 52, bold: true, color: C.gray, fontFace: "Calibri", align: "center", margin: 0,
  });
  s.addText("Inter-word variance (ε²)", {
    x: 5.4, y: 3.82, w: 4.0, h: 0.25,
    fontSize: 12, color: C.gray, fontFace: "Calibri", align: "center", margin: 0,
  });

  s.addText("Each subject has a unique EEG 'fingerprint'.\nInter-subject variability dominates — not the word.", {
    x: 5.4, y: 4.2, w: 4.0, h: 0.7,
    fontSize: 11, color: C.light, fontFace: "Calibri", align: "center", italic: true, margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 4 — Task Reduction
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addSlideHeader(s, "Task Reduction: 110 → 4–5 Classes", "Word-based clustering schemes — stable and replicable across subjects");

  const rows = [
    ["ward4",  "4", "Auto (Ward+SBERT)",   "Hierarchical clustering on semantic embeddings", "25%"],
    ["ward5",  "5", "Auto (Ward+SBERT)",   "Hierarchical clustering on semantic embeddings", "20%"],
    ["ward6",  "6", "Auto (Ward+SBERT)",   "Fine-grained, 6 semantic clusters",              "16.7%"],
    ["pos4",   "4", "Grammatical",         "Verbs / Nouns / Modifiers / Function words",     "25%"],
    ["sem5",   "5", "Neuroscientific",     "Actions / Emotions / Objects / Abstract / Func", "20%"],
    ["concr4", "4", "Concreteness",        "CONCR / ACTION / STATE / ABSTRACT",              "25%"],
    ["phon4",  "4", "Phonological",        "VOC / LAB / COR / DOR (consonantal onset)",      "25%"],
  ];

  const hdrs = ["Scheme", "k", "Type", "Description", "Chance"];
  const colW = [1.0, 0.4, 1.5, 4.2, 0.8];
  const colX = [0.4, 1.45, 1.9, 3.45, 7.7];

  // Header row
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.15, w: 9.2, h: 0.38,
    fill: { color: C.navy }, line: { color: C.navy },
  });
  hdrs.forEach((h, i) => {
    s.addText(h, {
      x: colX[i], y: 1.18, w: colW[i], h: 0.3,
      fontSize: 11, bold: true, color: C.white, fontFace: "Calibri", margin: 0,
    });
  });

  rows.forEach((row, ri) => {
    const y = 1.58 + ri * 0.47;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.4, y, w: 9.2, h: 0.43,
      fill: { color: ri % 2 === 0 ? C.light : C.white },
      line: { color: ri % 2 === 0 ? C.light : C.white },
    });
    // Highlight concr4 and phon4
    const highlight = row[0] === "concr4" || row[0] === "phon4";
    row.forEach((val, ci) => {
      s.addText(val, {
        x: colX[ci], y: y + 0.06, w: colW[ci], h: 0.3,
        fontSize: 11, color: highlight ? C.blue : C.dark,
        bold: highlight && ci === 0, fontFace: "Calibri", margin: 0,
      });
    });
    if (highlight) {
      s.addShape(pres.shapes.RECTANGLE, {
        x: 0.4, y, w: 0.06, h: 0.43,
        fill: { color: C.cyan }, line: { color: C.cyan },
      });
    }
  });
  s.addText("★ Highlighted: schemes tested in baseline experiments", {
    x: 0.4, y: 5.0, w: 9.2, h: 0.25,
    fontSize: 10, color: C.gray, italic: true, fontFace: "Calibri", margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 5 — concr4 Neurolinguistic Motivation
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addSlideHeader(s, "concr4 — Neurolinguistic Motivation", "4 categories based on neural substrates");

  const cats = [
    { name: "CONCR",    n: 18, color: "1E5FAD", bg: "D6E4F7", desc: "Physical objects, places\n→ Visual/ventral cortex",    ex: "acqua, casa, macchina" },
    { name: "ACTION",   n: 27, color: "2EC4B6", bg: "D4F1EF", desc: "Action verbs, cognitive processes\n→ Motor/frontal areas",  ex: "andare, fare, mangiare" },
    { name: "STATE",    n: 21, color: "FFB703", bg: "FFF3CD", desc: "Emotions, bodily states, qualities\n→ Limbic circuits",    ex: "amore, paura, felicità" },
    { name: "ABSTRACT", n: 44, color: "E63946", bg: "FADDDF", desc: "Relational concepts, function words\n→ Prefrontal cortex", ex: "forse, quindi, adesso" },
  ];

  cats.forEach((c, i) => {
    const x = 0.35 + (i % 2) * 4.7;
    const y = 1.2 + Math.floor(i / 2) * 2.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 4.45, h: 1.85,
      fill: { color: c.bg }, line: { color: c.color, width: 1.5 }, rectRadius: 0.1,
    });
    // Color badge
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x + 0.12, y: y + 0.12, w: 1.0, h: 0.35,
      fill: { color: c.color }, line: { color: c.color }, rectRadius: 0.06,
    });
    s.addText(c.name, {
      x: x + 0.12, y: y + 0.14, w: 1.0, h: 0.3,
      fontSize: 11, bold: true, color: C.white, fontFace: "Calibri", align: "center", margin: 0,
    });
    s.addText(`${c.n} words`, {
      x: x + 1.2, y: y + 0.15, w: 3.1, h: 0.28,
      fontSize: 12, color: c.color, fontFace: "Calibri", bold: true, margin: 0,
    });
    s.addText(c.desc, {
      x: x + 0.12, y: y + 0.58, w: 4.2, h: 0.6,
      fontSize: 12, color: C.dark, fontFace: "Calibri", margin: 0,
    });
    s.addText(`e.g.: ${c.ex}`, {
      x: x + 0.12, y: y + 1.3, w: 4.2, h: 0.3,
      fontSize: 11, color: C.gray, italic: true, fontFace: "Calibri", margin: 0,
    });
  });

  s.addText("References: Binder et al. 2011 JNeurosci  |  Vigliocco et al. 2014 PsychRev  |  Montefinese et al. 2014", {
    x: 0.4, y: 5.25, w: 9.2, h: 0.22,
    fontSize: 9, color: C.gray, fontFace: "Calibri", align: "center", margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 6 — Deep Learning Architectures
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addSlideHeader(s, "Deep Learning Architectures Tested", "6 end-to-end models on raw EEG signal (no feature engineering)");

  const models = [
    { name: "EEGNet",          type: "Compact CNN",      params: "~1K",   note: "Lightweight baseline — Lawhern 2018",    color: C.blue },
    { name: "ShallowFBCSPNet", type: "Shallow CNN",      params: "~10K",  note: "Based on FBCSP filter bank",             color: C.green },
    { name: "Deep4Net",        type: "Deep CNN",         params: "~300K", note: "4 convolutional blocks",                 color: C.cyan },
    { name: "EEGConformer",    type: "CNN + Transformer",params: "~800K", note: "Patch embedding + self-attention",       color: "9B59B6" },
    { name: "ATCNet",          type: "CNN + TCN",        params: "~200K", note: "Attention temporal convolution network", color: C.yellow },
    { name: "Labram",          type: "Foundation Model", params: "~5M",   note: "Pre-trained on large EEG corpus",        color: C.red },
  ];

  models.forEach((m, i) => {
    const x = 0.35 + (i % 3) * 3.15;
    const y = 1.2 + Math.floor(i / 3) * 1.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 3.0, h: 1.75,
      fill: { color: C.light }, line: { color: m.color, width: 1.5 }, rectRadius: 0.1,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 3.0, h: 0.38,
      fill: { color: m.color }, line: { color: m.color }, rectRadius: 0,
    });
    s.addText(m.name, {
      x: x + 0.1, y: y + 0.06, w: 2.8, h: 0.28,
      fontSize: 13, bold: true, color: C.white, fontFace: "Calibri", align: "center", margin: 0,
    });
    s.addText(m.type, {
      x: x + 0.1, y: y + 0.46, w: 2.8, h: 0.28,
      fontSize: 11, color: m.color, bold: true, fontFace: "Calibri", margin: 0,
    });
    s.addText(`Params: ${m.params}`, {
      x: x + 0.1, y: y + 0.76, w: 2.8, h: 0.25,
      fontSize: 11, color: C.gray, fontFace: "Calibri", margin: 0,
    });
    s.addText(m.note, {
      x: x + 0.1, y: y + 1.05, w: 2.8, h: 0.55,
      fontSize: 11, color: C.dark, fontFace: "Calibri", margin: 0,
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 7 — Baseline Results Subject-Independent
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addSlideHeader(s, "Baseline — Subject-Independent (concr4)", "Train: subjects 01–50  |  Val: 51–60  |  Test: 61–74");

  // Chance level banner
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.4, y: 1.15, w: 9.2, h: 0.42,
    fill: { color: C.yellow, transparency: 80 }, line: { color: C.yellow, width: 1.5 }, rectRadius: 0.06,
  });
  s.addText("Chance Level = 25%  (4 classes — random performance baseline)", {
    x: 0.4, y: 1.2, w: 9.2, h: 0.32,
    fontSize: 13, bold: true, color: C.navy, fontFace: "Calibri", align: "center", margin: 0,
  });

  // Table
  const hdrs = ["Model", "val_bacc", "Epochs", "Time"];
  const colX = [0.4, 3.8, 5.8, 7.5];
  const colW = [3.2, 1.8, 1.8, 2.0];
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.68, w: 9.2, h: 0.38,
    fill: { color: C.navy }, line: { color: C.navy },
  });
  hdrs.forEach((h, i) => {
    s.addText(h, {
      x: colX[i], y: 1.71, w: colW[i], h: 0.3,
      fontSize: 12, bold: true, color: C.white, fontFace: "Calibri", margin: 0,
    });
  });

  const rows = [
    ["ATCNet",          "0.2503", "22", "1.6h"],
    ["Deep4Net",        "0.2503", "41", "2.7h"],
    ["EEGConformer",    "0.2500", "15", "1.1h"],
    ["EEGNet",          "0.2491", "15", "1.0h"],
    ["Labram",          "0.2502", "15", "1.2h"],
    ["ShallowFBCSPNet", "0.2542 ★", "19", "1.3h"],
  ];
  rows.forEach((row, ri) => {
    const y = 2.1 + ri * 0.44;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.4, y, w: 9.2, h: 0.4,
      fill: { color: ri % 2 === 0 ? C.light : C.white },
      line: { color: ri % 2 === 0 ? C.light : C.white },
    });
    row.forEach((val, ci) => {
      s.addText(val, {
        x: colX[ci], y: y + 0.06, w: colW[ci], h: 0.28,
        fontSize: 12, color: val.includes("★") ? C.blue : C.dark,
        bold: val.includes("★"), fontFace: "Calibri", margin: 0,
      });
    });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.4, y: 4.85, w: 9.2, h: 0.48,
    fill: { color: C.red, transparency: 87 }, line: { color: C.red, width: 1 }, rectRadius: 0.06,
  });
  s.addText("All models converge to chance. Early stopping at 15–41 epochs with no improvement.", {
    x: 0.6, y: 4.9, w: 8.8, h: 0.36,
    fontSize: 12, color: C.red, bold: true, fontFace: "Calibri", align: "center", margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 8 — Instance Normalization
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addSlideHeader(s, "Instance Normalization — Bomatter et al. 2024", "Removes per-subject bias from each trial independently");

  // Formula box
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 2.5, y: 1.15, w: 5.0, h: 0.55,
    fill: { color: C.navy }, line: { color: C.navy }, rectRadius: 0.08,
  });
  s.addText("x = (x − μ) / σ    (per trial, per channel)", {
    x: 2.5, y: 1.2, w: 5.0, h: 0.42,
    fontSize: 14, bold: true, color: C.cyan, fontFace: "Consolas", align: "center", margin: 0,
  });

  const hdrs = ["Model", "No Norm", "With Norm", "Δ"];
  const colX = [0.4, 3.5, 5.5, 7.8];
  const colW = [3.0, 1.8, 1.8, 1.8];
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.85, w: 9.2, h: 0.38,
    fill: { color: C.navy }, line: { color: C.navy },
  });
  hdrs.forEach((h, i) => {
    s.addText(h, {
      x: colX[i], y: 1.88, w: colW[i], h: 0.3,
      fontSize: 12, bold: true, color: C.white, fontFace: "Calibri", margin: 0,
    });
  });

  const rows = [
    ["ATCNet",          "0.2502", "0.2499", "−0.0003"],
    ["Deep4Net",        "0.2504", "0.2500", "−0.0004"],
    ["EEGConformer",    "0.2500", "0.2500", " 0.0000"],
    ["EEGNet",          "0.2491", "0.2517", "+0.0026"],
    ["Labram",          "0.2505", "0.2439", "−0.0066"],
    ["ShallowFBCSPNet", "0.2542", "0.2525", "−0.0017"],
  ];
  rows.forEach((row, ri) => {
    const y = 2.27 + ri * 0.42;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.4, y, w: 9.2, h: 0.38,
      fill: { color: ri % 2 === 0 ? C.light : C.white },
      line: { color: ri % 2 === 0 ? C.light : C.white },
    });
    row.forEach((val, ci) => {
      const isPos = val.startsWith("+");
      const isNeg = val.startsWith("−") && val !== "−0.0003" && val !== "−0.0004" && val !== "−0.0017";
      s.addText(val, {
        x: colX[ci], y: y + 0.05, w: colW[ci], h: 0.28,
        fontSize: 12, fontFace: "Calibri", margin: 0,
        color: isPos ? C.green : isNeg ? C.red : C.dark,
      });
    });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.4, y: 4.88, w: 9.2, h: 0.45,
    fill: { color: C.red, transparency: 87 }, line: { color: C.red, width: 1 }, rectRadius: 0.06,
  });
  s.addText("Instance Norm produces no significant improvement. The domain shift problem is deeper than normalization.", {
    x: 0.6, y: 4.92, w: 8.8, h: 0.35,
    fontSize: 12, color: C.red, bold: true, fontFace: "Calibri", align: "center", margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 9 — Why at Chance?
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addSlideHeader(s, "Why Do All Models Stay at Chance?", "Root cause analysis");

  const reasons = [
    { title: "ε² = 0.85\nInter-subject variance", desc: "85% of EEG variance is\ndue to the subject, not\nthe imagined word", color: C.red },
    { title: "ε² = 0.03\nInter-word variance",   desc: "Only 3% of variance is\nlinked to the specific\nimagined word",      color: C.gray },
    { title: "Δ acc ≈ 0\nInstance Norm",         desc: "Simple normalization\ncannot solve deep\ncross-subject domain shift", color: C.yellow },
  ];

  reasons.forEach((r, i) => {
    const x = 0.6 + i * 3.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 1.2, w: 2.7, h: 3.2,
      fill: { color: r.color, transparency: 91 }, line: { color: r.color, width: 2 }, rectRadius: 0.12,
    });
    s.addText(r.title, {
      x: x + 0.1, y: 1.35, w: 2.5, h: 1.2,
      fontSize: 18, bold: true, color: r.color, fontFace: "Calibri", align: "center", margin: 0,
    });
    s.addShape(pres.shapes.LINE, {
      x: x + 0.4, y: 2.65, w: 1.9, h: 0,
      line: { color: r.color, width: 1, dashType: "dash" }
    });
    s.addText(r.desc, {
      x: x + 0.1, y: 2.8, w: 2.5, h: 1.4,
      fontSize: 12, color: C.dark, fontFace: "Calibri", align: "center", margin: 0,
    });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.4, y: 4.55, w: 9.2, h: 0.68,
    fill: { color: C.navy }, line: { color: C.navy }, rectRadius: 0.08,
  });
  s.addText("→  Domain adaptation required: MMD, CORAL, Adversarial Training, or Contrastive Learning", {
    x: 0.6, y: 4.62, w: 8.8, h: 0.52,
    fontSize: 14, color: C.cyan, bold: true, fontFace: "Calibri", align: "center", margin: 0,
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 10 — Roadmap
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.white };
  addSlideHeader(s, "Roadmap", "From baseline to hypergraph");

  const phases = [
    {
      phase: "IMMEDIATE",
      color: C.red,
      items: [
        "Subject-specific baseline (leave-one-session-out)",
        "Validate: does EEG contain decodable information?",
      ],
    },
    {
      phase: "SHORT TERM",
      color: C.yellow,
      items: [
        "Domain adaptation: MMD, CORAL",
        "Adversarial cross-subject training",
        "Contrastive subject-invariant learning (Shen 2022)",
      ],
    },
    {
      phase: "MEDIUM TERM",
      color: C.green,
      items: [
        "Graph Attention Networks (GAT)",
        "Spatial electrode structure",
        "Multi-head attention on EEG channels",
      ],
    },
    {
      phase: "FINAL TARGET",
      color: C.blue,
      items: [
        "Hypergraph Neural Networks (DHSLP/DHSLF)",
        "Li et al. 2025 — target: >40% on 4–5 classes",
        "Dynamic hyperedge construction",
      ],
    },
  ];

  phases.forEach((p, i) => {
    const x = 0.35 + (i % 2) * 4.7;
    const y = 1.2 + Math.floor(i / 2) * 2.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 4.45, h: 1.85,
      fill: { color: C.light }, line: { color: p.color, width: 2 }, rectRadius: 0.1,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.45, h: 0.38,
      fill: { color: p.color }, line: { color: p.color },
    });
    s.addText(p.phase, {
      x: x + 0.1, y: y + 0.06, w: 4.25, h: 0.26,
      fontSize: 12, bold: true, color: C.white, fontFace: "Calibri",
      charSpacing: 2, align: "center", margin: 0,
    });
    p.items.forEach((item, ii) => {
      s.addText([{ text: "▸  " + item }], {
        x: x + 0.15, y: y + 0.48 + ii * 0.42, w: 4.1, h: 0.38,
        fontSize: 11.5, color: C.dark, fontFace: "Calibri", margin: 0,
      });
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// WRITE
// ═══════════════════════════════════════════════════════════════════════════
pres.writeFile({ fileName: "EN_EEG_Project_Status_Mar2026.pptx" })
  .then(() => console.log("Done: EN_EEG_Project_Status_Mar2026.pptx"))
  .catch(e => console.error(e));
