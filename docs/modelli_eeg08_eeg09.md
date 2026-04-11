# Modelli EEG_08 e EEG_09 — Documentazione Tecnica

> Documento di riferimento per comprendere le architetture implementate,
> le scelte progettuali e le loro motivazioni scientifiche.

---

## Contesto: perché i grafi?

Il segnale EEG non è una sequenza 1D e non è un'immagine 2D — è un insieme di
**59 serie temporali** registrate da sensori disposti sulla testa, tra cui esistono
relazioni spaziali (posizione degli elettrodi) e funzionali (correlazione dell'attività).

Una rete convoluzionale standard tratta i canali come indipendenti o li processa
su una griglia regolare che non esiste (il casco non è una griglia). I **Graph Neural
Networks** permettono di modellare esplicitamente le relazioni tra canali attraverso
un grafo, dove:

- **Nodo** = un elettrodo EEG
- **Arco** = una relazione tra due elettrodi (spaziale o funzionale)
- **Feature del nodo** = la serie temporale di quell'elettrodo

---

## Il grafo: costruzione condivisa tra EEG_08 e EEG_09

### Metodo: k-NN sul PCC

Per costruire il grafo si usa la **Pearson Correlation Coefficient (PCC)** tra canali,
calcolata mediando su 1000 trial del training set.

Per ogni elettrodo `i`:
```
collega i ai K=6 elettrodi con cui ha |PCC| più alta
```

Questo approccio è chiamato **k-NN funzionale** ed è ispirato a:
> Lun et al. 2022, *GCNs-Net*, IEEE TNSRE — arXiv:2006.08924

**Perché k-NN invece di una soglia fissa?**
Con una soglia fissa (es. |PCC| > 0.5), se un elettrodo ha poca correlazione con tutti
gli altri, rimane isolato — nodo con grado zero. Il k-NN garantisce che ogni nodo abbia
almeno K vicini, mantenendo la connettività del grafo.

### Caratteristica critica: il grafo è statico

Il grafo viene calcolato **una volta sola** sulla media di 1000 trial di training e poi
usato **invariato** per tutti i trial di tutti i soggetti. Questo è un limite importante:
il grafo rappresenta la connettività funzionale *media* della popolazione di training,
non quella specifica di un soggetto o di un trial.

```
Grafo risultante: 59 nodi, 420 archi, grado medio 7.1
```

---

## EEG_08 — TemporalChebGCN

**Paper di riferimento**: Lun et al. 2022, *GCNs-Net* (PCC + ChebConv)

### Architettura completa

```
Input: (59 nodi × 384 campioni)
         │
         ▼
 ┌────────────────────────┐
 │    Temporal Encoder    │
 │    (1D CNN per-nodo)   │
 └────────────────────────┘
         │
         ▼ (59 nodi × 64 embedding)
 ┌────────────────────────┐
 │  ChebConv(K=2) + BN    │   layer 1: 64 → 128
 │  ChebConv(K=2) + BN    │   layer 2: 128 → 128
 └────────────────────────┘
         │
         ▼ global_mean_pool
 ┌────────────────────────┐
 │  MLP classifier        │   128 → 64 → N_CLASSES
 └────────────────────────┘
```

---

### Componente 1: Temporal Encoder

```python
Conv1d(1→32, kernel=25, stride=2)  # 384 → 192 campioni
BatchNorm + ELU
Conv1d(32→64, kernel=10, stride=2) # 192 → 96 campioni
BatchNorm + ELU
AdaptiveAvgPool1d(4)               # 96 → 4 campioni
Flatten + Linear(256→64) + ELU    # → vettore 64
```

**Scopo**: comprimere la serie temporale di ogni elettrodo (384 campioni, ~1.5s)
in un vettore compatto di 64 numeri.

**I pesi sono condivisi** tra tutti i 59 nodi: la rete impara una trasformazione
"universale" applicata allo stesso modo a ogni elettrodo. Questo riduce il numero
di parametri e forza il modello a imparare una rappresentazione generalizzabile.

**Perché il trial intero?** L'imagined speech si distribuisce sull'intera finestra
temporale (~1.5s). Tagliare il segnale o processarlo a finestre corte rischierebbe
di perdere informazione semantica che emerge gradualmente durante il processo cognitivo.

**Nota**: il Temporal Encoder non è presente nel paper GCNs-Net — è un contributo
originale di questo progetto, progettato per il task di imagined speech.

---

### Componente 2: ChebConv (Chebyshev Convolution)

ChebConv è una **convoluzione nel dominio spettrale del grafo**.

#### La matematica

La convoluzione spettrale classica sul grafo richiede la decomposizione agli autovettori
del Laplaciano L, operazione costosa O(n³). ChebConv la approssima usando polinomi
di Chebyshev:

```
y = Σ_{k=0}^{K} θ_k · T_k(L̃) · x
```

Dove:
- `L̃ = 2L/λ_max - I` = Laplaciano riscalato nell'intervallo [-1, 1]
- `T_k` = polinomio di Chebyshev di ordine k, definito ricorsivamente:
  - `T_0(x) = 1`
  - `T_1(x) = x`
  - `T_k(x) = 2x·T_{k-1}(x) - T_{k-2}(x)`
- `θ_k` = parametri apprendibili (i "pesi del filtro")
- `K = 2` = ordine del polinomio (dal paper GCNs-Net)

#### Interpretazione pratica

Con K=2, ogni nodo aggrega informazioni da:
- Se stesso (hop 0)
- I vicini diretti (hop 1)
- I vicini dei vicini (hop 2)

È analogo a una convoluzione su immagini con campo recettivo 3×3, ma applicata
alla struttura irregolare del grafo degli elettrodi.

#### Limite chiave

I pesi `θ_k` sono **gli stessi per tutti gli archi**. ChebConv non distingue tra
un arco forte (|PCC|=0.9) e uno debole (|PCC|=0.3) — la distinzione è binaria:
l'arco c'è o non c'è. L'intensità della connessione non viene usata nel calcolo,
solo la topologia del grafo.

---

### Varianti implementate in EEG_08

| Modello | Architettura | Note |
|---------|-------------|------|
| `ChebGCN_2L` | 2 layer ChebConv | Baseline |
| `ChebGCN_3L` | 3 layer ChebConv | Più profondo |
| `ChebGCNSkip` | 2 layer + skip connection | Residual connection |

---

### Risultati e diagnosi

Val_bacc ≈ 25% (chance level) per tutti i modelli. Motivazione: ChebConv usa
lo stesso grafo medio per tutti i soggetti e non ha nessun meccanismo per
gestire la dominanza della variabilità inter-soggetto (ε²=0.85).

---

## EEG_09 — Domain Adversarial GAT

**Paper di riferimento**: Xu et al. 2023, *DAGAM*, arXiv:2202.12948

### Il problema che EEG_09 vuole risolvere

Dall'analisi esplorativa (EEG_00/01):

```
ε²(soggetto) = 0.85  → 85% della varianza EEG è identità del soggetto
ε²(parola)   = 0.03  → solo 3% è contenuto semantico della parola
```

Qualsiasi modello che non contrasta attivamente questa dominanza imparerà
a classificare il soggetto, non la parola. La soluzione è **forzare la rete
a imparare feature indipendenti dal soggetto**.

---

### Architettura completa

```
Input: (59 nodi × 384 campioni)
         │
         ▼
 ┌────────────────────────┐
 │    Temporal Encoder    │   invariato da EEG_08
 │    (1D CNN per-nodo)   │   384 → 64
 └────────────────────────┘
         │
         ▼ (59 nodi × 64 embedding)
 ┌────────────────────────┐
 │  GATConv(8 heads) × 2  │   64 → 8×64 → 64
 └────────────────────────┘
         │
         ▼ global_mean_pool
 ┌────────────────────────┐
 │   Shared Embedding h   │   (batch, 64)
 └──────┬─────────────────┘
        │                        │
        ▼                        ▼ ── GRL(α) ──
 Task Classifier           Subject Discriminator
 h → 4 classi parola       h_rev → 50 soggetti training
 L_task                    L_subj
        │
 L_tot = L_task + λ_adv · L_subj
```

---

### Componente 1: GATConv (Graph Attention Network)

**Paper**: Veličković et al. 2018, ICLR — arXiv:1710.10903

#### La matematica

L'aggiornamento del nodo `i` è una media pesata delle feature dei vicini:

```
h_i' = Σ_{j ∈ N(i) ∪ {i}} α_ij · W · h_j
```

Dove il peso di attenzione `α_ij` è calcolato dinamicamente:

```
e_ij = LeakyReLU( a^T · [W·h_i || W·h_j] )
α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_{k ∈ N(i)} exp(e_ik)
```

- `W` = matrice di trasformazione lineare (apprendibile)
- `a` = vettore di attenzione (apprendibile)
- `||` = concatenazione
- `softmax` normalizza i pesi su tutti i vicini di `i` → somma a 1

#### Multi-head attention (GAT_HEADS=8)

Si eseguono 8 meccanismi di attenzione indipendenti in parallelo:

```
h_i' = ||_{k=1}^{8} σ( Σ_{j ∈ N(i)} α_ij^k · W^k · h_j )
```

Ogni "testa" impara a prestare attenzione ad aspetti diversi delle relazioni
tra elettrodi. Il risultato è la concatenazione dei 8 output → 8×64=512 dim,
poi ridotto a 64 nel layer finale (media invece di concatenazione).

#### Differenza rispetto a ChebConv

| | ChebConv | GATConv |
|--|----------|---------|
| Pesi archi | Fissi (0 o 1) | Dinamici (dipendono dalle feature) |
| Adattabilità | Stessa per tutti i trial | Cambia per ogni trial |
| Interpretabilità | Filtro spettrale | Attention scores per arco |
| Complessità | O(K · \|E\| · d) | O(\|E\| · d) per testa × n_heads |

**In pratica**: due trial dello stesso soggetto ma parole diverse riceveranno
pesi di attenzione diversi sugli stessi archi. Il modello può imparare
"per questa parola, la connessione FP1-F3 è importante; per quest'altra, è Cz-Pz".

---

### Componente 2: Gradient Reversal Layer (GRL)

**Paper**: Ganin et al. 2016, JMLR — *Domain-Adversarial Training of Neural Networks*

#### L'idea

La rete viene divisa in tre parti:
1. **Feature extractor** (Temporal Encoder + GAT): produce embedding `h`
2. **Task classifier**: usa `h` per classificare la parola → minimizza `L_task`
3. **Subject discriminator**: usa `h` (attraverso GRL) per classificare il soggetto → minimizza `L_subj`

Il GRL è un layer che durante il **forward pass** è trasparente:
```
output = input   (non fa nulla)
```

Durante il **backward pass** inverte il gradiente:
```
∂L/∂input = -α · ∂L/∂output
```

#### Il gioco minimax

```
Feature extractor:  minimizza L_task  +  massimizza L_subj
                    (vuole classificare bene la parola)
                    (vuole confondere il discriminatore di soggetto)

Discriminatore:     minimizza L_subj
                    (vuole classificare bene il soggetto)
```

Questo crea un **gioco competitivo**: il discriminatore diventa sempre più bravo
a identificare il soggetto, ma il feature extractor viene contemporaneamente
spinto a rimuovere dall'embedding `h` qualsiasi informazione sull'identità
del soggetto. All'equilibrio, `h` dovrebbe contenere solo informazione utile
per la parola e non per il soggetto.

#### La loss totale

```
L_tot = L_task + λ_adv · L_subj
```

Con `λ_adv = 0.1`: il termine adversariale pesa il 10% della loss totale.
Il GRL fa sì che `L_subj` abbia effetto opposto sul feature extractor.

#### Lo schedule alpha (DANN)

`α` (intensità del GRL) non è costante ma segue uno schedule sigmoidale:

```
p = epoch / max_epochs         ∈ [0, 1]
α = 2 / (1 + exp(-10·p)) - 1  ∈ [0, 1]
```

Nei primi epoch α ≈ 0: la rete impara a fare il task senza pressione adversariale.
Progressivamente α → 1: la pressione verso l'invarianza al soggetto aumenta.

**Motivazione**: se si applica il GRL a piena intensità dall'inizio, il feature
extractor non impara mai a fare il task perché viene immediatamente destabilizzato.

---

### Il discriminatore soggetto durante validazione e test

Il discriminatore ha **50 output** (i soggetti di training, ID 0-49). I soggetti
di validation (50-59) e test (60-73) non esistono come classi nel discriminatore.

**Soluzione**: durante validation e test, `α = 0.0`. Il GRL non inverte nessun
gradiente e il discriminatore è effettivamente disabilitato. Si valuta solo
la classificazione della parola su soggetti mai visti.

Questo è corretto: il discriminatore serve *solo* come pressione durante il
training per rimuovere informazione sul soggetto dall'embedding. A inferenza
non è necessario.

---

### Varianti implementate in EEG_09

| Modello | GATConv | Adversarial | Note |
|---------|---------|-------------|------|
| `GAT_2L` | 2 layer | ✗ | Baseline: il GAT da solo aiuta? |
| `GAT_2L_ADV` | 2 layer | ✓ | DAGAM-style |
| `GAT_3L_ADV` | 3 layer | ✓ | Versione più profonda |

`GAT_2L` serve a isolare il contributo della domain adaptation: se `GAT_2L_ADV`
va meglio di `GAT_2L` ma entrambi vanno meglio di EEG_08 (ChebGCN), allora
sia il cambio di convolution che la domain adaptation contribuiscono. Se solo
`GAT_2L_ADV` migliora, la domain adaptation è il fattore chiave.

---

## Confronto diretto EEG_08 vs EEG_09

| Caratteristica | EEG_08 ChebGCN | EEG_09 GAT + GRL |
|----------------|---------------|-----------------|
| Grafo | PCC k-NN statico | PCC k-NN statico (identico) |
| Convoluzione | Spettrale (ChebConv K=2) | Attention-based (GATConv 8 heads) |
| Pesi archi | Fissi (topologia binaria) | Dinamici (attention per trial) |
| Adattabilità al trial | Nessuna | Sì (attention scores) |
| Gestione soggetto | Ignorata | Contrastata con GRL |
| Loss | CE(parola) | CE(parola) + λ·CE(soggetto, reversed) |
| Parametri aggiuntivi | — | Discriminatore soggetto (~4K param) |
| Ipotesi centrale | Il grafo cattura la semantica | L'invarianza al soggetto libera il segnale semantico |

---

## Limitazioni comuni a entrambi

1. **Grafo statico**: né EEG_08 né EEG_09 hanno un grafo specifico per soggetto
   o per trial. GAT ha attenzione dinamica *sui pesi* degli archi esistenti,
   ma gli archi stessi sono sempre gli stessi. Un grafo dinamico sarebbe il
   passo successivo (DDGHFNet, 2025).

2. **Nessuna normalizzazione per-trial (InstanceNorm)**: le differenze di ampiezza
   tra soggetti entrano direttamente nel Temporal Encoder. Bomatter et al. 2024
   dimostrano che aggiungere `nn.InstanceNorm1d` prima del temporal encoder
   riduce significativamente la variabilità inter-soggetto con una sola riga di codice.

3. **Split hard subject-independent**: 50 soggetti di training potrebbero non
   coprire sufficientemente la variabilità della distribuzione test (14 soggetti).
   Cross-validation subject-independent (LOSO) darebbe una stima più robusta.

4. **50 soggetti nel discriminatore**: il discriminatore impara l'identità di 50
   soggetti specifici. Non generalizza a nuovi soggetti — ma non deve farlo.
   Il suo scopo è solo creare pressione durante il training.

---

## Collocazione nella narrativa della tesi

```
EEG_00/01  Feature manuali + ANOVA
           → Il problema è nei dati: ε²(soggetto)=0.85 domina

EEG_05/07  EEGNet, EEG Conformer (Braindecode)
           → Deep learning end-to-end non basta da solo: chance level

EEG_08     ChebGCN (grafo PCC statico)
           → I grafi spaziali senza domain adaptation: ancora chance

EEG_09     GAT + Domain Adversarial (DAGAM-style)
           → Forziamo l'invarianza al soggetto: vediamo se emerge il segnale

EEG_10     Hypergraph Neural Networks
           → Relazioni di ordine superiore tra elettrodi (obiettivo tesi)
```

Ogni step fallisce per un motivo preciso che motiva il passo successivo.
La domanda finale della tesi diventa:

> *Le Hypergraph Neural Networks con domain adaptation riescono a estrarre
> il segnale semantico dell'imagined speech nonostante la dominanza
> della variabilità inter-soggetto?*
