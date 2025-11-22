# Guida Completa alle Metriche delle Feature EEG

Questo documento fornisce spiegazioni dettagliate di tutte le metriche calcolate dal sistema di estrazione delle feature.

---

## Indice
1. [Feature Temporali (13 metriche)](#feature-temporali)
2. [Feature Spettrali (22 metriche)](#feature-spettrali)
3. [Feature Funzionali (6 metriche)](#feature-funzionali)

---

## Feature Temporali

Le feature temporali catturano caratteristiche statistiche e nel dominio del tempo del segnale EEG.

### 1. Misure Statistiche di Base

#### `temp_mean` - Media del Segnale
- **Formula**: `μ = (1/N) Σ x(n)`
- **Unità**: µV (microvolt)
- **Descrizione**: Ampiezza media del segnale nell'epoca
- **Interpretazione**: 
  - Dovrebbe essere vicina a 0 per EEG correttamente preprocessato
  - Deviazioni elevate possono indicare drift della baseline o artefatti
- **Range**: Tipicamente -5 a +5 µV

#### `temp_std` - Deviazione Standard
- **Formula**: `σ = sqrt((1/N) Σ (x(n) - μ)²)`
- **Unità**: µV
- **Descrizione**: Misura della variabilità del segnale
- **Interpretazione**:
  - Valori più alti → segnale più variabile
  - Valori più bassi → segnale più stabile/piatto
  - Correlata alla potenza del segnale
- **Range**: 1-50 µV (tipico per EEG)

#### `temp_var` - Varianza
- **Formula**: `σ² = (1/N) Σ (x(n) - μ)²`
- **Unità**: µV²
- **Descrizione**: Quadrato della deviazione standard
- **Interpretazione**: 
  - Direttamente proporzionale alla potenza del segnale
  - Varianza maggiore → attività maggiore
- **Range**: 1-2500 µV²

#### `temp_min` / `temp_max` - Valori Minimo e Massimo
- **Unità**: µV
- **Descrizione**: Valori estremi di ampiezza nell'epoca
- **Interpretazione**:
  - Utile per rilevare artefatti (spike, saturazione)
  - Valori estremamente alti (>100 µV) spesso indicano artefatti

#### `temp_range` - Range del Segnale
- **Formula**: `Range = max - min`
- **Unità**: µV
- **Descrizione**: Estensione totale dell'ampiezza
- **Interpretazione**: 
  - Indica il range dinamico del segnale
  - Range molto ampi possono indicare artefatti

#### `temp_ptp` - Ampiezza Picco-Picco
- **Formula**: Uguale al range
- **Unità**: µV
- **Descrizione**: Differenza tra il punto più alto e più basso
- **Interpretazione**: 
  - Misura comune nell'EEG clinico
  - EEG tipico: 10-100 µV picco-picco

---

### 2. Momenti Statistici di Ordine Superiore

#### `temp_skewness` - Asimmetria (Skewness)
- **Formula**: `γ₁ = E[(X-μ)³] / σ³`
- **Unità**: Adimensionale
- **Descrizione**: Misura dell'asimmetria della distribuzione
- **Interpretazione**:
  - **Skewness = 0**: Distribuzione simmetrica (normale)
  - **Skewness > 0**: Asimmetria a destra (coda si estende a destra, più outlier positivi)
  - **Skewness < 0**: Asimmetria a sinistra (coda si estende a sinistra, più outlier negativi)
  - Nell'EEG: spike acuti creano skewness positiva
- **Range**: Tipicamente -2 a +2 per EEG

#### `temp_kurtosis` - Curtosi (Kurtosis)
- **Formula**: `γ₂ = E[(X-μ)⁴] / σ⁴ - 3` (eccesso di curtosi)
- **Unità**: Adimensionale
- **Descrizione**: Misura della "pesantezza delle code" o del picco
- **Interpretazione**:
  - **Kurtosis = 0**: Distribuzione normale (mesocurtica)
  - **Kurtosis > 0**: Code pesanti, picco acuto (leptocurtica) - più outlier
  - **Kurtosis < 0**: Code leggere, picco piatto (platicurtica) - meno outlier
  - Curtosi elevata nell'EEG spesso indica artefatti o attività epilettiforme
- **Range**: Tipicamente -1 a +5 per EEG pulito

---

### 3. Energia e Attività del Segnale

#### `temp_rms` - Root Mean Square (RMS)
- **Formula**: `RMS = sqrt((1/N) Σ x(n)²)`
- **Unità**: µV
- **Descrizione**: Media quadratica del segnale
- **Interpretazione**:
  - Rappresenta la "forza" o energia complessiva del segnale
  - Correlata alla potenza (RMS² = potenza media)
  - RMS più alto → attività neurale maggiore
- **Range**: 5-50 µV tipico per EEG
- **Rilevanza clinica**: Usato per quantificare attività convulsiva, stadi del sonno

---

### 4. Dinamiche del Segnale

#### `temp_zcr` - Zero Crossing Rate (Tasso di Attraversamento dello Zero)
- **Formula**: `ZCR = (1/N) Σ |sign(x(n)) - sign(x(n-1))|`
- **Unità**: Attraversamenti per campione
- **Descrizione**: Frequenza con cui il segnale attraversa l'ampiezza zero
- **Interpretazione**:
  - **ZCR alto**: Oscillazioni veloci, contenuto ad alta frequenza
  - **ZCR basso**: Oscillazioni lente, contenuto a bassa frequenza
  - Stima approssimativa della frequenza dominante
- **Range**: 0.0 a 0.5 (normalizzato)
- **Esempio**: 
  - Onde delta lente: ZCR ≈ 0.01-0.02
  - Beta/gamma veloce: ZCR ≈ 0.1-0.2

---

### 5. Parametri di Hjorth

Denominati dopo Bo Hjorth (1970), questi parametri descrivono la complessità del segnale.

#### `temp_hjorth_activity` - Attività di Hjorth
- **Formula**: `Activity = var(signal)`
- **Unità**: µV²
- **Descrizione**: Varianza del segnale (potenza)
- **Interpretazione**:
  - Rappresenta la potenza totale nel segnale
  - Attività maggiore → attivazione neurale più forte
  - Equivalente a `temp_var`
- **Uso clinico**: Monitoraggio profondità sedazione, rilevamento convulsioni

#### `temp_hjorth_mobility` - Mobilità di Hjorth
- **Formula**: `Mobility = sqrt(var(dx/dt) / var(x))`
- **Unità**: Adimensionale (o simile a Hz)
- **Descrizione**: Rappresenta la "frequenza media" - quanto rapidamente cambia il segnale
- **Interpretazione**:
  - **Mobilità maggiore**: Il segnale cambia rapidamente (frequenze più alte)
  - **Mobilità minore**: Il segnale cambia lentamente (frequenze più basse)
  - Proporzionale alla deviazione standard dello spettro di potenza
- **Range**: Tipicamente 1-20 per EEG
- **Esempio**:
  - Attività delta lenta: Mobility ≈ 1-3
  - Attività gamma veloce: Mobility ≈ 10-20

#### `temp_hjorth_complexity` - Complessità di Hjorth
- **Formula**: `Complexity = Mobility(dx/dt) / Mobility(x)`
- **Unità**: Adimensionale
- **Descrizione**: Misura di quanto il segnale assomiglia a un'onda sinusoidale pura
- **Interpretazione**:
  - **Complexity = 1**: Onda sinusoidale pura (frequenza singola)
  - **Complexity > 1**: Componenti di frequenza multiple
  - **Complessità maggiore**: Segnale più irregolare, caotico
  - Indica l'ampiezza di banda del segnale
- **Range**: Tipicamente 1.1-2.5 per EEG
- **Significato clinico**: 
  - Aumenta durante le convulsioni
  - Diminuisce durante il sonno
  - Indicatore di carico cognitivo

---

## Feature Spettrali

Le feature spettrali sono estratte dalla Densità Spettrale di Potenza (PSD) calcolata usando il metodo di Welch.

### 1. Potenze di Banda (Assolute)

I segnali EEG sono divisi in bande di frequenza standard basate sul significato neurofisiologico.

#### `spec_delta` - Potenza Banda Delta (1-4 Hz)
- **Unità**: µV²
- **Correlati Fisiologici**:
  - **Sonno profondo** (Stadio 3-4 NREM)
  - **Incoscienza** (anestesia, coma)
  - **Lesioni cerebrali** (patologico)
- **Ampiezza Tipica**: Alta (30-200 µV)
- **Distribuzione Spaziale**: Regioni frontali
- **Significato Clinico**:
  - Aumenta con sonnolenza e profondità del sonno
  - Patologicamente alto nell'encefalopatia
- **Durante Compiti**: Solitamente soppressa durante cognizione attiva

#### `spec_theta` - Potenza Banda Theta (4-8 Hz)
- **Unità**: µV²
- **Correlati Fisiologici**:
  - **Codifica e recupero della memoria** (theta ippocampale)
  - **Sonnolenza**, sonno leggero
  - **Meditazione**, rilassamento profondo
  - **Controllo cognitivo** (theta frontale mediana)
  - **Monitoraggio degli errori**
- **Ampiezza Tipica**: Moderata (10-50 µV)
- **Distribuzione Spaziale**: Fronto-mediana (Fz, Cz) durante compiti; temporale durante memoria
- **Significato Clinico**:
  - Aumenta nell'ADHD
  - Burst theta frontale nel decision-making
- **Durante Linguaggio Immaginato**: Può aumentare con carico di memoria di lavoro

#### `spec_alpha` - Potenza Banda Alpha (8-13 Hz)
- **Unità**: µV²
- **Correlati Fisiologici**:
  - **Veglia rilassata** (occhi chiusi)
  - **Cortical idling** (inibizione del processamento attivo)
  - **Soppressione visiva**
  - **Modulazione dell'attenzione**
- **Ampiezza Tipica**: Massima nel riposo da svegli (20-60 µV)
- **Distribuzione Spaziale**: Occipitale (O1, O2) - corteccia visiva
- **Varianti**:
  - **α1 (8-10 Hz)**: Alpha inferiore, più cognitiva
  - **α2 (10-13 Hz)**: Alpha superiore, più percettiva
- **Significato Clinico**:
  - **Alpha blocking**: Scompare quando gli occhi si aprono o durante attività mentale
  - Ridotta nella demenza
- **Durante Compiti**: Tipicamente si sopprime (ERD - Desincronizzazione Evento-Correlata)

#### `spec_beta` - Potenza Banda Beta (13-30 Hz)
- **Unità**: µV²
- **Correlati Fisiologici**:
  - **Pensiero attivo** e concentrazione
  - **Pianificazione ed esecuzione motoria**
  - **Ansia** e arousal
  - **Processamento sensomotorio**
- **Ampiezza Tipica**: Bassa-moderata (5-20 µV)
- **Distribuzione Spaziale**: Corteccia sensomotoria (C3, C4)
- **Sub-bande**:
  - **β1 (13-20 Hz)**: Beta basso, controllo motorio
  - **β2 (20-30 Hz)**: Beta alto, allerta, ansia
- **Significato Clinico**:
  - Aumenta con ansia, stress
  - Beta rebound dopo movimento
  - Beta eccessivo nell'insonnia
- **Durante Linguaggio Immaginato**: Aumenta con impegno cognitivo, pianificazione del linguaggio

#### `spec_gamma` - Potenza Banda Gamma (30-45 Hz)
- **Unità**: µV²
- **Correlati Fisiologici**:
  - **Binding sensoriale** (integrazione di caratteristiche)
  - **Attenzione** e coscienza
  - **Mantenimento memoria di lavoro**
  - **Processamento linguaggio**
  - **Funzione cognitiva ad alto livello**
- **Ampiezza Tipica**: Molto bassa (2-10 µV)
- **Distribuzione Spaziale**: Dipendente dal compito, spesso parietale/frontale
- **Significato Clinico**:
  - Ridotta in schizofrenia, autismo
  - Anormale nell'epilessia
  - Correla con livello di coscienza
- **Durante Linguaggio Immaginato**: Può mostrare aumenti correlati al compito
- **Nota**: Gamma superiore (>60 Hz) può contenere artefatti muscolari

---

### 2. Potenze di Banda (Relative)

#### `spec_delta_rel`, `spec_theta_rel`, `spec_alpha_rel`, `spec_beta_rel`, `spec_gamma_rel`
- **Formula**: `Potenza Relativa = Potenza Banda / Potenza Totale`
- **Unità**: Adimensionale (proporzione)
- **Range**: 0 a 1 (somma ≈ 1 tra tutte le bande)
- **Descrizione**: Potenza di banda normalizzata indipendente dall'ampiezza assoluta
- **Vantaggi**:
  - **Riduce la variabilità inter-soggetto** (differente spessore del cranio, impedenza elettrodi)
  - **Migliore per la classificazione** (più stabile tra sessioni)
  - **Confrontabile tra diversi amplificatori/impostazioni**
- **Interpretazione**: Rappresenta la composizione spettrale piuttosto che la forza assoluta

---

### 3. Potenza Totale

#### `spec_total_power` - Potenza Spettrale Totale (1-45 Hz)
- **Formula**: `∫₁⁴⁵ PSD(f) df`
- **Unità**: µV²
- **Descrizione**: Potenza totale del segnale attraverso tutte le frequenze analizzate
- **Interpretazione**:
  - Potenza totale maggiore → più attività neurale o artefatti
  - Può variare notevolmente tra soggetti ed elettrodi
  - Influenzata dalla qualità degli elettrodi, conduttanza dello scalpo

---

### 4. Rapporti tra Potenze di Banda

Questi rapporti catturano relazioni tra bande di frequenza e sono clinicamente significativi.

#### `spec_alpha_beta_ratio` - Rapporto Alpha/Beta
- **Formula**: `α/β = P_alpha_rel / P_beta_rel`
- **Unità**: Adimensionale
- **Descrizione**: Bilancio tra rilassamento e attivazione
- **Interpretazione**:
  - **Rapporto alto (>1)**: Rilassato, basso arousal, idling
  - **Rapporto basso (<1)**: Allerta, processamento attivo, aroused
- **Applicazioni Cliniche**:
  - Diagnosi ADHD (tipicamente rapporto più basso)
  - Target di training neurofeedback
  - Monitoraggio meditazione (aumenta)
- **Range**: 0.5-3.0 tipico

#### `spec_theta_alpha_ratio` - Rapporto Theta/Alpha
- **Formula**: `θ/α = P_theta_rel / P_alpha_rel`
- **Unità**: Adimensionale
- **Descrizione**: Indicatore di affaticamento cognitivo e stato attentivo
- **Interpretazione**:
  - **Rapporto alto**: Sonnolenza, affaticamento, bassa vigilanza
  - **Rapporto basso**: Allerta, attenzione focalizzata
- **Applicazioni Cliniche**:
  - Rilevamento sonnolenza (guida, operatori)
  - ADHD (rapporto elevato)
  - Predizione inizio sonno
- **Range**: 0.3-2.0 tipico

#### `spec_theta_beta_ratio` - Rapporto Theta/Beta
- **Formula**: `θ/β = P_theta_rel / P_beta_rel`
- **Unità**: Adimensionale
- **Descrizione**: Biomarcatore classico dell'ADHD
- **Interpretazione**:
  - **Rapporto alto (>2.0)**: Possibile ADHD, scarsa regolazione dell'attenzione
  - **Rapporto normale (0.5-2.0)**: Controllo dell'attenzione tipico
- **Applicazioni Cliniche**:
  - Diagnosi ADHD (elevato nel ~90% dei casi)
  - Monitoraggio trattamento (diminuisce con farmaci)
- **Approvato FDA**: Come ausilio diagnostico ADHD (sistema NEBA)

---

### 5. Caratteristiche Spettrali

#### `spec_edge_freq` - Frequenza di Bordo Spettrale (95%)
- **Formula**: Frequenza sotto la quale è contenuto il 95% della potenza
- **Unità**: Hz
- **Descrizione**: Indica dove è concentrata la maggior parte dell'energia del segnale
- **Interpretazione**:
  - **SEF basso (<10 Hz)**: Dominante onde lente (delta/theta)
  - **SEF alto (>15 Hz)**: Dominante attività veloce (beta/gamma)
- **Uso Clinico**:
  - Monitoraggio profondità anestesia (più basso durante anestesia profonda)
  - Caratterizzazione convulsioni
- **Range**: 5-30 Hz tipico per EEG da sveglio

#### `spec_entropy` - Entropia Spettrale
- **Formula**: `H = -Σ p(f) log₂ p(f)` dove `p(f) = PSD(f) / Σ PSD`
- **Unità**: Bits
- **Descrizione**: Misura di complessità/casualità spettrale
- **Interpretazione**:
  - **Entropia alta**: Potenza distribuita su molte frequenze (complesso, irregolare)
  - **Entropia bassa**: Potenza concentrata in poche frequenze (semplice, ritmico)
  - **Entropia massima**: Distribuzione uniforme (rumore bianco)
  - **Entropia minima**: Frequenza singola (onda sinusoidale pura)
- **Applicazioni Cliniche**:
  - Monitoraggio anestesia (diminuisce con profondità)
  - Rilevamento convulsioni (spesso diminuisce durante ictale)
  - Valutazione coscienza
- **Range**: 
  - Sinusoide pura: ~0 bits
  - EEG normale da sveglio: 3-6 bits
  - Rumore bianco: ~log₂(numero di bin di frequenza)

---

### 6. Analisi della Frequenza Dominante

#### `spec_dominant_freq` - Frequenza Dominante
- **Formula**: Frequenza con valore PSD massimo
- **Unità**: Hz
- **Descrizione**: Componente di frequenza più prominente
- **Interpretazione**:
  - Identifica la "frequenza portante" del segnale
  - **8-13 Hz**: Dominante alpha (rilassato)
  - **1-4 Hz**: Dominante delta (sonno/patologia)
  - **13-30 Hz**: Dominante beta (attivo)
- **Uso Clinico**: Valutazione rapida del ritmo dominante

#### `spec_dominant_power` - Potenza della Frequenza Dominante
- **Formula**: Valore PSD alla frequenza dominante
- **Unità**: µV²/Hz
- **Descrizione**: Forza della frequenza dominante
- **Interpretazione**:
  - Valore alto → attività forte, ritmica a quella frequenza
  - Valore basso → contenuto spettrale debole o diffuso

---

### 7. Misure di Tendenza Centrale

#### `spec_mean_freq` - Frequenza Media
- **Formula**: `f_mean = Σ(f × PSD(f)) / Σ PSD(f)`
- **Unità**: Hz
- **Descrizione**: Frequenza media pesata per potenza (centro di massa dello spettro)
- **Interpretazione**:
  - **Freq media bassa (<8 Hz)**: Dominante onde lente
  - **Freq media alta (>12 Hz)**: Dominante attività veloce
  - Si sposta verso l'alto con arousal/attivazione
- **Vantaggio**: Meno sensibile agli outlier rispetto alla frequenza dominante

#### `spec_median_freq` - Frequenza Mediana
- **Formula**: Frequenza che divide lo spettro in due metà di potenza uguale
- **Unità**: Hz
- **Descrizione**: Frequenza sotto la quale si trova il 50% della potenza
- **Interpretazione**:
  - Misura robusta della posizione spettrale (non influenzata da valori estremi)
  - Inferiore alla media se dominano le basse frequenze
- **Uso Clinico**: Valutazione affaticamento muscolare (diminuisce con affaticamento)

---

## Feature Funzionali

Le feature funzionali quantificano le interazioni e la sincronizzazione tra canali EEG.

### 1. Connettività Basata su Correlazione

#### `func_mean_corr` - Correlazione Media
- **Formula**: `mean(|corr(ch_i, ch_j)|)` per tutti j ≠ i
- **Unità**: Adimensionale (0 a 1)
- **Descrizione**: Correlazione assoluta media tra il canale e tutti gli altri
- **Interpretazione**:
  - **Correlazione media alta (>0.7)**: Forte connettività funzionale
    - Il canale è molto sincronizzato con la rete
    - Può indicare attività coordinata (es. convulsione, impegno globale nel compito)
  - **Correlazione media bassa (<0.3)**: Connettività debole
    - Processamento indipendente
    - Possibile artefatto o elettrodo difettoso
- **Range**: 0.2-0.8 tipico per EEG
- **Neuroscienza**: 
  - Reti resting state mostrano corr ~0.3-0.6
  - Reti correlate al compito possono aumentare a 0.6-0.8

#### `func_max_corr` - Correlazione Massima
- **Formula**: `max(|corr(ch_i, ch_j)|)` per tutti j ≠ i
- **Unità**: Adimensionale (0 a 1)
- **Descrizione**: Correlazione più forte con qualsiasi altro canale
- **Interpretazione**:
  - Identifica la coppia di canali più connessa
  - **Max alto (>0.9)**: Accoppiamento molto forte (elettrodi vicini spesso correlati)
  - **Max basso (<0.5)**: Nessuna connessione forte
- **Uso Clinico**: Rilevamento attività patologica sincronizzata

#### `func_std_corr` - Deviazione Standard delle Correlazioni
- **Formula**: `std(|corr(ch_i, ch_j)|)` per tutti j ≠ i
- **Unità**: Adimensionale
- **Descrizione**: Variabilità nelle forze di connessione
- **Interpretazione**:
  - **Std alta**: Connettività eterogenea (alcune connessioni forti, altre deboli)
  - **Std bassa**: Connettività omogenea (forza simile tra connessioni)
- **Neuroscienza**: Regioni hub spesso hanno std alta (poche connessioni forti)

#### `func_num_strong_conn` - Numero di Connessioni Forti
- **Formula**: `count(|corr(ch_i, ch_j)| > 0.7)` per tutti j ≠ i
- **Unità**: Conteggio (intero)
- **Descrizione**: Quanti canali sono fortemente correlati con questo
- **Interpretazione**:
  - **Conteggio alto**: Nodo hub nella rete
  - **Conteggio basso**: Nodo periferico o isolato
- **Range**: 0 a (N_canali - 1)
- **Teoria dei Grafi**: Analogo al grado del nodo nelle reti di correlazione

---

### 2. Sincronizzazione di Fase

#### `func_mean_plv` - Valore Medio di Phase Locking (PLV)
- **Formula**: `PLV = |⟨e^(i(φ₁(t) - φ₂(t)))⟩|`
  - Dove φ(t) è la fase istantanea dalla trasformata di Hilbert
- **Unità**: Adimensionale (0 a 1)
- **Descrizione**: Sincronizzazione di fase media tra canale e tutti gli altri
- **Interpretazione**:
  - **PLV = 0**: Nessuna relazione di fase (fasi casuali)
  - **PLV = 1**: Phase locking perfetto (differenza di fase costante)
  - **PLV = 0.3-0.5**: Sincronizzazione moderata (tipico per EEG a riposo)
  - **PLV > 0.7**: Sincronizzazione forte
- **Vantaggi rispetto alla Correlazione**:
  - Insensibile a differenze di ampiezza
  - Cattura meglio le relazioni di fase
  - Più sensibile all'accoppiamento oscillatorio
- **Applicazioni Neuroscienza**:
  - Comunicazione attraverso coerenza (teoria CTC)
  - Accoppiamento cross-frequenza
  - Comunicazione corticale a lungo raggio
- **Clinica**: 
  - Aumenta nell'epilessia (ipersincronia)
  - Alterata in schizofrenia, autismo
  - Cambia con carico cognitivo

#### `func_max_plv` - Valore Massimo di Phase Locking
- **Formula**: PLV massimo con qualsiasi altro canale
- **Unità**: Adimensionale (0 a 1)
- **Descrizione**: Sincronizzazione di fase più forte
- **Interpretazione**:
  - Identifica coppie di canali con relazione di fase più consistente
  - Elettrodi vicini: spesso PLV alto dovuto a conduzione di volume
  - PLV alto a distanza: probabilmente connettività funzionale
- **Uso**: Rilevamento hub di rete, tracciamento pathway

---

## Riferimenti Rapidi: Tabella Riassuntiva

| Tipo Feature | Conteggio | Metriche Chiave | Uso Principale |
|--------------|-----------|-----------------|----------------|
| **Temporali** | 13 | RMS, Hjorth, Skewness | Qualità segnale, dinamiche |
| **Spettrali** | 22 | Potenze di banda, rapporti, entropia | Contenuto frequenziale, stati cerebrali |
| **Funzionali** | 6 | Correlazione, PLV | Connettività di rete |

---

## Linee Guida per l'Interpretazione

### EEG Adulto Normale da Sveglio (Occhi Chiusi, Riposo)

| Feature | Range Tipico | Posizione Dominante |
|---------|--------------|---------------------|
| Potenza Alpha | 20-60 µV² | Occipitale (O1, O2) |
| Potenza Beta | 5-15 µV² | Centrale (C3, C4) |
| Potenza Theta | 5-20 µV² | Frontale |
| Potenza Delta | 5-30 µV² | Frontale |
| Rapporto Alpha/Beta | 1.5-3.0 | Posteriore |
| Entropia Spettrale | 4-6 bits | Diffusa |
| Correlazione Media | 0.3-0.6 | Reti |

---

## Glossario

- **PSD**: Power Spectral Density - distribuzione della potenza sulle frequenze
- **PLV**: Phase Locking Value - misura di sincronizzazione di fase
- **Parametri di Hjorth**: Misure nel dominio del tempo della complessità EEG
- **ERD**: Event-Related Desynchronization - diminuzione della potenza oscillatoria
- **ERS**: Event-Related Synchronization - aumento della potenza oscillatoria
- **Conduzione di Volume**: Diffusione passiva dell'attività elettrica attraverso i tessuti
- **Connettività Funzionale**: Dipendenze statistiche tra regioni cerebrali
- **Hub**: Nodo altamente connesso in una rete
