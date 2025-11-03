# Comprehensive Guide to EEG Feature Metrics

This document provides detailed explanations of all metrics calculated by the feature extraction system.

---

## Table of Contents
1. [Temporal Features (13 metrics)](#temporal-features)
2. [Spectral Features (22 metrics)](#spectral-features)
3. [Functional Features (6 metrics)](#functional-features)

---

## Temporal Features

Temporal features capture statistical and time-domain characteristics of the EEG signal.

### 1. Basic Statistical Measures

#### `temp_mean` - Signal Mean
- **Formula**: `μ = (1/N) Σ x(n)`
- **Unit**: µV (microvolts)
- **Description**: Average amplitude of the signal over the epoch
- **Interpretation**: 
  - Should be close to 0 for properly preprocessed EEG
  - Large deviations may indicate baseline drift or artifacts
- **Range**: Typically -5 to +5 µV

#### `temp_std` - Standard Deviation
- **Formula**: `σ = sqrt((1/N) Σ (x(n) - μ)²)`
- **Unit**: µV
- **Description**: Measure of signal variability/dispersion
- **Interpretation**:
  - Higher values → more variable signal
  - Lower values → more stable/flat signal
  - Related to signal power
- **Range**: 1-50 µV (typical for EEG)

#### `temp_var` - Variance
- **Formula**: `σ² = (1/N) Σ (x(n) - μ)²`
- **Unit**: µV²
- **Description**: Square of standard deviation
- **Interpretation**: 
  - Directly proportional to signal power
  - Higher variance → higher activity
- **Range**: 1-2500 µV²

#### `temp_min` / `temp_max` - Minimum and Maximum Values
- **Unit**: µV
- **Description**: Extreme amplitude values in the epoch
- **Interpretation**:
  - Useful for detecting artifacts (spikes, saturation)
  - Extremely high values (>100 µV) often indicate artifacts

#### `temp_range` - Signal Range
- **Formula**: `Range = max - min`
- **Unit**: µV
- **Description**: Total amplitude span
- **Interpretation**: 
  - Indicates dynamic range of the signal
  - Very large ranges may indicate artifacts

#### `temp_ptp` - Peak-to-Peak Amplitude
- **Formula**: Same as range
- **Unit**: µV
- **Description**: Difference between highest and lowest points
- **Interpretation**: 
  - Common measure in clinical EEG
  - Typical EEG: 10-100 µV peak-to-peak

---

### 2. Higher-Order Statistical Moments

#### `temp_skewness` - Skewness
- **Formula**: `γ₁ = E[(X-μ)³] / σ³`
- **Unit**: Dimensionless
- **Description**: Measure of distribution asymmetry
- **Interpretation**:
  - **Skewness = 0**: Symmetric distribution (normal)
  - **Skewness > 0**: Right-skewed (tail extends right, more positive outliers)
  - **Skewness < 0**: Left-skewed (tail extends left, more negative outliers)
  - In EEG: Sharp spikes create positive skewness
- **Range**: Typically -2 to +2 for EEG

#### `temp_kurtosis` - Kurtosis
- **Formula**: `γ₂ = E[(X-μ)⁴] / σ⁴ - 3` (excess kurtosis)
- **Unit**: Dimensionless
- **Description**: Measure of "tailedness" or peakedness
- **Interpretation**:
  - **Kurtosis = 0**: Normal distribution (mesokurtic)
  - **Kurtosis > 0**: Heavy tails, sharp peak (leptokurtic) - more outliers
  - **Kurtosis < 0**: Light tails, flat peak (platykurtic) - fewer outliers
  - High kurtosis in EEG often indicates artifacts or epileptiform activity
- **Range**: Typically -1 to +5 for clean EEG

---

### 3. Signal Energy and Activity

#### `temp_rms` - Root Mean Square
- **Formula**: `RMS = sqrt((1/N) Σ x(n)²)`
- **Unit**: µV
- **Description**: Effective or quadratic mean of the signal
- **Interpretation**:
  - Represents overall signal "strength" or energy
  - Related to power (RMS² = average power)
  - Higher RMS → higher neural activity
- **Range**: 5-50 µV typical for EEG
- **Clinical relevance**: Used to quantify seizure activity, sleep stages

---

### 4. Signal Dynamics

#### `temp_zcr` - Zero Crossing Rate
- **Formula**: `ZCR = (1/N) Σ |sign(x(n)) - sign(x(n-1))|`
- **Unit**: Crossings per sample
- **Description**: Frequency at which signal crosses zero amplitude
- **Interpretation**:
  - **High ZCR**: Fast oscillations, high frequency content
  - **Low ZCR**: Slow oscillations, low frequency content
  - Rough estimate of dominant frequency
- **Range**: 0.0 to 0.5 (normalized)
- **Example**: 
  - Slow delta waves: ZCR ≈ 0.01-0.02
  - Fast beta/gamma: ZCR ≈ 0.1-0.2

---

### 5. Hjorth Parameters

Named after Bo Hjorth (1970), these parameters describe signal complexity.

#### `temp_hjorth_activity` - Hjorth Activity
- **Formula**: `Activity = var(signal)`
- **Unit**: µV²
- **Description**: Signal variance (power)
- **Interpretation**:
  - Represents total power in the signal
  - Higher activity → stronger neural activation
  - Equivalent to `temp_var`
- **Clinical use**: Monitoring sedation depth, seizure detection

#### `temp_hjorth_mobility` - Hjorth Mobility
- **Formula**: `Mobility = sqrt(var(dx/dt) / var(x))`
- **Unit**: Dimensionless (or Hz-like)
- **Description**: Represents "mean frequency" - how rapidly the signal changes
- **Interpretation**:
  - **Higher mobility**: Signal changes rapidly (higher frequencies)
  - **Lower mobility**: Signal changes slowly (lower frequencies)
  - Proportional to standard deviation of power spectrum
- **Range**: Typically 1-20 for EEG
- **Example**:
  - Slow delta activity: Mobility ≈ 1-3
  - Fast gamma activity: Mobility ≈ 10-20

#### `temp_hjorth_complexity` - Hjorth Complexity
- **Formula**: `Complexity = Mobility(dx/dt) / Mobility(x)`
- **Unit**: Dimensionless
- **Description**: Measure of how much the signal resembles a pure sine wave
- **Interpretation**:
  - **Complexity = 1**: Pure sine wave (single frequency)
  - **Complexity > 1**: Multiple frequency components
  - **Higher complexity**: More irregular, chaotic signal
  - Indicates bandwidth of the signal
- **Range**: Typically 1.1-2.5 for EEG
- **Clinical significance**: 
  - Increases during seizures
  - Decreases during sleep
  - Indicator of cognitive load

---

## Spectral Features

Spectral features are extracted from the Power Spectral Density (PSD) computed using Welch's method.

### Power Spectral Density Background

**Welch's Method**:
- Divides signal into overlapping segments
- Computes FFT for each segment
- Averages to reduce noise
- Output: Power (µV²) per frequency (Hz)

---

### 1. Band Powers (Absolute)

EEG signals are divided into standard frequency bands based on neurophysiological significance.

#### `spec_delta` - Delta Band Power (1-4 Hz)
- **Unit**: µV²
- **Physiological Correlates**:
  - **Deep sleep** (Stage 3-4 NREM)
  - **Unconsciousness** (anesthesia, coma)
  - **Brain lesions** (pathological)
- **Typical Amplitude**: High (30-200 µV)
- **Spatial Distribution**: Frontal regions
- **Clinical Significance**:
  - Increases with drowsiness and sleep depth
  - Pathologically high in encephalopathy
- **During Task**: Usually suppressed during active cognition

#### `spec_theta` - Theta Band Power (4-8 Hz)
- **Unit**: µV²
- **Physiological Correlates**:
  - **Memory encoding and retrieval** (hippocampal theta)
  - **Drowsiness**, light sleep
  - **Meditation**, deep relaxation
  - **Cognitive control** (frontal midline theta)
  - **Error monitoring**
- **Typical Amplitude**: Moderate (10-50 µV)
- **Spatial Distribution**: Frontal-midline (Fz, Cz) during tasks; temporal during memory
- **Clinical Significance**:
  - Increases in ADHD
  - Frontal theta burst in decision-making
- **During Imagined Speech**: May increase with working memory load

#### `spec_alpha` - Alpha Band Power (8-13 Hz)
- **Unit**: µV²
- **Physiological Correlates**:
  - **Relaxed wakefulness** (eyes closed)
  - **Cortical idling** (inhibition of active processing)
  - **Visual suppression**
  - **Attention modulation**
- **Typical Amplitude**: Highest in awake rest (20-60 µV)
- **Spatial Distribution**: Occipital (O1, O2) - visual cortex
- **Variants**:
  - **α1 (8-10 Hz)**: Lower alpha, more cognitive
  - **α2 (10-13 Hz)**: Upper alpha, more perceptual
- **Clinical Significance**:
  - **Alpha blocking**: Disappears when eyes open or during mental activity
  - Reduced in dementia
- **During Tasks**: Typically suppresses (ERD - Event-Related Desynchronization)

#### `spec_beta` - Beta Band Power (13-30 Hz)
- **Unit**: µV²
- **Physiological Correlates**:
  - **Active thinking** and concentration
  - **Motor planning and execution**
  - **Anxiety** and arousal
  - **Sensorimotor processing**
- **Typical Amplitude**: Low-moderate (5-20 µV)
- **Spatial Distribution**: Sensorimotor cortex (C3, C4)
- **Sub-bands**:
  - **β1 (13-20 Hz)**: Low beta, motor control
  - **β2 (20-30 Hz)**: High beta, alertness, anxiety
- **Clinical Significance**:
  - Increases with anxiety, stress
  - Beta rebound after movement
  - Excessive beta in insomnia
- **During Imagined Speech**: Increases with cognitive engagement, speech planning

#### `spec_gamma` - Gamma Band Power (30-45 Hz)
- **Unit**: µV²
- **Physiological Correlates**:
  - **Sensory binding** (integrating features)
  - **Attention** and consciousness
  - **Working memory** maintenance
  - **Language processing**
  - **High-level cognitive function**
- **Typical Amplitude**: Very low (2-10 µV)
- **Spatial Distribution**: Task-dependent, often parietal/frontal
- **Clinical Significance**:
  - Reduced in schizophrenia, autism
  - Abnormal in epilepsy
  - Correlates with consciousness level
- **During Imagined Speech**: May show task-related increases
- **Note**: Higher gamma (>60 Hz) may contain muscle artifacts

---

### 2. Band Powers (Relative)

#### `spec_delta_rel`, `spec_theta_rel`, `spec_alpha_rel`, `spec_beta_rel`, `spec_gamma_rel`
- **Formula**: `Relative Power = Band Power / Total Power`
- **Unit**: Dimensionless (proportion)
- **Range**: 0 to 1 (sum ≈ 1 across all bands)
- **Description**: Normalized band power independent of absolute amplitude
- **Advantages**:
  - **Reduces inter-subject variability** (different skull thickness, electrode impedance)
  - **Better for classification** (more stable across sessions)
  - **Comparable across different amplifiers/settings**
- **Interpretation**: Represents spectral composition rather than absolute strength

---

### 3. Total Power

#### `spec_total_power` - Total Spectral Power (1-45 Hz)
- **Formula**: `∫₁⁴⁵ PSD(f) df`
- **Unit**: µV²
- **Description**: Total signal power across all analyzed frequencies
- **Interpretation**:
  - Higher total power → more neural activity or artifacts
  - Can vary greatly between subjects and electrodes
  - Influenced by electrode quality, scalp conductance

---

### 4. Band Power Ratios

These ratios capture relationships between frequency bands and are clinically meaningful.

#### `spec_alpha_beta_ratio` - Alpha/Beta Ratio
- **Formula**: `α/β = P_alpha_rel / P_beta_rel`
- **Unit**: Dimensionless
- **Description**: Balance between relaxation and activation
- **Interpretation**:
  - **High ratio (>1)**: Relaxed, low arousal, idling
  - **Low ratio (<1)**: Alert, active processing, aroused
- **Clinical Applications**:
  - ADHD diagnosis (typically lower ratio)
  - Neurofeedback training target
  - Meditation monitoring (increases)
- **Range**: 0.5-3.0 typical

#### `spec_theta_alpha_ratio` - Theta/Alpha Ratio
- **Formula**: `θ/α = P_theta_rel / P_alpha_rel`
- **Unit**: Dimensionless
- **Description**: Indicator of cognitive fatigue and attentional state
- **Interpretation**:
  - **High ratio**: Drowsiness, fatigue, low vigilance
  - **Low ratio**: Alert, focused attention
- **Clinical Applications**:
  - Drowsiness detection (driving, operators)
  - ADHD (elevated ratio)
  - Sleep onset prediction
- **Range**: 0.3-2.0 typical

#### `spec_theta_beta_ratio` - Theta/Beta Ratio
- **Formula**: `θ/β = P_theta_rel / P_beta_rel`
- **Unit**: Dimensionless
- **Description**: Classic ADHD biomarker
- **Interpretation**:
  - **High ratio (>2.0)**: Possible ADHD, poor attention regulation
  - **Normal ratio (0.5-2.0)**: Typical attention control
- **Clinical Applications**:
  - ADHD diagnosis (elevated in ~90% of cases)
  - Treatment monitoring (decreases with medication)
- **FDA-approved**: As ADHD diagnostic aid (NEBA system)

---

### 5. Spectral Characteristics

#### `spec_edge_freq` - Spectral Edge Frequency (95%)
- **Formula**: Frequency below which 95% of power is contained
- **Unit**: Hz
- **Description**: Indicates where most signal energy is concentrated
- **Interpretation**:
  - **Low SEF (<10 Hz)**: Slow-wave dominant (delta/theta)
  - **High SEF (>15 Hz)**: Fast activity dominant (beta/gamma)
- **Clinical Use**:
  - Anesthesia depth monitoring (lower during deep anesthesia)
  - Seizure characterization
- **Range**: 5-30 Hz typical for awake EEG

#### `spec_entropy` - Spectral Entropy
- **Formula**: `H = -Σ p(f) log₂ p(f)` where `p(f) = PSD(f) / Σ PSD`
- **Unit**: Bits
- **Description**: Measure of spectral complexity/randomness
- **Interpretation**:
  - **High entropy**: Power distributed across many frequencies (complex, irregular)
  - **Low entropy**: Power concentrated in few frequencies (simple, rhythmic)
  - **Maximum entropy**: Uniform distribution (white noise)
  - **Minimum entropy**: Single frequency (pure sine wave)
- **Clinical Applications**:
  - Anesthesia monitoring (decreases with depth)
  - Seizure detection (often decreases during ictal)
  - Consciousness assessment
- **Range**: 
  - Pure sine: ~0 bits
  - Normal awake EEG: 3-6 bits
  - White noise: ~log₂(number of frequency bins)

---

### 6. Dominant Frequency Analysis

#### `spec_dominant_freq` - Dominant Frequency
- **Formula**: Frequency with maximum PSD value
- **Unit**: Hz
- **Description**: Most prominent frequency component
- **Interpretation**:
  - Identifies the "carrier frequency" of the signal
  - **8-13 Hz**: Alpha dominant (relaxed)
  - **1-4 Hz**: Delta dominant (sleep/pathology)
  - **13-30 Hz**: Beta dominant (active)
- **Clinical Use**: Quick assessment of dominant rhythm

#### `spec_dominant_power` - Dominant Frequency Power
- **Formula**: PSD value at dominant frequency
- **Unit**: µV²/Hz
- **Description**: Strength of the dominant frequency
- **Interpretation**:
  - High value → strong, rhythmic activity at that frequency
  - Low value → weak or diffuse spectral content

---

### 7. Central Tendency Measures

#### `spec_mean_freq` - Mean Frequency
- **Formula**: `f_mean = Σ(f × PSD(f)) / Σ PSD(f)`
- **Unit**: Hz
- **Description**: Power-weighted average frequency (center of mass of spectrum)
- **Interpretation**:
  - **Low mean freq (<8 Hz)**: Slow-wave dominant
  - **High mean freq (>12 Hz)**: Fast activity dominant
  - Shifts higher with arousal/activation
- **Advantage**: Less sensitive to outliers than dominant frequency

#### `spec_median_freq` - Median Frequency
- **Formula**: Frequency that divides spectrum into two equal power halves
- **Unit**: Hz
- **Description**: Frequency below which 50% of power lies
- **Interpretation**:
  - Robust measure of spectral location (unaffected by extreme values)
  - Lower than mean if low frequencies dominate
- **Clinical Use**: Muscle fatigue assessment (decreases with fatigue)

---

## Functional Features

Functional features quantify interactions and synchronization between EEG channels.

### 1. Correlation-Based Connectivity

#### `func_mean_corr` - Mean Correlation
- **Formula**: `mean(|corr(ch_i, ch_j)|)` for all j ≠ i
- **Unit**: Dimensionless (0 to 1)
- **Description**: Average absolute correlation between the channel and all others
- **Interpretation**:
  - **High mean correlation (>0.7)**: Strong functional connectivity
    - Channel is highly synchronized with the network
    - May indicate coordinated activity (e.g., seizure, global task engagement)
  - **Low mean correlation (<0.3)**: Weak connectivity
    - Independent processing
    - Possible artifact or bad electrode
- **Range**: 0.2-0.8 typical for EEG
- **Neuroscience**: 
  - Resting state networks show corr ~0.3-0.6
  - Task-related networks may increase to 0.6-0.8

#### `func_max_corr` - Maximum Correlation
- **Formula**: `max(|corr(ch_i, ch_j)|)` for all j ≠ i
- **Unit**: Dimensionless (0 to 1)
- **Description**: Strongest correlation with any other channel
- **Interpretation**:
  - Identifies the most connected channel pair
  - **High max (>0.9)**: Very strong coupling (neighboring electrodes often correlated)
  - **Low max (<0.5)**: No strong connections
- **Clinical Use**: Detecting synchronized pathological activity

#### `func_std_corr` - Standard Deviation of Correlations
- **Formula**: `std(|corr(ch_i, ch_j)|)` for all j ≠ i
- **Unit**: Dimensionless
- **Description**: Variability in connection strengths
- **Interpretation**:
  - **High std**: Heterogeneous connectivity (some strong, some weak connections)
  - **Low std**: Homogeneous connectivity (similar strength across connections)
- **Neuroscience**: Hub regions often have high std (few strong connections)

#### `func_num_strong_conn` - Number of Strong Connections
- **Formula**: `count(|corr(ch_i, ch_j)| > 0.7)` for all j ≠ i
- **Unit**: Count (integer)
- **Description**: How many channels are strongly correlated with this one
- **Interpretation**:
  - **High count**: Hub node in network
  - **Low count**: Peripheral or isolated node
- **Range**: 0 to (N_channels - 1)
- **Graph Theory**: Analogous to node degree in correlation networks

---

### 2. Phase Synchronization

#### `func_mean_plv` - Mean Phase Locking Value
- **Formula**: `PLV = |⟨e^(i(φ₁(t) - φ₂(t)))⟩|`
  - Where φ(t) is instantaneous phase from Hilbert transform
- **Unit**: Dimensionless (0 to 1)
- **Description**: Average phase synchronization between channel and all others
- **Interpretation**:
  - **PLV = 0**: No phase relationship (random phases)
  - **PLV = 1**: Perfect phase locking (constant phase difference)
  - **PLV = 0.3-0.5**: Moderate synchronization (typical for resting EEG)
  - **PLV > 0.7**: Strong synchronization
- **Advantages over Correlation**:
  - Insensitive to amplitude differences
  - Captures phase relationships better
  - More sensitive to oscillatory coupling
- **Neuroscience Applications**:
  - Communication through coherence (CTC theory)
  - Cross-frequency coupling
  - Long-range cortical communication
- **Clinical**: 
  - Increases in epilepsy (hypersynchrony)
  - Altered in schizophrenia, autism
  - Changes with cognitive load

#### `func_max_plv` - Maximum Phase Locking Value
- **Formula**: Maximum PLV with any other channel
- **Unit**: Dimensionless (0 to 1)
- **Description**: Strongest phase synchronization
- **Interpretation**:
  - Identifies channel pairs with most consistent phase relationship
  - Neighboring electrodes: often high PLV due to volume conduction
  - Distant high PLV: likely functional connectivity
- **Use**: Network hub detection, pathway tracing

---

## Summary Table: Quick Reference

| Feature Type | Count | Key Metrics | Primary Use |
|--------------|-------|-------------|-------------|
| **Temporal** | 13 | RMS, Hjorth, Skewness | Signal quality, dynamics |
| **Spectral** | 22 | Band powers, ratios, entropy | Frequency content, brain states |
| **Functional** | 6 | Correlation, PLV | Network connectivity |

---

## Interpretation Guidelines

### Normal Awake Adult EEG (Eyes Closed, Resting)

| Feature | Typical Range | Dominant Location |
|---------|---------------|-------------------|
| Alpha Power | 20-60 µV² | Occipital (O1, O2) |
| Beta Power | 5-15 µV² | Central (C3, C4) |
| Theta Power | 5-20 µV² | Frontal |
| Delta Power | 5-30 µV² | Frontal |
| Alpha/Beta Ratio | 1.5-3.0 | Posterior |
| Spectral Entropy | 4-6 bits | Widespread |
| Mean Correlation | 0.3-0.6 | Networks |

### Task-Engaged State

| Change | Interpretation |
|--------|----------------|
| ↓ Alpha | Cortical activation (alpha blocking) |
| ↑ Beta | Cognitive engagement |
| ↑ Gamma | Attention, binding |
| ↑ Theta (frontal) | Working memory load |
| ↑ Connectivity | Network coordination |

---

## Clinical Applications

### ADHD Markers
- ↑ Theta/Beta ratio (>2.0)
- ↓ Beta power
- ↑ Slow wave activity

### Drowsiness/Sleep
- ↑ Alpha → Theta → Delta progression
- ↓ Beta, gamma
- ↓ Spectral entropy
- ↓ Hjorth complexity

### Seizure Activity
- ↑ Synchronization (PLV, correlation)
- Rhythmic activity (low entropy)
- ↑ Hjorth complexity (initially)
- Abnormal band power distribution

### Cognitive Load
- ↑ Frontal theta
- ↑ Gamma
- ↑ Network connectivity
- ↓ Alpha (posterior)

---

## References

1. **Hjorth, B. (1970)**. "EEG analysis based on time domain properties." *Electroencephalography and Clinical Neurophysiology*, 29(3), 306-310.

2. **Welch, P. (1967)**. "The use of fast Fourier transform for the estimation of power spectra." *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70-73.

3. **Lachaux, J.P., et al. (1999)**. "Measuring phase synchrony in brain signals." *Human Brain Mapping*, 8(4), 194-208.

4. **Klimesch, W. (1999)**. "EEG alpha and theta oscillations reflect cognitive and memory performance: a review and analysis." *Brain Research Reviews*, 29(2-3), 169-195.

5. **Başar, E., et al. (2001)**. "Gamma, alpha, delta, and theta oscillations govern cognitive processes." *International Journal of Psychophysiology*, 39(2-3), 241-248.

6. **Nunez, P.L., & Srinivasan, R. (2006)**. *Electric fields of the brain: the neurophysics of EEG*. Oxford University Press.

7. **Stam, C.J. (2005)**. "Nonlinear dynamical analysis of EEG and MEG." *Clinical Neurophysiology*, 116(10), 2197-2208.

---

## Glossary

- **PSD**: Power Spectral Density - distribution of power across frequencies
- **PLV**: Phase Locking Value - measure of phase synchronization
- **Hjorth Parameters**: Time-domain measures of EEG complexity
- **ERD**: Event-Related Desynchronization - decrease in oscillatory power
- **ERS**: Event-Related Synchronization - increase in oscillatory power
- **Volume Conduction**: Passive spread of electrical activity through tissue
- **Functional Connectivity**: Statistical dependencies between brain regions
- **Hub**: Highly connected node in a network
