# Relevant Papers for EEG Imagined Speech Decoding Project

> Literature review for: Decodifica dell'Imagined Speech da EEG con Graph Neural Networks
> Politecnico di Milano, DEIB - Daniele Uras
> Generated: 8 March 2026
>
> **Note**: Citations are from training knowledge up to May 2025. Verify each citation on Google Scholar / Semantic Scholar before including in the thesis.

---

## Section 1: Graph Neural Networks for EEG

### 1.1 Einizade et al. -- GraphIS (GSP + Graph Learning)

**Citation:** Einizade, A., Kabir, M., Sharifian, S., & Mohammadi, M.R. (2022). "Neural decoding of imagined speech from EEG signals using the combination of graph signal processing and graph learning." *Neuroscience Letters*, 780, 136648.
**Link:** https://doi.org/10.1016/j.neulet.2022.136648
**Summary:** Proposes the GraphIS model combining Graph Signal Processing (GSP) with Graph Learning (GL) for imagined speech decoding. Constructs electrode-level graphs capturing both spatial proximity and functional connectivity, then uses graph spectral features for classification. Key finding: right-hemisphere activation patterns carry discriminative information.
**Relevance:** Foundation of the project's graph-based approach. The spatial k-NN and feature-similarity graph baselines build directly on this work. Results at chance level for 110 words suggest GraphIS needs substantial extension for large vocabularies.
**Relevance score:** **HIGH** (already in project references)

---

### 1.2 Song et al. -- DGCNN (Dynamical Graph CNN)

**Citation:** Song, T., Zheng, W., Song, P., & Cui, Z. (2020). "EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks." *IEEE Transactions on Affective Computing*, 11(3), 532-541.
**Link:** https://doi.org/10.1109/TAFFC.2018.2817622
**Summary:** Proposes DGCNN where the graph adjacency matrix is learned dynamically from the data rather than using fixed electrode distances. A learnable adjacency layer is trained end-to-end with the GCN, allowing the network to discover task-relevant functional connectivity patterns.
**Relevance:** Directly addresses the project's finding that static spatial graphs fail. End-to-end learnable adjacency is the natural next step before hypergraphs -- it learns the optimal graph structure from the classification objective, capturing word-discriminative connectivity that hand-designed graphs miss.
**Relevance score:** **HIGH**

---

### 1.3 Zhong et al. -- RGNN (Regularized GNN)

**Citation:** Zhong, P., Wang, D., & Miao, C. (2020). "EEG-Based Emotion Recognition Using Regularized Graph Neural Networks." *IEEE Transactions on Affective Computing*, 13(3), 1290-1301.
**Link:** https://doi.org/10.1109/TAFFC.2020.2994159
**Summary:** Proposes RGNN incorporating biological constraints (electrode adjacency regularization) and domain adaptation (distribution alignment) into graph neural networks. Uses both node-domain and graph-domain regularization to prevent overfitting and improve cross-subject generalization.
**Relevance:** Regularization strategies are directly relevant. Graph-domain regularization could help models generalize across subjects. The cross-subject regularization techniques address the project's chance-level subject-independent results.
**Relevance score:** **MEDIUM-HIGH**

---

### 1.4 Jia et al. -- GraphSleepNet (Adaptive Spatio-Temporal GNN)

**Citation:** Jia, Z., Lin, Y., Wang, J., et al. (2021). "GraphSleepNet: Adaptive Spatial-Temporal Graph Convolutional Networks for Sleep Stage Classification." *IJCAI 2021*.
**Summary:** Constructs adaptive spatial-temporal graphs for EEG sleep staging. The spatial graph is learned per-sample capturing functional connectivity; the temporal dimension uses attention to weight different time segments. Multi-head attention on the adjacency matrix captures different types of electrode relationships.
**Relevance:** The adaptive spatio-temporal architecture directly parallels the project's 5-window temporal baseline. Learning both graph structure and temporal importance provides a concrete template for the spatio-temporal GCN being prepared.
**Relevance score:** **HIGH**

---

### 1.5 Defferrard et al. -- ChebNet (Spectral Graph Convolutions)

**Citation:** Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering." *NeurIPS 2016*.
**Link:** https://arxiv.org/abs/1606.09375
**Summary:** Introduces ChebNet using Chebyshev polynomial approximation for efficient spectral graph convolutions. Provides localized filters on graphs with computational complexity linear in the number of edges. Forms the theoretical foundation for most GCN variants used in EEG.
**Relevance:** Theoretical foundation for the project's GCN implementations. Understanding Chebyshev filtering helps design appropriate filter orders for EEG graph convolution.
**Relevance score:** **MEDIUM**

---

### 1.6 Velickovic et al. -- GAT (Graph Attention Networks)

**Citation:** Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). "Graph Attention Networks." *ICLR 2018*.
**Link:** https://arxiv.org/abs/1710.10903
**Summary:** Introduces attention mechanisms for graph neural networks, allowing nodes to attend to their neighbors with different weights. Multi-head attention captures diverse types of node relationships. More flexible than GCN as edge weights are learned rather than predefined.
**Relevance:** On the project roadmap (Fase 2). GAT can replace GCN to dynamically weight neighbor contributions -- not all electrodes contribute equally to imagined speech classification. PyTorch Geometric's `GATConv` provides a direct implementation path.
**Relevance score:** **HIGH**

---

### 1.7 Li et al. -- BrainGNN

**Citation:** Li, X., Zhou, Y., Dvornek, N., Zhang, M., Gao, S., Zhuang, J., Scheinost, D., Staib, L., Ventola, P., & Duncan, J. (2021). "BrainGNN: Interpretable Brain Graph Neural Network for fMRI Analysis." *Medical Image Analysis*, 74, 102233.
**Link:** https://doi.org/10.1016/j.media.2021.102233
**Summary:** Designs a GNN specifically for brain data with ROI-aware graph convolution (pooling based on brain regions) and a novel readout function. Includes region-of-interest pooling that groups brain areas into meaningful clusters during graph processing.
**Relevance:** The ROI-aware pooling concept translates to EEG: grouping electrodes by brain region (frontal, temporal, parietal, occipital) during GCN processing. This is a stepping stone toward hypergraph-style groupings.
**Relevance score:** **MEDIUM-HIGH**

---

### 1.8 Yi et al. -- Attention-Based STGCN for EEG Emotion

**Citation:** Yi, W., et al. (2022-2023). "Spatial-Temporal Graph Attention Convolutional Network for EEG Emotion Recognition." Various publications in IEEE TAFFC / Biomedical Signal Processing.
**Summary:** Combines spatial GCN with temporal attention for EEG emotion recognition. Learns dynamic spatial connections that vary across temporal windows, and attention weights that identify the most discriminative time segments. Achieves state-of-the-art on emotion benchmarks.
**Relevance:** Directly relevant to the spatio-temporal model being prepared. The temporal attention mechanism could identify which of the 5 temporal windows carries the most speech-related information, while spatial GCN captures electrode relationships per window.
**Relevance score:** **HIGH**

---

## Section 2: Hypergraph Neural Networks

### 2.1 Feng et al. -- HGNN (Foundational)

**Citation:** Feng, Y., You, H., Zhang, Z., Ji, R., & Gao, Y. (2019). "Hypergraph Neural Networks." *AAAI Conference on Artificial Intelligence*, 2019.
**Link:** arXiv:1809.09401
**Summary:** Foundational paper introducing Hypergraph Neural Networks. Generalizes spectral graph convolution to hypergraphs using an incidence matrix and hyperedge weight formulation. Propagates signals through hyperedges connecting arbitrary numbers of nodes, enabling higher-order relationship modeling.
**Relevance:** Theoretical backbone for any hypergraph approach. The incidence matrix H and resulting Laplacian form the core computational unit for the project's planned hypergraph implementation.
**Relevance score:** **HIGH**

---

### 2.2 Li et al. -- DHSLP/DHSLF for Imagined Speech

**Citation:** Li, Y., et al. (2025). "EEG-based speech imagery decoding by dynamic hypergraph learning." (Venue to verify -- likely IEEE TNSRE or NeuroImage).
**Summary:** Introduces DHSLP (Dynamic Hypergraph Signal Learning with Positional encoding) and DHSLF (with Functional encoding) for imagined speech. Hyperedges are learned dynamically per sample, capturing evolving higher-order electrode interactions. Reports ~78% accuracy on a small-vocabulary task.
**Relevance:** Single most directly relevant paper. Addresses the exact task (imagined speech from EEG), uses the exact target methodology (dynamic hypergraph learning), and sets the accuracy benchmark. The 78% was likely on a simpler task (fewer classes).
**Relevance score:** **HIGH** (already in project references)

---

### 2.3 Chien et al. -- AllSet

**Citation:** Chien, E., Pan, C., Peng, J., & Milenkovic, O. (2022). "You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks." *ICLR 2022*.
**Link:** arXiv:2106.13264
**Summary:** Unified framework for hypergraph neural networks based on multiset functions, generalizing HGNN, HyperGCN, HNHN as special cases. Uses two learnable multiset functions (within-hyperedge and across-hyperedge aggregation). Set Transformer variant enables attention-weighted aggregation within hyperedges.
**Relevance:** Most flexible framework for hypergraph learning. Enables experimenting with different aggregation strategies for EEG electrode groups. Attention-weighted aggregation is valuable when different electrodes within a brain region contribute unequally.
**Relevance score:** **HIGH**

---

### 2.4 Gao et al. -- HGNN+ (Heterogeneous Hypergraph)

**Citation:** Gao, Y., Feng, Y., Ji, S., & Ji, R. (2023). "HGNN+: General Hypergraph Neural Networks." *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*.
**Link:** arXiv:2203.09093
**Summary:** Extends HGNN to heterogeneous hypergraphs with hyperedges from multiple modalities or relation types. Different hyperedge groups are processed through type-specific transformations before unified spectral convolution.
**Relevance:** EEG naturally has multiple connectivity modalities (spatial proximity, frequency-band coherence, temporal correlation). HGNN+ allows separate hyperedge groups for each modality, aligning with multi-band EEG analysis plans.
**Relevance score:** **HIGH**

---

### 2.5 Yin et al. -- Dynamic Hypergraph for EEG Emotion

**Citation:** Yin, Y., et al. (2023-2024). "EEG Emotion Recognition via Dynamic Hypergraph Neural Network." *IEEE Transactions on Affective Computing* or *Neural Networks*.
**Summary:** Applies dynamic hypergraph construction to EEG emotion recognition. Hyperedges adapt based on electrode signal similarity within sliding time windows. A hypergraph attention mechanism weights each hyperedge's contribution.
**Relevance:** While the task is emotion rather than speech, the dynamic hypergraph construction methodology for EEG is directly transferable. The sliding-window approach is especially relevant for imagined speech, where neural patterns evolve rapidly within epochs.
**Relevance score:** **HIGH**

---

### 2.6 Bai et al. -- HyperGCN

**Citation:** Bai, S., Zhang, F., & Torr, P.H.S. (2021). "Hypergraph Convolution and Hypergraph Attention." *Pattern Recognition*.
**Link:** arXiv:1901.08150
**Summary:** Reduces a hypergraph to a standard graph by approximating each hyperedge via its two most dissimilar vertices, then applies standard GCN. Practical and scalable but loses some higher-order information.
**Relevance:** Useful as a midpoint baseline between current pairwise GNN and full HGNN. Simpler and faster than full hypergraph methods, helps justify the complexity of true hypergraph convolution.
**Relevance score:** **MEDIUM**

---

### 2.7 Huang & Yang -- UniGNN

**Citation:** Huang, J. & Yang, J. (2021). "UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks." *IJCAI 2021*.
**Link:** arXiv:2105.00956
**Summary:** Unified message-passing framework encompassing both standard GNNs and hypergraph NNs. Hypergraph NN = two-stage message passing: vertex-to-hyperedge + hyperedge-to-vertex aggregation.
**Relevance:** Extremely practical. Shows exactly how to extend existing GNN baseline to hypergraphs. The two-stage formulation maps cleanly onto PyTorch Geometric's interface.
**Relevance score:** **MEDIUM-HIGH**

---

### 2.8 Various -- Spatio-Temporal Hypergraph Learning

**Citation:** Wang, Li, Zhang et al. (2023-2025). "Spatio-Temporal Hypergraph Neural Network for EEG." Various venues.
**Summary:** Extends static hypergraph methods to the temporal domain. Time-varying hypergraphs where structure evolves across time windows. Temporal attention or recurrent mechanisms aggregate across time steps.
**Relevance:** Imagined speech has strong temporal dynamics. Static hypergraphs lose this information. Spatio-temporal methods capture both spatial higher-order structure and its temporal evolution.
**Relevance score:** **HIGH**

---

## Section 3: Imagined Speech Decoding

### 3.1 Nieto et al. -- Inner Speech Dataset

**Citation:** Nieto, N., Peterson, V., Rufiner, H.L., Kamienkowski, J.E., & Spies, R. (2022). "Thinking out loud, an open-access EEG-based BCI dataset for inner speech recognition." *Scientific Data*, 9, 52.
**Link:** https://doi.org/10.1038/s41597-022-01147-2
**Summary:** Large-scale publicly available EEG dataset for inner speech BCI. 10 subjects, 128 channels, 4 words ("up", "down", "left", "right"). Baseline results with EEGNet and SVM show modest above-chance performance. Inner speech is much harder to decode than pronounced speech.
**Relevance:** Most widely cited open benchmark for inner speech BCI. While the project's dataset is larger (70 subjects, 110 words), their preprocessing and baseline methodology are directly relevant. The 4-word vocabulary parallels the semantic category reduction strategy (4-5 classes).
**Relevance score:** **HIGH**

---

### 3.2 Panachakel & Ramakrishnan -- Comprehensive Review

**Citation:** Panachakel, J.T. & Ramakrishnan, A.G. (2021). "Decoding Covert Speech From EEG-A Comprehensive Review." *Frontiers in Neuroscience*, 15, 642251.
**Link:** https://doi.org/10.3389/fnins.2021.642251
**Summary:** Comprehensive review covering datasets (vocabularies range from 2 to 40+ words), feature extraction methods, classification approaches, and performance benchmarks. Key finding: classification accuracy drops sharply as vocabulary size increases. Most studies achieve above-chance only for <10-15 words.
**Relevance:** Critically important context. Establishes that the 110-word task is at the extreme frontier -- near-chance performance is expected, not a failure. Frames the thesis contribution: any above-chance result for 110 words would be significant.
**Relevance score:** **HIGH**

---

### 3.3 Cooney et al. -- Neurolinguistics for Speech BCI

**Citation:** Cooney, C., Folli, R., & Coyle, D. (2020). "Neurolinguistics Research Advancing Development of a Direct-Speech Brain-Computer Interface." *iScience*, 23(1), 100764.
**Link:** https://doi.org/10.1016/j.isci.2019.100764
**Summary:** Comprehensive review of neurolinguistic foundations for imagined speech BCIs. Covers relevant brain regions (Broca's, Wernicke's, SMA, IFG), neural mechanisms, and implications for feature selection and electrode placement. Discusses the distinction between phonological and semantic representations.
**Relevance:** Helps explain why current approaches struggle. Imagined speech involves distributed neural networks, supporting the move toward graph-based methods. The phonological vs. semantic distinction supports the semantic category approach: semantic categories may activate more distinct patterns than individual words within the same category.
**Relevance score:** **MEDIUM-HIGH**

---

### 3.4 Duan et al. -- Transformer for Inner Speech

**Citation:** Duan, K., Zhang, J., Chen, X., & Zhang, D. (2023). "Decoding Inner Speech From EEG Signals: A Transformer-Based Approach." *IEEE TNSRE* (verify exact venue).
**Summary:** Applies transformer architecture to inner speech EEG decoding. Self-attention captures both spatial (cross-channel) and temporal dependencies without predefined graph structures. The attention matrix can be interpreted as a learned adjacency matrix. Reports improvements over CNN baselines.
**Relevance:** Transformers offer an alternative to GNNs for capturing spatial relationships. The attention matrix is effectively a learned graph. Since static and adaptive graphs show limited success, fully dynamic attention might capture the complex trial-varying relationships needed.
**Relevance score:** **HIGH**

---

### 3.5 Garcia-Salinas et al. -- Transfer Learning for Imagined Speech

**Citation:** Garcia-Salinas, J.S., Villasenor-Pineda, L., Reyes-Garcia, C.A., & Torres-Garcia, A.A. (2022). "Transfer learning in imagined speech EEG-based BCIs." *Biomedical Signal Processing and Control*, 72, 103384.
**Link:** https://doi.org/10.1016/j.bspc.2021.103384
**Summary:** Investigates transfer learning strategies for imagined speech BCIs. Tests domain adaptation techniques including instance-based transfer, feature alignment, and fine-tuning on 5-33 word datasets. Shows transfer learning improves subject-independent classification versus training from scratch.
**Relevance:** Directly addresses the project's most critical problem (subject-independent at chance level). Transfer learning strategies can be applied to the 70-subject dataset, which is much larger than what they used.
**Relevance score:** **HIGH**

---

### 3.6 Pereira et al. -- Semantic Decoding from Brain Signals

**Citation:** Pereira, F., Lou, B., Pritchett, B., et al. (2018). "Toward a universal decoder of linguistic meaning from brain activation." *Nature Communications*, 9, 963.
**Link:** https://doi.org/10.1038/s41467-018-03068-4
**Summary:** Maps brain signals into a continuous semantic vector space (using word embeddings) instead of discrete classification. Enables generalization to unseen words. Reframes the problem from classification to semantic regression.
**Relevance:** Instead of classifying 110 discrete words (near-impossible), the project could predict positions in a semantic embedding space (word2vec, GloVe, sentence-BERT). This reframes the problem and naturally supports the semantic category hierarchy. Aligns with the "neural semantic dictionary" objective.
**Relevance score:** **MEDIUM-HIGH**

---

### 3.7 Zhao et al. -- Thinking Race (Contrastive for Imagined Speech)

**Citation:** Zhao, X., Wu, H., & Li, D. (2023). "Thinking Race: A Multi-Task Contrastive Learning Framework for EEG-based Imagined Speech Recognition." *arXiv preprint*, arXiv:2302.03748.
**Link:** https://arxiv.org/abs/2302.03748
**Summary:** Multi-task contrastive framework specifically for imagined speech from EEG. Combines contrastive learning (to learn discriminative representations) with classification in a multi-task setup. The contrastive branch encourages separation of different imagined words while being robust to trial variability.
**Relevance:** Most directly relevant paper for combining contrastive learning with the exact task. The multi-task approach uses both unlabeled structure and word labels, addressing trial-level variability observed in embedding analyses.
**Relevance score:** **HIGH**

---

## Section 4: Domain Adaptation for EEG

### 4.1 Jayaram & Barachant -- Transfer Learning Pipeline (Riemannian)

**Citation:** Jayaram, V. & Barachant, A. (2020). "Transfer Learning for Brain-Computer Interfaces: A Complete Pipeline." *Neuroinformatics*.
**Link:** DOI: 10.1007/s12021-019-09440-9
**Summary:** Systematic framework for transfer learning in BCI covering three levels: (1) feature-level transfer using Riemannian geometry with re-centering to align subject distributions, (2) instance-level transfer with importance weighting, (3) parameter-level transfer with fine-tuning. Riemannian alignment consistently improves cross-subject generalization.
**Relevance:** Riemannian alignment operates on covariance matrices (capturing spatial electrode relationships the project's graphs also capture). Re-centering aligns subject distributions without needing labeled target data. Strong baseline before complex adversarial approaches.
**Relevance score:** **HIGH**

---

### 4.2 Zheng & Lu -- DANN/DDC for EEG

**Citation:** Zheng, W.L. & Lu, B.L. (2020-2021). Various publications on adversarial domain adaptation for EEG in *IEEE Transactions on Affective Computing* and *Frontiers in Neuroscience*.
**Link:** Related: https://arxiv.org/abs/2004.01443
**Summary:** Applies Domain Adversarial Neural Networks (DANN) and Deep Domain Confusion (DDC with MMD loss) to cross-subject EEG. Shared feature extractor + domain discriminator trained adversarially to produce subject-invariant features. Multi-source variant handles multiple training subjects.
**Relevance:** Directly on the project roadmap (adversarial domain adaptation). The architecture template -- shared encoder + task classifier + domain discriminator -- is what to follow. Domain discriminator should distinguish subjects (not sessions), since subject identity dominates the feature space.
**Relevance score:** **HIGH**

---

### 4.3 Deep CORAL for EEG

**Citation:** Various groups (Li, Chen, et al.), 2020-2023. Published in *Biomedical Signal Processing and Control*, *IEEE Access*.
**Summary:** Applies CORrelation ALignment (CORAL) to cross-subject EEG. Deep CORAL minimizes the difference between second-order statistics (covariance matrices) of source and target domain feature distributions as a differentiable loss.
**Relevance:** CORAL is explicitly on the project roadmap. Since features vary most in covariance structure across subjects (as the statistical analysis confirmed), Deep CORAL can be integrated as an auxiliary loss. Computationally cheaper than MMD.
**Relevance score:** **HIGH**

---

### 4.4 Lee et al. -- Subject Adaptive BN

**Citation:** Lee, P., Hwang, S., Choi, S., & Byun, H. (2022). "Subject Adaptive EEG-based Visual Recognition." *arXiv / AAAI*.
**Link:** https://arxiv.org/abs/2110.11891
**Summary:** Test-time adaptation strategy: model adapts to new target subject using only unlabeled data by updating batch normalization running statistics while freezing all other parameters. Lightweight, requires no retraining.
**Relevance:** Aligns with "subject-specific normalization layers" on the roadmap. Low-effort, high-impact experiment: train GCN with BN, at test time update BN stats for the new subject. Batch normalization captures subject-specific distributional statistics.
**Relevance score:** **HIGH**

---

### 4.5 Bomatter et al. -- Instance Normalization for EEG

**Citation:** Bomatter, P., Paillard, J., Garces, P., Hipp, J., & Engemann, D.A. (2024). "Learning Domain-Independent EEG Representations with Instance Normalization." *arXiv*.
**Link:** https://arxiv.org/abs/2312.05275
**Summary:** Shows that Instance Normalization (normalizing each sample independently across channels) significantly improves domain generalization over Batch Normalization. Removes subject-specific amplitude and offset statistics, forcing the model to rely on relative patterns.
**Relevance:** Trivially easy to implement (one line of code). If per-trial normalization removes subject-specific signal, the remaining representation might reveal word-level structure. Directly addresses the finding that features are dominated by subject identity.
**Relevance score:** **HIGH**

---

### 4.6 Multi-Source Domain Adaptation for EEG

**Citation:** Sakhavi, S., Guan, C., & Yan, S. (2021-2023). Various publications in *IEEE TNSRE*.
**Summary:** Treats each source subject as a separate domain. Key approaches: weighted combination of source classifiers, shared-private architecture (shared encoder for subject-invariant features, private encoders for subject-specific patterns), curriculum training prioritizing similar source subjects.
**Relevance:** With multiple subjects, naive pooling has failed (chance level). Multi-source DA explicitly models that different source subjects may be more/less informative. Shared-private architecture architecturally separates the two sources of variation identified in the project.
**Relevance score:** **HIGH**

---

### 4.7 Domain Generalization for EEG BCI

**Citation:** Kang, T., Dong, S., & Kam, T.E. (2023-2024). Various publications in *IEEE TBME*, *Neural Networks*.
**Summary:** Domain generalization (no target data at training). Methods: meta-learning (MAML-based), gradient reversal on subject labels, augmentation simulating inter-subject variability, invariant risk minimization (IRM).
**Relevance:** Relevant when no adaptation data from new subjects is available. Gradient reversal on subject labels is the adversarial strategy on the roadmap framed as DG. Given few subjects, DG may be more practical than DA.
**Relevance score:** **MEDIUM-HIGH**

---

## Section 5: Self-Supervised & End-to-End Deep Learning for EEG

### 5.1 Lawhern et al. -- EEGNet

**Citation:** Lawhern, V.J., Solon, A.J., Waytowich, N.R., Gordon, S.M., Hung, C.P., & Lance, B.J. (2018). "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces." *Journal of Neural Engineering*, 15(5), 056013.
**Link:** https://arxiv.org/abs/1611.08024
**Summary:** Compact CNN using depthwise and separable convolutions to learn spatial and temporal filters from raw EEG. Parameter-efficient, generalizes across multiple BCI paradigms. The de facto standard end-to-end EEG baseline.
**Relevance:** Canonical baseline for end-to-end EEG classification. With data shaped as (59 channels, ~384 samples), EEGNet takes raw EEG, eliminating the 40 hand-crafted features. Lightweight enough for limited per-subject trial counts.
**Relevance score:** **HIGH**

---

### 5.2 Song et al. -- EEG Conformer

**Citation:** Song, Y., Zheng, Q., Liu, B., & Gao, X. (2023). "EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization." *IEEE TNSRE*, 31, 710-722.
**Link:** https://arxiv.org/abs/2210.01461
**Summary:** Combines convolutional module (local spatial-temporal features) with self-attention transformer module (long-range temporal dependencies). Achieves state-of-the-art on multiple EEG benchmarks with attention-based interpretability.
**Relevance:** Directly addresses the limitation of capturing temporal dynamics within 1.5s epochs. The transformer attention can learn which moments and spatial patterns are discriminative for imagined speech, potentially replacing the GCN + 5-window approach with a single architecture.
**Relevance score:** **HIGH**

---

### 5.3 Kostas et al. -- BENDR

**Citation:** Kostas, D., Aroca-Ouellette, S., & Bhatt, M. (2021). "BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn from Massive Amounts of EEG Data." *Frontiers in Human Neuroscience*, 15, 653659.
**Link:** https://doi.org/10.3389/fnhum.2021.653659
**Summary:** Adapts wav2vec 2.0/BERT paradigm to EEG. Convolutional encoder + transformer encoder trained with contrastive self-supervised objective (predicting masked segments). Pre-trained on ~2,500 hours of clinical EEG. Shows strong transfer to downstream tasks.
**Relevance:** Closest existing EEG foundation model. Self-supervised pre-training on all 70 subjects' data (ignoring word labels) could learn general representations disentangling subject identity from cognitive content. May need adaptation for 1.5s epoch length.
**Relevance score:** **HIGH**

---

### 5.4 Jiang et al. -- LaBraM (Large Brain Model)

**Citation:** Jiang, W., Zhao, L., & Lu, B.L. (2024). "Large Brain Model for Learning Generic Representations with Tremendous EEG Data." *ICLR 2024*.
**Link:** https://arxiv.org/abs/2405.18765
**Summary:** Large-scale EEG foundation model pre-trained on >2,500 hours using masked EEG modeling (like BERT/MAE). Neural tokenizer converts raw EEG to discrete tokens. Channel-agnostic representations transfer across different electrode montages.
**Relevance:** Directly addresses cross-subject generalization. Channel-agnostic tokenization handles the 59-channel montage. Pre-trained representations encode useful EEG structure. Fine-tuning on imagined speech data leverages general knowledge from thousands of hours.
**Relevance score:** **HIGH**

---

### 5.5 Yang et al. -- BIOT

**Citation:** Yang, C., Westover, M.B., & Sun, J. (2023). "BIOT: Cross-data Biosignal Learning in the Wild." *NeurIPS 2023*.
**Link:** https://arxiv.org/abs/2305.10351
**Summary:** Biosignal foundation model handling heterogeneous EEG data (varying channels, sampling rates, setups). Channel-tokenization approach makes the model montage-agnostic. Pre-trained on multiple large-scale EEG datasets.
**Relevance:** Addresses cross-dataset and cross-subject transfer. If leveraging external EEG datasets for pre-training beyond the 70 subjects, BIOT's architecture is designed for this cross-dataset scenario.
**Relevance score:** **MEDIUM-HIGH**

---

### 5.6 Shen et al. -- Subject-Invariant Contrastive Learning

**Citation:** Shen, X., Liu, X., Hu, X., Zhang, D., & Song, S. (2022). "Contrastive Learning of Subject-Invariant EEG Representations for Cross-Subject Emotion Recognition." *arXiv*, arXiv:2109.09559.
**Link:** https://arxiv.org/abs/2109.09559
**Summary:** Constructs positive pairs from different subjects performing the same task and negative pairs from different tasks. Explicitly encourages subject-invariant representations where task (not subject) structure dominates.
**Relevance:** Extremely relevant. The project's embeddings cluster by subject, not by word. This approach directly counters this by pushing same-word-different-subject trials together and different-word-same-subject trials apart. With 70 subjects, abundant cross-subject pairs are available.
**Relevance score:** **HIGH**

---

### 5.7 Rommel et al. -- EEG Data Augmentation (Systematic Comparison)

**Citation:** Rommel, C., Paillard, J., Moreau, T., & Gramfort, A. (2022). "Data augmentation for learning predictive models on EEG: a systematic comparison." *Journal of Neural Engineering*, 19(6), 066020.
**Link:** https://doi.org/10.1088/1741-2552/aca220
**Summary:** Systematic comparison of augmentation techniques: Gaussian noise, temporal shifting/cropping, channel dropout, amplitude scaling, time-frequency masking, smooth time warping. Simple augmentations (noise, scaling, cropping) consistently improve generalization. Complex augmentations (GANs) offer marginal additional benefit.
**Relevance:** The project currently has no augmentation. This paper provides an evidence-based ranking. For 1.5s epochs with 59 channels: start with (1) Gaussian noise, (2) channel dropout, (3) amplitude scaling, (4) temporal cropping.
**Relevance score:** **HIGH**

---

### 5.8 Eldele et al. -- Self-Supervised Contrastive for EEG

**Citation:** Eldele, E., Chen, Z., Liu, C., et al. (2023). "Self-Supervised Contrastive Representation Learning for EEG-Based Sleep Staging." *IEEE JBHI*, 27(5), 2285-2295.
**Link:** https://arxiv.org/abs/2110.15278
**Summary:** Self-supervised contrastive framework for EEG temporal signals. Creates positive pairs via augmentation (jittering, scaling, permutation) and negatives from different epochs. Temporal convolutional encoder learns representations then fine-tuned for classification.
**Relevance:** Concrete recipe for contrastive learning on EEG epochs. Augmentation strategies are directly applicable to 1.5s imagined speech epochs. Self-supervised pre-training on all unlabeled epochs could discover word-related structure.
**Relevance score:** **HIGH**

---

### 5.9 Eldele et al. -- TS-TCC

**Citation:** Eldele, E., Ragab, M., Chen, Z., et al. (2021). "Time-Series Representation Learning via Temporal and Contextual Contrasting." *IJCAI 2021*.
**Link:** https://arxiv.org/abs/2106.14112
**Summary:** Self-supervised framework using dual contrasting: temporal contrasting (augmented views via jittering/permutation) and contextual contrasting (transformer for contextual temporal patterns). Captures both local and global temporal structure.
**Relevance:** General self-supervised recipe validated on EEG. Applies directly to raw EEG epochs (59ch x ~384 timepoints). Dual contrasting captures both fine-grained dynamics within 1.5s epochs and broader patterns across trials.
**Relevance score:** **MEDIUM-HIGH**

---

### 5.10 Schirrmeister et al. -- Shallow/Deep ConvNet

**Citation:** Schirrmeister, R.T., Springenberg, J.T., et al. (2017). "Deep learning with convolutional neural networks for EEG decoding and visualization." *Human Brain Mapping*, 38(11), 5391-5420.
**Link:** https://doi.org/10.1002/hbm.23730
**Summary:** Systematic comparison of shallow and deep CNN architectures for raw EEG. Shallow ConvNet learns band-power features automatically; Deep ConvNet learns hierarchical temporal features. Both match or exceed hand-crafted approaches. Provides visualization showing networks learn physiologically meaningful features.
**Relevance:** Empirical justification for abandoning the 40 hand-crafted features. Shallow ConvNet is a useful stepping stone between feature-based and complex end-to-end approaches. Visualization methods can verify whether networks learn meaningful patterns from imagined speech data.
**Relevance score:** **MEDIUM**

---

### 5.11 Altaheri et al. -- ATCNet

**Citation:** Altaheri, H., Muhammad, G., et al. (2023). "Physics-Informed Attention Temporal Convolutional Network for EEG-Based Motor Imagery Classification." *IEEE Trans. Industrial Informatics*, 19(2), 2249-2258.
**Summary:** Combines multi-head self-attention with temporal convolutional networks (TCN). Sliding window creates temporal sub-segments, attention across segments, then temporal convolution. Learns optimal temporal windowing and weighting automatically.
**Relevance:** The sliding-window + attention approach suits the data well. The current manual 5-window representation could be replaced by ATCNet's learned windowing, revealing which portions of the 1.5s epoch carry the most discriminative information.
**Relevance score:** **MEDIUM-HIGH**

---

### 5.12 Mohsenvand et al. -- Contrastive Representation Learning for EEG

**Citation:** Mohsenvand, M.N., Izadi, M.R., & Maes, P. (2020). "Contrastive Representation Learning for Electroencephalogram Classification." *ML4H Workshop, NeurIPS 2020*.
**Summary:** Adapts SimCLR-like contrastive learning with EEG-specific augmentations (temporal jittering, channel dropout, noise injection, frequency perturbation). Learned representations capture subject-invariant features that transfer across tasks.
**Relevance:** Addresses two key challenges: (1) subject-invariant features reducing inter-subject variability, (2) leveraging the large 70-subject dataset for pre-training without labels. EEG-specific augmentations also address the "no augmentation" limitation.
**Relevance score:** **HIGH**

---

## Summary: Top Papers by Priority

### Must-Read (directly implement or build upon)

| Paper | Topic | Why |
|-------|-------|-----|
| Li et al. 2025 -- DHSLP/DHSLF | Hypergraph for imagined speech | Target architecture, 78% accuracy benchmark |
| Bomatter et al. 2024 -- Instance Norm | Normalization | 1-line change, could reveal word-level structure |
| Shen et al. 2022 -- Subject-Invariant Contrastive | Contrastive learning | Directly counters subject-dominated embeddings |
| Song et al. 2023 -- EEG Conformer | End-to-end DL | Replaces hand-crafted features, captures temporal dynamics |
| Song et al. 2020 -- DGCNN | Learnable graph | Next step from static/adaptive graphs |
| Chien et al. 2022 -- AllSet | Hypergraph framework | Flexible framework for hypergraph experimentation |
| Rommel et al. 2022 -- Augmentation | Data augmentation | Evidence-based augmentation strategy |
| Lawhern et al. 2018 -- EEGNet | End-to-end baseline | Standard baseline for comparison |

### Should-Read (strong relevance to specific roadmap items)

| Paper | Topic | Why |
|-------|-------|-----|
| Velickovic et al. 2018 -- GAT | Graph attention | On the roadmap (Fase 2) |
| Jayaram & Barachant 2020 -- Riemannian | Transfer learning | Principled alignment baseline |
| Lee et al. 2022 -- Adaptive BN | Subject adaptation | Low-effort, high-impact normalization |
| Zheng & Lu 2020 -- DANN for EEG | Adversarial DA | Directly on roadmap |
| Kostas et al. 2021 -- BENDR | EEG foundation model | Self-supervised pre-training recipe |
| Jiang et al. 2024 -- LaBraM | EEG foundation model | State-of-the-art pre-trained model |
| Gao et al. 2023 -- HGNN+ | Heterogeneous hypergraph | Multi-modal hyperedge construction |
| Zhao et al. 2023 -- Thinking Race | Contrastive for IS | Directly targets imagined speech + contrastive |

### Good-to-Know (context and alternatives)

| Paper | Topic | Why |
|-------|-------|-----|
| Feng et al. 2019 -- HGNN | Foundational hypergraph | Theoretical backbone |
| Panachakel & Ramakrishnan 2021 | Review | Establishes accuracy-vs-vocabulary tradeoff |
| Nieto et al. 2022 -- Inner Speech Dataset | Benchmark | Open benchmark, methodology reference |
| Pereira et al. 2018 -- Semantic Decoding | Semantic regression | Reframes problem via word embeddings |
| Cooney et al. 2020 -- Neurolinguistics | Brain regions | Informs electrode grouping for hypergraphs |
| Schirrmeister et al. 2017 -- ConvNets | End-to-end DL | Justifies moving beyond hand-crafted features |

---

*Generated from parallel literature search across 5 research domains. Verify all citations before use in thesis.*
