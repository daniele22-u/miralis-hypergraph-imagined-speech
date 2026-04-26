# Pathway B — Engineering & Generalization

Notebook di miglioramento ingegneristico, paralleli alla strada principale della tesi.
Obiettivo: migliorare la generalizzazione dei modelli baseline senza cambiare architettura.

## Roadmap

| Notebook | Contenuto | Stato |
|----------|-----------|-------|
| `B01_data_augmentation.ipynb` | Jittering temporale, crop random, channel dropout su SS | 🔲 |
| `B02_domain_adaptation_session.ipynb` | CORAL / MMD cross-sessione (stesso soggetto) | 🔲 |
| `B03_domain_adaptation_subject.ipynb` | Adversarial training cross-soggetto | 🔲 |
| `B04_contrastive_learning.ipynb` | Subject-invariant representations (Shen 2022) | 🔲 |

## Relazione con Pathway A

Pathway A (cartella `notebooks/`) segue la strada della tesi:
GCN → GAT → Hypergraph Neural Networks.

I risultati di Pathway B possono essere integrati come baseline comparativi
nella tesi, ma non sono il contributo principale.
