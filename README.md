# Music Genre Classification Using Multi-Branch Convolutional Neural Networks on the GTZAN Dataset

This project builds and compares multi-branch CNN architectures for music genre classification on the [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data) dataset (1000 × 30 s audio clips spanning 10 genres). It investigates in three model stages: a single-modality mel-spectrogram baseline, a three-branch fusion of mel/STFT/chromagram inputs, and the same fusion extended with hand-crafted MIR features which tests whether complementary acoustic representations yield additive discriminative signal under strict track-level evaluation.

> **Final result:** 81.11% top-1 accuracy on a strictly track-split test set, up from 75.96% for a single-modality mel-spectrogram baseline.

Full write-up available in [report/multimodal-music-genre-classifier.pdf](report/multimodal-music-genre-classifier.pdf). Trained checkpoints for all three stages, generated spectrograms, and processed tabular splits are mirrored on [Hugging Face](https://huggingface.co/datasets/tristantanjh/gtzan-multi-cnn) for fast reuse: see [Reproducing](#reproducing).

## Research overview

**Question.** Does fusing complementary acoustic representations improve genre classification beyond a single-modality spectrogram model, and does adding hand-crafted MIR features on top of learned spectrogram embeddings yield further gains?

**Hypothesis.** Mel, STFT and chromagram representations encode largely non-overlapping information (perceptual timbre, high-frequency spectral detail, and harmonic content respectively) so a multi-branch CNN that processes them in parallel should outperform a mel-only baseline. Hand-crafted MIR features (MFCCs, spectral centroid/rolloff/bandwidth, ZCR, RMS, tempo) summarise long-horizon statistics that a 3-second CNN window cannot easily capture, and should be additive when fused at the embedding level.

**Approach.** Three model stages of increasing modality, each evaluated on the same track-level 80/10/10 split to isolate the marginal contribution of each fusion step. ResNet-18 ImageNet pretraining is reused across all CNN branches under a common training protocol; a small tabular MLP encodes the MIR feature vector before late fusion with the CNN embedding.

**Findings.** Each added component yields a positive accuracy increment: +2.83 pp from spectrogram-modality fusion, +2.32 pp from MIR-feature fusion, finally reaching 81.11% test accuracy at Stage 3. The Stage 3 model trails the most upvoted Kaggle baseline (90.22%, XGBoost on tabular features), but the baseline applies `train_test_split` at the segment level and fits `MinMaxScaler` before splitting; both leak information and inflate the reported accuracy. The residual gap reflects a structural advantage of gradient-boosted trees on small, expert-designed feature matrices.

## Results

| Stage | Architecture | Test Accuracy |
|-------|--------------|---------------:|
| Stage 1 | Single-branch ResNet-18 on mel spectrograms | 75.96% |
| Stage 2 | Three-branch CNN fusion (mel + STFT + chroma) | 78.79% |
| Stage 3 | Three-branch CNN fusion + tabular MLP (54 MIR features) | **81.11%** |

Each added component yields a positive increment: +2.83 pp from the STFT and chroma branches, +2.32 pp from the hand-crafted MIR features.

### Per-genre breakdown (Stage 3)

| Genre | Precision | Recall | F1 |
|-------|----------:|-------:|---:|
| Blues | 0.81 | 0.91 | 0.85 |
| Classical | 0.93 | 0.94 | **0.94** |
| Country | 0.79 | 0.82 | 0.80 |
| Disco | 0.72 | 0.70 | 0.71 |
| Hip-hop | 0.89 | 0.78 | 0.83 |
| Jazz | 0.95 | 0.86 | **0.90** |
| Metal | 0.82 | 0.91 | 0.86 |
| Pop | 0.66 | 0.69 | 0.67 |
| Reggae | 0.87 | 0.88 | 0.88 |
| Rock | 0.71 | 0.63 | 0.67 |
| **Macro avg** | **0.81** | **0.81** | **0.81** |

<p align="center">
  <img src="figures/stage3_confusion.png" alt="Stage 3 confusion matrix on the held-out test set" width="520">
</p>

## Architecture

<p align="center">
  <img src="figures/architecture.png" alt="Three-stage multi-branch CNN architecture" width="820">
</p>

Three CNN stages share a ResNet-18 backbone (ImageNet-pretrained, progressively unfrozen) and a common training protocol:

- **Stage 1.** Mel spectrogram → ResNet-18 → GAP (512-d) → FC head (512 → 256 → 10).
- **Stage 2.** Three parallel ResNet-18 branches process mel, STFT and chromagram representations. The three 512-d embeddings are concatenated (1536-d) and fed to a shared fusion head (1536 → 512 → 256 → 10).
- **Stage 3.** Adds a fourth branch — a two-layer MLP (54 → 128) over hand-crafted MIR features (MFCCs, spectral centroid/rolloff/bandwidth, ZCR, RMS, tempo). The 128-d tabular embedding is concatenated directly with the 1536-d CNN embedding (1664-d total) before the FC head, so the head learns the optimal joint compression of both modalities.

The three modalities are chosen for complementarity:

| Representation | Resolution | Captures |
|----------------|-----------|----------|
| **Mel spectrogram** (128 bins) | Perceptual frequency | Timbre |
| **STFT spectrogram** (1025 bins) | Linear frequency | High-frequency spectral detail discarded by mel |
| **Chromagram** (12 pitch classes) | Western chromatic scale | Harmonic / chord vocabulary |

<p align="center">
  <img src="figures/spectrogram_trio.png" alt="Mel spectrogram, STFT spectrogram, and chromagram for the same 3-second segment" width="820">
</p>

All representations use $N_\text{fft}=2048$ and hop length $H=512$ at $f_s=22{,}050$ Hz, are resized to $128\times128$ RGB plasma-colourised images, and are ImageNet-normalised.

## Repository structure

```
multimodal-music-genre-classifier/
├── multimodal_music_genre_classification.ipynb   # End-to-end notebook (94 cells)
├── report/
│   ├── multimodal-music-genre-classifier.pdf     # Full project report
│   ├── report_shortened.tex                      # LaTeX source
│   └── references.bib
├── figures/                                       # Generated figures used in the report
│   ├── spectrogram_trio.pdf
│   ├── augmentation_examples.pdf
│   ├── pca_scatter.pdf
│   ├── correlation_heatmap.pdf
│   ├── feature_distributions.pdf
│   ├── stage{1,2,3}_curves.pdf
│   └── stage{1,2,3}_confusion.pdf
└── README.md
```

The notebook runs end-to-end: dataset exploration → cleaning → track-level splitting → spectrogram generation → augmentation → three model stages, each with training curves and a held-out confusion matrix.

## Reproducing

Spectrograms, augmented training set, processed tabular splits, and **trained checkpoints for all three stages** are mirrored on Hugging Face:

> [huggingface.co/datasets/tristantanjh/gtzan-multi-cnn](https://huggingface.co/datasets/tristantanjh/gtzan-multi-cnn)

Raw audio source: [GTZAN on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data) (1000 × 30 s WAVs, 3-second tabular CSV).

Tested with Python 3.10, PyTorch 2.x (CUDA), `librosa`, `scikit-learn`, `pandas`, `matplotlib`. Seed 42 is fixed for `torch` and `numpy`. Total training time ≈ 30–60 min per stage on a single consumer GPU.

## Citation

If you use this work, please cite:

```bibtex
@misc{tanjhmultimodalmusicclassification,
  author = {Tan, Tristan Jin Hau},
  title  = {Music Genre Classification Using Multi-Branch Convolutional Neural Networks on the GTZAN Dataset},
  year   = {2026},
  url    = {https://github.com/tristantanjh/multimodal-music-genre-classifier},
  note   = {Dataset and trained model checkpoints at https://huggingface.co/datasets/tristantanjh/gtzan-multi-cnn}
}
```

## License

Code, figures, and report released under the [MIT License](LICENSE). The underlying GTZAN audio is the property of its original authors (Tzanetakis & Cook, 2002) and is not redistributed here; see the [original Kaggle release](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data) for dataset terms.
