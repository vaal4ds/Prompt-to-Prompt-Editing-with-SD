# Prompt to Prompt Editing with Stable Diffusion
This repository contains our experimental implementation of **Prompt-to-Prompt (P2P) editing** and **Null-Text Inversion (NTI)** applied to image generation using **Stable Diffusion** and to audio generation using **Riffusion**, for the MsC course in Statistical Machine Learning, 2024/2025.


---

## Authors
Valeria Avino,
Leonardo Rocci,
Riccardo Soleo.

## 📂 Repository Structure

```
.
├── riffusion/               # Custom implementation of Riffusion-based audio generation
├── nti_utils.py             # Adapted utilities for NTI (modified from existing repos)
├── ptp_utils.py             # Adapted utilities for P2P (modified from existing repos)
├── seq_aligner.py           # Utility functions for sequence alignment (adapted)
├── NTI.ipynb                # Notebook reproducing NTI experiments with custom prompts
├── P2P.ipynb                # Notebook reproducing Prompt-to-Prompt editing
├── P2P_riffusion.ipynb      # Notebook applying P2P to audio via Riffusion
├── example_images/          # Sample visualizations and reference examples
├── generated_audio.wav      # Example generated audio file
├── p2p_source_12345.wav     # Source audio before editing
├── p2p_edited_12345.wav     # Example of audio after P2P editing
└── __pycache__/             # Cached files
```

---

## 🚀 What We Implemented

* **From scratch**:

  * `riffusion/` – adaptation of Riffusion for audio editing tasks.
  * P2P and NTI notebooks with our own pipelines, prompts, and experiments.

* **Reused with modifications**:

  * `nti_utils.py`, `ptp_utils.py`, `seq_aligner.py`
    → originally from the public [Google Prompt-to-Prompt](https://github.com/google/prompt-to-prompt/) repo, adapted for compatibility with our hardware and newer model versions.

---

## 📓 Notebooks

* **`P2P.ipynb`** – Reproduction of Prompt-to-Prompt editing with our own experimental prompts.
* **`NTI.ipynb`** – Implementation of Null-Text Inversion with adapted utilities.
* **`P2P_riffusion.ipynb`** – Audio-focused editing pipeline combining P2P and Riffusion.

---
