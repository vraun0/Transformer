# Transformer

An **implementation (from scratch)** of the *Attention Is All You Need* (2017) transformer architecture in PyTorch. This project includes full implementations of the Multi-Head Attention, Encoder, and Decoder modules, and is trained on the IWSLT 2017 English–Italian translation dataset.

---

##  Table of Contents

- [Features](#features)  
- [Architecture](#architecture)  
- [Tech Stack](#tech-stack)  
- [Installation](#installation)  
- [Training the Model](#training-the-model)  
- [Translation / Inference](#translation--inference)  
- [Contributing](#contributing)  

---

##  Features

- From-scratch implementation of transformer components:  
  - Multi-Head Attention  
  - Positional Encoding  
  - Encoder & Decoder blocks  
- Trained on IWSLT 2017 English → Italian translation task
- Lightweight and educational—easy to trace and understand core transformer mechanics.

---

##  Architecture Overview

This project re-implements key modules from *Attention Is All You Need* in PyTorch, including:

- `MultiHeadAttention.py` – The attention mechanism  
- `PositionalEncoding.py` – Positional embeddings  
- `Embeddings.py`, `Encoder.py`, `Decoder.py`, `Transformer.py` – Complete model pipeline  
- `IWSLT_datamodule.py` – Data loader for IWSLT2017  
- `train.py` – Training script  
- `translator.py` – Model inference / translation interface  
- `config.py`, `helperFunctions.py` – Configuration and utility routines

---

##  Tech Stack

- **Language:** Python  
- **Framework:** PyTorch  
- **Dataset:** IWSLT 2017 (English ↔ Italian translation)

---

##  Installation

```bash
# Clone the repository
git clone https://github.com/vraun0/Transformer.git
cd Transformer

# (Optional) Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training the Model

``` bash
python train.py
```

--- 

## Translation/Inference

```bash
python translator.py
```

---

## Contributing

Contributions are welcome! To propose changes:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m "Add feature"`
4. Push to your branch: `git push origin feature-name`
5. Open a Pull Request—I'll be happy to review.

---

## Acknowledgements

- Based heavily upon _Attention Is All You Need_ (Vaswani et al., 2017)
- Dataset: IWSLT 2017 Machine Translation

---

## Contact

For any questions, feel free to raise an issue or reach out via GitHub Discussions.


