# Residual-Quantized Variational Autoencoder

RQ-VAE is a modification of VQ-VAE (Vector Quantized-Variational Autoencoder).

RQ-VAE is proposed in [this paper](https://arxiv.org/abs/2203.01941). Its official implementation is found in [the link](https://github.com/kakaobrain/rq-vae-transformer).

## CPU-only environment setup

Create and activate a virtual environment, then install CPU-only dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook rqvae.ipynb
```
