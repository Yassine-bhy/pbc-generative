# Synthetic data for rare blood-cell classes: utility and privacy

Does adding synthetic images from generative models (cVAE, cGAN) improve classification
of rare white-blood-cell types, and do those generators memorise their training images?

Dataset: PBC (Acevedo et al. 2020), Barcelona release, 8 classes.
Data is not included; set `ROOT` in each script to your copy.

- `PROTOCOL.md`: experimental decisions, each frozen before the experiment it governs
- `00_data_audit.py`: dataset audit (counts, duplicates, subtypes) → `excluded.txt`
- `data.py`: dataset class used by every experiment
- `sandbox/mnist/`: Phase A, the same pipeline built and debugged on MNIST.
  Not part of the results. `cVAE-cGAN.py` is unfinished.
