# Reconstructing heavy-ion charge state ratios and elemental composition of solar wind with interpretable deep learning

A TSMixer model that estimates solar wind ion composition ratios (C6+/C5+, O7+/O6+, Fe/O) from in-situ solar wind observations (ACE).

## Data

- `input/` : yearly 20-minute averaged solar wind data (1998–2011)
- `output/` : yearly ion composition ratios at 2-hour cadence
- `dataset_vF.npz` : training dataset built from the above with `data_preprocess.ipynb` (`inp`, `tar`, `tar_t`, `is_cme`)

| | Description |
|---|---|
| Input | 7 points at 20-minute cadence × 7 features: `Vp`, `Alpha_ratio`, `entropy`, `Alfvenicity`, `Np`, `Tp`, `coulomb num` |
| Target | one of `C6to5` (0), `O7to6` (1), `FetoO` (2) |

Preprocessing (`dataloader.py`)

- CME intervals are excluded, as are samples with a NaN target or with 50% or more NaN in any input feature
- Input: linear interpolation, then scaling by the per-feature maximum; the absolute value of Alfvenicity is used
- Target: log10 transform, then standardization
- Split: January–August of every year for training, September for validation, October–December for testing

## Requirements

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm torchinfo torchsummary
```


## Usage

Training is followed directly by prediction on the test set.

```bash
python main.py --ex_name oxygen_all --tar_idx 1
```

To use only some of the input features, pass their indices with `--inp_idx` (for example, without Alpha_ratio).

```bash
python main.py --ex_name oxygen_noalpha --tar_idx 1 --inp_idx 0 2 3 4 5 6
```

Main options: `--epochs` (100), `--batch_size` (128), `--lr` (0.0005), `--hidden_size` (512), `--n_blocks` (4)

Outputs

- `models/{ex_name}/best_model_state_dict.pth` : the model with the lowest validation loss
- `results/{ex_name}/TSMIXER/full_pred.npy` : test set predictions (standardized log10 values)


## Files

| File | Description |
|---|---|
| `main.py` | entry point for training and testing |
| `exp.py` | training, validation and test loops |
| `dataloader.py` | dataset loading and preprocessing |
| `models.py` | TSMixer, MLP and ResidualMLP model definitions |
