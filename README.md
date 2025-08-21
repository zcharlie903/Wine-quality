
# 🍷 Wine Quality Prediction — ML Ensembles & Deep Learning

Predict wine quality using classical ensemble models and advanced PyTorch deep learning approaches (MLP and multi-tower networks). With applied techniques like dropout, batch normalization, and early stopping — and validated through ablations — we improve predictive accuracy from ~50–60% to **~73%**.

---

## 📂 Project Structure

```
.
├── deep_learning_wine.py       # Core deep learning script (MLP + multi‑tower, ablations)
├── WIne_quality.ipynb          # Exploratory notebook + classical ML baselines
├── winequality-red.csv         # UCI Wine Quality dataset (red wine)
├── requirements.txt            # Python dependencies
└── README.md                   # (You’re here)
```

*Note: You can also add `winequality-white.csv` to extend the dataset.*

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/zcharlie903/Wine-quality.git
cd Wine-quality
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage Examples

### MLP with Dropout, BatchNorm, and Early Stopping
```bash
python deep_learning_wine.py   --csv winequality-red.csv --target quality   --model mlp --epochs 100 --patience 20   --hidden-sizes 128 64 --dropout 0.3 --batch-norm
```

### Multi‑Tower Network (3 towers)
```bash
python deep_learning_wine.py   --csv winequality-red.csv --target quality   --model multitower --towers 3 --tower-size 64   --no-batch-norm --dropout 0.0
```

### Full Ablation Study
```bash
python deep_learning_wine.py   --csv winequality-red.csv --target quality   --model mlp --ablate all
```

Results (best model, metrics, history, scaler) are saved in `runs/`.

---

## 📊 Results & Highlights

| Model Type                    | Accuracy Range |
|-------------------------------|----------------|
| Classical ML (SVM, Tree, kNN) | 50–60%         |
| Ensemble (Bagging/Boosting)   | ~65%           |
| **Deep Learning (MLP/Multi‑Tower)** | **~73%** |

- Applied **Dropout**, **Batch Normalization**, **Early Stopping**  
- Verified impact via **Ablation Studies**

---

## 🧪 Ablation Configurations

When running with `--ablate all`, the following are tested:

- **baseline**: BN ✅, Dropout=0.3 ✅, EarlyStopping ✅  
- **no_dropout**: BN ✅, Dropout ❌  
- **no_batchnorm**: BN ❌, Dropout=0.3 ✅  
- **no_earlystop**: EarlyStopping ❌  

Metrics are written under `runs/latest/summaries/`.

---

## 📘 Dataset

This project uses the [UCI Wine Quality dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality). Place the CSV in the project root as `winequality-red.csv`.

