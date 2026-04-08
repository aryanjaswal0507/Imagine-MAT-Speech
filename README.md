# 🧠 Imagine MAT: Signal Classification for Mental Activity

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-Signal--Classification-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

Imagine MAT is a cutting-edge research project focused on classifying EEG/sEMG signal data into actionable commands. By leveraging advanced feature engineering and machine learning, the system identifies distinct mental tasks or "silent speech" patterns across 8 unique classes.

---

## 🎯 Project Objectives

The core goal of this repository is to provide a robust pipeline for:
- Pre-processing raw signal data from Motor Imagery/Mental Activity tasks.
- Extracting discriminative features (Time, Frequency, and Time-Frequency domains).
- Optimizing sensor configurations through **Channel Selection**.
- Achieving high-accuracy classification using various ML architectures.

---

## 📊 Dataset & Classes

The system is trained and validated on a comprehensive dataset consisting of the following **8 Classes**:

| Class ID | Description | Mental Activity / Word |
| :--- | :--- | :--- |
| **(1)** | `a` | Vowel 'A' |
| **(2)** | `i` | Vowel 'I' |
| **(3)** | `u` | Vowel 'U' |
| **(4)** | `out` | Word 'Out' |
| **(5)** | `in` | Word 'In' |
| **(6)** | `up` | Word 'Up' |
| **(7)** | `cooperate` | Concept 'Cooperate' |
| **(8)** | `independent` | Concept 'Independent' |

---

## 🛠️ Project Structure

The repository is organized following a modular pipeline:

- `DATASET (OLD)/`: Original raw signal recordings.
- `WITH OVERLAP/` & `WITHOUT OVERLAP/`: Pre-processed datasets comparing windowing strategies.
- `5 sec/`: Data segments focused on a 5-second window.
- `new features/`: Extracted feature sets in `.csv` format (CSV per class).
- `channel selection/`: Optimized subsets of signal channels for reduced computational load.
- `1.ipynb`: Primary research notebook containing the pipeline from data loading to analysis.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook / VS Code
- Libraries: `numpy`, `pandas`, `sklearn`, `matplotlib`, `scipy`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/IMAGINE-MAT.git
   cd IMAGINE-MAT
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis:
   Open `1.ipynb` in your preferred environment and execute the cells.

---

## 🔬 Methodology

### 1. Pre-Processing
Signals are cleaned and segmented using various strategies, including **Overlapping vs. Non-Overlapping** windows, to ensure maximum data representation while minimizing noise.

### 2. Feature Extraction
The project utilizes a rich feature set extracted from the `new features/` directory:
- **Time Domain:** Root Mean Square (RMS), Variance, Zero Crossings.
- **Frequency Domain:** Power Spectral Density (PSD), Peak Frequency.
- **Specialized:** Feature Selection using specific channel importance.

### 3. Classification
Leveraging the `selected_features.csv`, models are trained to differentiate between the complex mental patterns of the 8 classes.

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

---

> [!TIP]
> For best performance when running the scripts, ensure you have allocated enough memory for processing the combined CSV files in the `new features/` directory.
