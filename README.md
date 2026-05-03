<![CDATA[<div align="center">

# 🔬 Model Equation Prediction

### Recovering Physical Laws from Noisy Data Using Symbolic Regression, Neural Networks, SINDy & Physics-Informed Neural Networks

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![PySR](https://img.shields.io/badge/PySR-Symbolic_Regression-blueviolet?style=for-the-badge)](https://github.com/MilesCranmer/PySR)
[![PySINDy](https://img.shields.io/badge/PySINDy-SINDy-green?style=for-the-badge)](https://pysindy.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](#license)

<br/>

> **Can machines rediscover the laws of physics?**
> This project investigates whether data-driven modeling techniques can reconstruct a known physical equation — the freefall kinematic formula — from noisy observational data alone.

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Formulation](#-problem-formulation)
- [Project Pipeline](#-project-pipeline)
- [Data Generation](#-data-generation)
- [Modeling Approaches](#-modeling-approaches)
  - [1. PySR — Symbolic Regression](#1-pysr--symbolic-regression)
  - [2. Keras — Dense Neural Network](#2-keras--dense-neural-network)
  - [3. SINDy — Sparse Identification of Nonlinear Dynamics](#3-sindy--sparse-identification-of-nonlinear-dynamics)
  - [4. PINN — Physics-Informed Neural Network](#4-pinn--physics-informed-neural-network)
- [Results & Comparison](#-results--comparison)
- [Key Findings](#-key-findings)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Dependencies](#-dependencies)
- [License](#-license)

---

## 🎯 Overview

This project addresses a fundamental question in scientific machine learning: **given noisy measurements of a physical system, can we recover the underlying governing equation?**

We simulate a classic **freefall scenario** under constant gravitational acceleration and corrupt the measurements with Gaussian noise. Four distinct modeling paradigms are then applied to reconstruct the original equation from data:

| # | Method | Paradigm | Output Type |
|---|--------|----------|-------------|
| 1 | **PySR** | Symbolic Regression | Closed-form equation |
| 2 | **Keras NN** | Deep Learning | Black-box function approximator |
| 3 | **SINDy** | Sparse Dynamics Discovery | Differential equation (ODE) |
| 4 | **PINN** | Physics-Informed Deep Learning | Neural network + physics constraints |

Each approach offers unique strengths and limitations, providing a comprehensive landscape of modern equation discovery techniques.

---

## 🧮 Problem Formulation

### The Ground-Truth Equation

We consider an object in **freefall** from an initial height $h_0 = 100\,\text{m}$ with zero initial velocity. Under constant gravitational acceleration $g = 9.81\,\text{m/s}^2$, the height as a function of time is:

$$h(t) = h_0 - \frac{1}{2} g t^2 = 100 - 4.905\,t^2$$

### The Challenge

In real-world scenarios, measurements are never perfect. Sensor noise, environmental perturbations, and discretization errors introduce uncertainty. We simulate this by adding **Gaussian noise** ($\sigma = 1.0$) to the true height measurements:

$$h_{\text{noisy}}(t) = h(t) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

**Objective:** Recover $h(t)$ (or an accurate approximation) from $\{t_i, h_{\text{noisy}}(t_i)\}_{i=1}^{N}$.

---

## 🔄 Project Pipeline

The project follows a structured experimental pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PROJECT PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐ │
│   │  1. DATA      │    │  2. TRAIN /   │    │  3. EVALUATE &      │ │
│   │  GENERATION   │───▶│  FIT MODELS   │───▶│  COMPARE MODELS     │ │
│   └──────────────┘    └──────────────┘    └──────────────────────┘ │
│         │                    │                       │              │
│         ▼                    ▼                       ▼              │
│   • Simulate freefall  • PySR (Symbolic)     • MSE per model       │
│   • Add Gaussian noise • Keras NN (Dense)    • Prediction curves   │
│   • Train/Test split   • SINDy (Sparse ODE)  • CSV export          │
│                        • PINN (Physics NN)   • Visualization        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Generation

### Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Initial height | $h_0$ | 100 m |
| Gravitational acceleration | $g$ | 9.81 m/s² |
| Number of samples | $N$ | 100 |
| Time range | $t$ | $[0, \sim4.5]$ s |
| Noise standard deviation | $\sigma$ | 1.0 |

### Implementation

```python
import numpy as np

# Physical parameters
g = 9.81        # Gravitational acceleration (m/s²)
h0 = 100.0      # Initial height (m)
N = 100          # Number of data points

# Generate time vector
t = np.linspace(0, np.sqrt(2 * h0 / g), N)

# True height (freefall kinematics)
h_true = h0 - 0.5 * g * t**2

# Add Gaussian noise
noise = np.random.normal(0, 1.0, size=N)
h_noisy = h_true + noise
```

### Train/Test Split

The dataset is divided using scikit-learn's `train_test_split`:

- **Training set:** 80% of the data — used to fit all models
- **Test set:** 20% of the data — used for evaluation and comparison

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    t.reshape(-1, 1),
    h_noisy.reshape(-1, 1),
    test_size=0.2,
    random_state=42
)
```

---

## 🧠 Modeling Approaches

### 1. PySR — Symbolic Regression

**Paradigm:** Evolutionary search over mathematical expression space

[PySR](https://github.com/MilesCranmer/PySR) employs genetic programming to discover **closed-form mathematical expressions** that best fit the data. Unlike neural networks, symbolic regression produces interpretable, human-readable equations.

#### Configuration

```python
from pysr import PySRRegressor

pysr_model = PySRRegressor(
    niterations=40,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "sqrt", "sin", "cos"],
    model_selection="best",
    verbosity=0
)

pysr_model.fit(X_train, y_train)
```

#### Key Characteristics

| Feature | Detail |
|---------|--------|
| **Search method** | Multi-population evolutionary algorithm |
| **Operators (binary)** | `+`, `-`, `*`, `/` |
| **Operators (unary)** | `exp`, `log`, `sqrt`, `sin`, `cos` |
| **Iterations** | 40 |
| **Model selection** | Best (Pareto-optimal for accuracy vs. complexity) |
| **Output** | Symbolic mathematical expression |

#### Strengths & Limitations

| ✅ Strengths | ⚠️ Limitations |
|-------------|----------------|
| Produces interpretable equations | Computationally expensive for large search spaces |
| Can exactly recover physical laws | Non-deterministic (evolutionary search) |
| Pareto front balances accuracy vs. complexity | Requires careful operator selection |

---

### 2. Keras — Dense Neural Network

**Paradigm:** Universal function approximation via deep learning

A fully-connected (dense) neural network is trained to learn the mapping $t \mapsto h(t)$ directly from data. This is a **black-box** approach — the network learns an internal representation but does not produce an interpretable equation.

#### Architecture

```
Input (1) ──▶ Dense(64, ReLU) ──▶ Dense(128, ReLU) ──▶ Dense(64, ReLU) ──▶ Dense(32, ReLU) ──▶ Dense(1, Linear) ──▶ Output
```

#### Implementation

```python
import tensorflow as tf

model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model2.compile(optimizer='adam', loss='mse')
model2.fit(X_train, y_train, epochs=500, verbose=0)
```

#### Key Characteristics

| Feature | Detail |
|---------|--------|
| **Architecture** | 5-layer dense network (64 → 128 → 64 → 32 → 1) |
| **Activation** | ReLU (hidden layers), Linear (output) |
| **Optimizer** | Adam |
| **Loss function** | Mean Squared Error (MSE) |
| **Epochs** | 500 |
| **Output** | Continuous function approximation |

#### Strengths & Limitations

| ✅ Strengths | ⚠️ Limitations |
|-------------|----------------|
| Powerful function approximator | Black-box — no interpretability |
| Handles complex nonlinear relationships | Prone to overfitting on small datasets |
| Well-established framework (Keras/TF) | Extrapolation is unreliable |

---

### 3. SINDy — Sparse Identification of Nonlinear Dynamics

**Paradigm:** Sparse regression on a library of candidate functions

[SINDy](https://pysindy.readthedocs.io/) (Sparse Identification of Nonlinear Dynamical Systems) discovers governing **differential equations** from data. It constructs a library of candidate nonlinear functions and uses sparse regression (STLSQ) to identify which terms are active in the dynamics.

#### Configuration

```python
import pysindy as ps

model3 = ps.SINDy(
    feature_library=ps.PolynomialLibrary(degree=3),
    differentiation_method=ps.FiniteDifference(),
    optimizer=ps.STLSQ()
)

# Fit on state variables [h, v] with time vector
model3.fit(states, t=t_train)
```

#### Discovered System

The SINDy model identified the following dynamical system:

$$\dot{x}_0 = 1.000 \cdot x_1$$
$$\dot{x}_1 = 0.000$$

Where $x_0 = h$ (height) and $x_1 = v$ (velocity). This result indicates that SINDy correctly identified the velocity relationship ($\dot{h} = v$) but **failed to capture the gravitational acceleration** ($\dot{v} = -g$), essentially recovering a constant-velocity model instead of the expected freefall dynamics.

#### Key Characteristics

| Feature | Detail |
|---------|--------|
| **Feature library** | Polynomial (degree 3) |
| **Differentiation** | Finite Difference |
| **Optimizer** | STLSQ (Sequential Thresholded Least Squares) |
| **Output** | Sparse ODE system |
| **Test MSE** | **2254.84** |

#### Strengths & Limitations

| ✅ Strengths | ⚠️ Limitations |
|-------------|----------------|
| Discovers interpretable ODEs | Requires derivative estimation (noise-sensitive) |
| Sparse solutions enhance interpretability | Sensitive to noise levels and hyperparameters |
| Efficient for dynamical systems | Needs multi-state formulation (h, v) |

---

### 4. PINN — Physics-Informed Neural Network

**Paradigm:** Neural network constrained by known physical laws

A [Physics-Informed Neural Network](https://en.wikipedia.org/wiki/Physics-informed_neural_networks) incorporates the known physics (in this case, Newton's second law for freefall: $\ddot{h} = -g$) directly into the loss function. The network learns to satisfy both the data and the governing differential equation simultaneously.

#### Architecture

```
Input (t) ──▶ Dense(64, tanh) ──▶ Dense(64, tanh) ──▶ Dense(1, Linear) ──▶ Output (h)
```

#### Custom Physics-Informed Loss

```python
g = 9.81

def pinn_loss(t, h_true, keras_model):
    t = tf.cast(tf.convert_to_tensor(t), tf.float32)
    h_true = tf.cast(tf.convert_to_tensor(h_true), tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        h_pred = keras_model(t)
        h_t  = tape.gradient(h_pred, t)   # dh/dt  (velocity)
        h_tt = tape.gradient(h_t, t)      # d²h/dt² (acceleration)

    # Data fidelity loss
    data_loss = tf.reduce_mean((h_true - h_pred)**2)

    # Physics residual loss: d²h/dt² + g = 0
    physics_loss = tf.reduce_mean((h_tt + g)**2)

    return data_loss + physics_loss
```

#### Training Loop

```python
optimizer = tf.keras.optimizers.Adam(0.001)

for i in range(1000):
    with tf.GradientTape() as tape:
        loss = pinn_loss(X_train, y_train, model4)

    grads = tape.gradient(loss, model4.trainable_variables)
    optimizer.apply_gradients(zip(grads, model4.trainable_variables))
```

#### Training Progress

| Epoch | Total Loss |
|-------|-----------|
| 0 | 4761.65 |
| 100 | 3305.75 |
| 200 | 2736.76 |
| 300 | 2343.97 |
| 400 | 2066.91 |
| 500 | 1870.99 |
| 600 | 1735.33 |
| 700 | 1644.25 |
| 800 | 1585.27 |
| 900 | 1548.51 |

#### Key Characteristics

| Feature | Detail |
|---------|--------|
| **Architecture** | 2 hidden layers (64 neurons each, `tanh` activation) |
| **Optimizer** | Adam (lr = 0.001) |
| **Epochs** | 1000 |
| **Loss** | Data MSE + Physics residual ($h'' + g = 0$) |
| **Test MSE** | **1293.99** |

> ⚠️ **Note:** The high MSE indicates the PINN struggled to converge in this configuration. PINN performance is highly sensitive to network architecture, learning rate scheduling, and the balance between data and physics loss terms. Further hyperparameter tuning (e.g., more epochs, adaptive loss weighting, learning rate decay) could significantly improve results.

#### Strengths & Limitations

| ✅ Strengths | ⚠️ Limitations |
|-------------|----------------|
| Embeds physical laws into learning | Requires prior knowledge of the governing PDE/ODE |
| Can work with sparse / noisy data | Training is computationally expensive (double backprop) |
| Physics constraints regularize the solution | Sensitive to loss weighting and hyperparameters |

---

## 📈 Results & Comparison

All four models were evaluated on the **same held-out test set** (20% of data). Predictions were sorted by time for visualization and exported to `model_predictions_comparison.csv`.

### Sample Predictions (First 5 Test Points)

| Time (s) | True Noisy Height | PySR | Keras NN | SINDy | PINN |
|----------|-------------------|------|----------|-------|------|
| 0.000 | 100.06 | 100.00 | 101.36 | 100.00 | 49.85 |
| 0.202 | 98.72 | 99.80 | 100.00 | 99.95 | 50.92 |
| 0.505 | 97.44 | 98.75 | 97.96 | 99.87 | 51.77 |
| 0.606 | 98.00 | 98.19 | 97.29 | 99.85 | 51.94 |
| 0.909 | 95.34 | 95.94 | 95.25 | 99.77 | 52.24 |

### Performance Summary

| Model | Approach | Interpretable? | Test MSE | Verdict |
|-------|----------|:--------------:|----------|---------|
| **PySR** | Symbolic Regression | ✅ Yes | **Best** 🏆 | Closest to true equation |
| **Keras NN** | Deep Learning | ❌ No | Good | Strong interpolation |
| **SINDy** | Sparse Dynamics | ✅ Yes | 2254.84 | Failed to capture gravity |
| **PINN** | Physics-Constrained NN | ❌ No | 1293.99 | Needs hyperparameter tuning |

### Visualization

All model predictions are plotted against the true noisy height values in the notebook, producing a comparison chart with:

- **Black dashed line:** True Noisy Height (ground truth)
- **Blue line:** PySR Predictions
- **Red line:** Keras NN Predictions
- **Green line:** SINDy Predictions
- **Purple line:** PINN Predictions

---

## 🔑 Key Findings

### 1. PySR Excels at Equation Discovery
PySR successfully recovered a symbolic expression closely matching the true kinematic equation $h(t) = 100 - 4.905t^2$. This demonstrates the power of symbolic regression for **scientific discovery** — producing interpretable, generalizable equations from noisy data.

### 2. Keras NN Provides Strong Interpolation
The dense neural network achieved good accuracy within the training domain. However, as a black-box model, it offers **no insight into the underlying physics** and is unreliable for extrapolation beyond the training range.

### 3. SINDy Requires Careful Noise Handling
SINDy's reliance on **numerical differentiation** makes it inherently sensitive to measurement noise. The algorithm correctly identified the velocity relationship but failed to recover the gravitational acceleration term, resulting in a near-constant prediction. Potential improvements include:
- Smoothing the data before differentiation
- Using total-variation regularized differentiation
- Tuning the STLSQ threshold parameter

### 4. PINNs Need Extensive Hyperparameter Tuning
Despite embedding the correct physics ($h'' = -g$), the PINN underperformed due to insufficient training and suboptimal hyperparameters. The loss was still decreasing at epoch 1000, suggesting:
- **More epochs** (5,000–10,000) could improve convergence
- **Learning rate scheduling** (e.g., cosine annealing) may help escape local minima
- **Adaptive loss weighting** between data and physics terms could balance the optimization

### 5. Method Selection Depends on the Goal

| Goal | Best Method |
|------|-------------|
| Discover the governing equation | **PySR** |
| Accurate predictions (interpolation) | **Keras NN** |
| Discover dynamical system (ODE) | **SINDy** (with clean data) |
| Leverage known physics | **PINN** (with tuning) |

---

## 📁 Project Structure

```
model_equation_prediction/
│
├── model_equation_prediction.ipynb   # Main Jupyter notebook (full pipeline)
├── model_predictions_comparison.csv  # Exported comparison results
├── Maths.csv                         # Supplementary dataset
├── README.md                         # This file
│
└── outputs/                          # Generated during notebook execution
    ├── hall_of_fame_*.csv            # PySR Pareto-optimal expressions
    └── *.pkl                         # Saved PySR model checkpoints
```

---

## ⚙️ Installation & Setup

### Prerequisites

- **Python** ≥ 3.10
- **pip** (Python package manager)
- **Julia** (required by PySR — installed automatically on first run)
- **Jupyter Notebook** or **Google Colab**

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/model_equation_prediction.git
cd model_equation_prediction
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### Step 3: Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow pysr pysindy
```

> **Note:** PySR requires Julia. On first import, PySR will automatically download and configure Julia. This may take several minutes.

### Step 4: Launch the Notebook

```bash
jupyter notebook model_equation_prediction.ipynb
```

---

## 🚀 Usage

1. **Open** `model_equation_prediction.ipynb` in Jupyter Notebook or Google Colab
2. **Run all cells** sequentially (Cell → Run All)
3. **Observe** the data generation, model training, and evaluation outputs
4. **Review** the final comparison plot and `model_predictions_comparison.csv`
5. **Experiment** by modifying:
   - Noise level (`sigma`)
   - Number of data points (`N`)
   - PySR operators and iterations
   - Keras architecture (layers, neurons, activation functions)
   - SINDy library degree and optimizer thresholds
   - PINN epochs, learning rate, and loss weighting

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | Numerical computation |
| `pandas` | ≥ 2.0 | Data manipulation & CSV export |
| `matplotlib` | ≥ 3.7 | Plotting & visualization |
| `seaborn` | ≥ 0.12 | Statistical visualization |
| `scikit-learn` | ≥ 1.3 | Train/test split & metrics |
| `tensorflow` | ≥ 2.15 | Keras NN & PINN implementation |
| `pysr` | ≥ 0.16 | Symbolic regression |
| `pysindy` | ≥ 1.7 | SINDy algorithm |

---

## 📝 License

This project is developed as part of the **MTE 421 Mini-Project** coursework. It is provided for **educational and research purposes**.

---

<div align="center">

**Built with 🧪 Science & 🤖 Machine Learning**

*Exploring the frontier where data meets physics*

</div>
]]>
