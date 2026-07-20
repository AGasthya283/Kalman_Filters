# 🛰️ Kalman Filters: Theory, Intuition, and Applications  
### *A Mathematical, Visual, and Practical Journey through Estimation and Tracking*

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/1d/Kalman_filter_estimation_example.gif" width="600" alt="Kalman Filter Tracking Visualization">
</p>

---

## 📘 Overview

This repository provides a **comprehensive and hands-on exploration of Kalman Filters (KFs)** — from the **classical Linear Kalman Filter** to advanced variants like the **Extended (EKF)**, **Unscented (UKF)**, and **Ensemble (EnKF)** Kalman Filters, and on to the multi-target tracking and sensor-fusion problems that motivated this repository in the first place.

It bridges **mathematics, geometry, and implementation**, showing how these probabilistic filters elegantly merge **prediction and correction** across domains like **robotics, tracking, and navigation**.

Two notebooks in this repo (`03` and `04`) grew directly out of the author's M.Tech thesis research (DIAT-DRDO) on UAV-based maritime multi-object tracking and sensor fusion — the multi-target data-association problem and the AIS+vision fusion/trajectory-matching problem are the same ones tackled there, rebuilt here from scratch with synthetic data for a public, self-contained tutorial.

Every notebook is **self-contained** (no cross-notebook imports) and, wherever a notebook claims one filter beats another, converges, or fails, that claim is backed by a standalone, verified run of the code — not just asserted.

---

## 🧭 Repository Structure
```
Kalman_Filters/
│
├── 01_Linear_and_Extended_Kalman_Filters.ipynb    # Linear KF derivation + EKF for nonlinear (bearing-range) tracking
├── 02_Unscented_and_Ensemble_Kalman_Filters.ipynb # UKF (sigma points) vs EKF; EnKF on the chaotic Lorenz-63 system
├── 03_Multi_Target_Tracking_and_Data_Association.ipynb  # Per-track KFs, Hungarian/Mahalanobis gating, adaptive Q, 18-target benchmark
├── 04_Sensor_Fusion_and_Trajectory_Matching.ipynb # Geodetic coordinate transforms, async AIS+vision fusion, DTW trajectory matching
├── 05_Advanced_Filters.ipynb                      # Information filter, square-root filter, adaptive & augmented-state KFs
│
├── requirements.txt                               # 📦 Python dependencies (NumPy, SciPy, Matplotlib, ipywidgets)
├── LICENSE                                         # MIT
└── README.md                                       # 🧾 This file.
```

---

## 🔬 Topics Covered

| Notebook | Theme | Highlights |
|--------|--------|-------------|
| **01. Linear & Extended KF** | Foundations of Bayesian, linear-Gaussian state estimation. | Predict-update derivation, Joseph-form covariance update, EKF via Jacobian linearization, bearing-range tracking demo |
| **02. Unscented & Ensemble KF** | Estimation without linearization. | Scaled unscented transform (sigma points), EKF-vs-UKF RMSE by nonlinearity regime, EnKF on the chaotic Lorenz-63 attractor |
| **03. Multi-Target Tracking & Data Association** | Tracking many unlabeled targets at once. | Mahalanobis gating, Hungarian assignment, track birth/death/coasting, innovation-based adaptive process noise, an 18-simultaneous-target MOTA-style benchmark |
| **04. Sensor Fusion & Trajectory Matching** | Fusing asynchronous, heterogeneous sensors. | WGS84 geodetic→ENU coordinate transforms (validated against Vincenty's published test case), variable-Δt fusion of a sparse low-noise stream with a fast noisy stream, DTW vs. nearest-neighbor trajectory matching |
| **05. Advanced Filters** | Specialized and numerically robust KF forms. | Information (inverse-covariance) filter, square-root (Cholesky) filter under numerical stress, innovation-based adaptive noise estimation, augmented-state bias estimation |

---

## 🧮 Mathematical Depth

Each notebook builds from the **Bayesian estimation foundation**:

$$
p(x_k | z_{1:k}) = \frac{p(z_k | x_k) \, p(x_k | z_{1:k-1})}{p(z_k | z_{1:k-1})}
$$

and leads to recursive formulations of the predict-update cycle:

#### Prediction Step
$$
\hat{x}_{k|k-1} = F_k \hat{x}_{k-1|k-1} + B_k u_k
$$

#### Kalman Gain
$$
K_k = P_{k|k-1} H_k^T \left( H_k P_{k|k-1} H_k^T + R_k \right)^{-1}
$$

#### Update Step
$$
\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k \left( z_k - H_k \hat{x}_{k|k-1} \right)
$$

You'll gain intuition into:

- Gaussian belief propagation
- Linearization (EKF) vs. sigma-point sampling (UKF) vs. Monte Carlo ensembles (EnKF)
- Mahalanobis-gated data association and track-level noise adaptation
- Geodesy-correct sensor fusion across asynchronous, heterogeneous sources
- Information form, square-root numerical stability, and augmented-state estimation

A concrete example of the rigor this repo holds itself to: while building notebook 02, the original EKF-vs-UKF comparison seeded its scenarios with Python's built-in `hash()` on strings — which is randomized per process — silently making the "measured" RMSE improvement non-reproducible between runs. This was caught during verification and fixed with a deterministic integer seed scheme before the numbers were written into the markdown. Notebook 05 similarly *demonstrates* rather than asserts a real failure mode: under a deliberately ill-conditioned, reduced-precision (`float32`) setup, the naive covariance-form filter's `P` loses positive-definiteness by iteration 3 and fails a Cholesky check on 99.9% of steps, while the square-root filter stays valid throughout — the whole point of the notebook, proven by a real run, not just stated.

---

## 🎨 Visualization Examples
- Prediction vs. correction animations for noisy trajectories
- Sigma-point transformations and ensemble spread in the Unscented/Ensemble filters
- 18-target crossing/occlusion tracking scenes with track ID continuity
- Geodetic vs. local-tangent-plane trajectory overlays and DTW alignment paths
- Numerical-stability traces (minimum eigenvalue of `P` over time, naive vs. square-root)

<p align="center"> <img src="https://upload.wikimedia.org/wikipedia/commons/4/4b/Kalman_filter_animation.gif" width="600" alt="Kalman Filter Animation"> </p>

---
## 🧠 Learning Outcomes
After completing this repository, you will:
- Understand the mathematical core of Kalman filtering, from linear to nonlinear to ensemble forms
- Design and evaluate a multi-target tracker: gating, assignment, track lifecycle, adaptive noise
- Fuse asynchronous, heterogeneous sensors correctly, including the geodesy behind real-world coordinate transforms
- Recognize and defend against numerical-stability failure modes in production filtering code
- Compare filters honestly — including when a "better" method's advantage is small, or comes with a real tradeoff

---
## 🧰 Tech Stack

- Python 3.11+
- Jupyter Notebooks (Jupyter Lab / Notebook)
- NumPy, SciPy (`scipy.optimize.linear_sum_assignment` for data association, `scipy.stats`/linear algebra elsewhere)
- Matplotlib (static plots and animations)
- ipywidgets for interactive, in-notebook hyperparameter exploration

All filters (Linear, Extended, Unscented, Ensemble, information-form, square-root, adaptive, augmented-state) are implemented **from scratch in NumPy/SciPy** — no third-party Kalman-filter library does the actual estimation.

---
## ⚙️ Installation
```bash
git clone https://github.com/AGasthya283/Kalman_Filters.git
cd Kalman_Filters
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 📊 Example Usage

Every notebook redefines its filter classes locally (the repo is a set of self-contained tutorials, not an importable package). The API taught throughout looks like this, from `01_Linear_and_Extended_Kalman_Filters.ipynb`:

```python
kf = LinearKalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)
for z in measurements:
    kf.predict()
    kf.update(z)
    print(kf.x, kf.P)   # current state estimate and covariance
```

Later notebooks build on the same predict/update shape for `ExtendedKalmanFilter`, `UnscentedKalmanFilter`, `EnsembleKalmanFilter`, `InformationFilter`, and `SquareRootKalmanFilter`, and compose per-track instances of it for multi-target tracking in `03`.

---
## 📚 Suggested Reading

**Foundations**
- Kalman (1960): A New Approach to Linear Filtering and Prediction Problems
- Welch & Bishop (1995): An Introduction to the Kalman Filter (UNC Chapel Hill tutorial)
- Simon (2006): Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches
- Anderson & Moore (1979): Optimal Filtering
- Grewal & Andrews: Kalman Filtering: Theory and Practice

**Nonlinear & ensemble methods**
- Julier & Uhlmann (1997): A New Extension of the Kalman Filter to Nonlinear Systems
- Wan & Van der Merwe (2000): The Unscented Kalman Filter for Nonlinear Estimation
- Evensen (1994, 2003): The Ensemble Kalman Filter — formulation and practical implementation
- Burgers, van Leeuwen & Evensen (1998): Analysis Scheme in the Ensemble Kalman Filter
- Lorenz (1963): Deterministic Nonperiodic Flow

**Tracking, data association & sensor fusion**
- Bar-Shalom, Willett & Tian: Tracking and Data Fusion: A Handbook of Algorithms
- Kuhn (1955): The Hungarian Method for the Assignment Problem
- Bewley et al. (2016): Simple Online and Realtime Tracking (SORT)
- Wojke, Bewley & Paulus (2017): Simple Online and Realtime Tracking with a Deep Association Metric (DeepSORT)
- Zhang et al. (2022): ByteTrack — Multi-Object Tracking by Associating Every Detection Box
- Bernardin & Stiefelhagen (2008): Evaluating Multiple Object Tracking Performance (CLEAR MOT Metrics)
- Vincenty (1975): Direct and Inverse Solutions of Geodesics on the Ellipsoid
- Karney (2013): Algorithms for Geodesics, *Journal of Geodesy*
- Sakoe & Chiba (1978): Dynamic Programming Algorithm Optimization for Spoken Word Recognition
- Mehra (1970): On the Identification of Variances and Adaptive Kalman Filtering, IEEE TAC

---
## 🚀 Roadmap
- [x] Linear and Extended Kalman Filters (`01`)
- [x] Unscented and Ensemble Kalman Filters (`02`)
- [x] Multi-target tracking and data association (`03`)
- [x] Asynchronous sensor fusion and trajectory matching (`04`)
- [x] Information, square-root, adaptive, and augmented-state filters (`05`)
- [ ] Interactive Streamlit dashboard for real-time filtering demos
- [ ] Particle filter comparison for strongly non-Gaussian problems

---
<!-- ## 🧭 Contributing

Contributions are welcome!
If you have visualizations, derivations, or new applications:

- Fork the repository
- Create a feature branch
- Submit a pull request with a concise description

--- -->
## 🧑‍🏫 Author

### Agasthya

This repository grew out of M.Tech research (Applied Mathematics, DIAT-DRDO) on UAV-based maritime multi-object tracking and sensor fusion. Kalman filtering is the estimation backbone behind that research and behind the production tracking/vision systems this author has since built and maintained.

---
## 🧾 License

This repository is licensed under the MIT License.
Use freely for academic, educational, and research purposes.

---
## 🌟 Acknowledgements

Gratitude to the open-source community, researchers, and educators advancing the theory and practice of probabilistic estimation.

`"Kalman Filters are not just algorithms — they are the mathematics of trust under uncertainty."`
