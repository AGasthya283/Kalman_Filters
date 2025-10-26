# 🛰️ Kalman Filters: Theory, Intuition, and Applications  
### *A Mathematical, Visual, and Practical Journey through Estimation and Tracking*

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/1d/Kalman_filter_estimation_example.gif" width="600" alt="Kalman Filter Tracking Visualization">
</p>

---

## 📘 Overview

This repository provides a **comprehensive and hands-on exploration of Kalman Filters (KFs)** — from the **classical Linear Kalman Filter** to advanced variants like the **Extended (EKF)** and **Unscented (UKF)** Kalman Filters.  

It bridges **mathematics, geometry, and implementation**, showing how these probabilistic filters elegantly merge **prediction and correction** across domains like **robotics, finance, and navigation**.

Each notebook blends:
- 🎓 **Rigorous derivations from Bayesian estimation theory**
- 📈 **Interactive and animated visualizations**
- 🤖 **Applications in robotics, tracking, and sensor fusion**
- 💹 **Extensions to nonlinear and stochastic systems (EKF, UKF, EnKF)**

All examples are **self-contained**, progressively deepening intuition behind state estimation, uncertainty propagation, and real-world filtering design.

---

## 🧭 Repository Structure
```
kalman-filters-tutorial/
│
├── 01_fundamentals/ # Linear Kalman Filter Foundations
│ ├── intro_to_state_estimation.ipynb # Probabilistic estimation and Bayes filtering basics.
│ ├── linear_kalman_filter.ipynb # Derivation and implementation of the standard KF.
│ ├── filter_vs_smoother.ipynb # Comparing filtering and smoothing approaches.
│
├── 02_nonlinear_filters/ # Extensions for Nonlinear Systems
│ ├── extended_kalman_filter.ipynb # EKF: Linearization-based nonlinear estimation.
│ ├── unscented_kalman_filter.ipynb # UKF: Sigma-point based estimation.
│ ├── ensemble_kalman_filter.ipynb # EnKF: Monte Carlo and particle-inspired filters.
│
├── 03_applications/ # Practical Applications Across Domains
│ ├── tracking_2d_object.ipynb # Object tracking in noisy 2D environments.
│ ├── sensor_fusion_robotics.ipynb # Sensor fusion for mobile robots and drones.
│ ├── finance_state_space_models.ipynb # Estimating latent variables in financial time series.
│ ├── gps_imu_fusion.ipynb # Real-world sensor fusion example (GPS + IMU).
│
├── 04_visualizations/ # Visualization Tools and Animations
│ ├── uncertainty_ellipses.ipynb # Visualizing Gaussian uncertainty propagation.
│ ├── sigma_point_visualization.ipynb # Demonstrating Unscented Transform geometry.
│ ├── prediction_update_animation.ipynb # Animations of the Kalman cycle (predict + correct).
│
├── 05_advanced_topics/ # Beyond Standard Filters
│ ├── information_filter.ipynb # Dual representation of KF using information form.
│ ├── square_root_filters.ipynb # Numerically stable Kalman variants.
│ ├── adaptive_and_augmented_filters.ipynb # Adaptive noise estimation, augmented state KFs.
│
├── assets/ # Supporting media for notebooks
│ ├── figures/ # Diagrams and static plots.
│ ├── gifs/ # Animated filter visualizations.
│
├── requirements.txt # List of Python dependencies.
└── README.md # This file.
```

---

## 🔬 Topics Covered

| Module | Theme | Highlights |
|--------|--------|-------------|
| **01. Fundamentals** | Linear estimation and probabilistic filtering. | Derivation of the classical Kalman Filter; prediction & correction cycle |
| **02. Nonlinear Filters** | Handling real-world nonlinear dynamics. | EKF, UKF, and EnKF with detailed derivations and visual intuition |
| **03. Applications** | Real-world domains from robotics to finance. | Object tracking, sensor fusion, and state-space modeling |
| **04. Visualizations** | Making uncertainty and estimation tangible. | Covariance ellipses, sigma points, and animated updates |
| **05. Advanced Topics** | Specialized and modern Kalman variants. | Information and square-root filters, adaptive models |

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

You’ll gain intuition into:

- Gaussian belief propagation
- Linearization (EKF) vs sigma-point sampling (UKF)
- Covariance geometry and uncertainty ellipses
- Information form and computational trade-offs

---
## 🎨 Visualization Examples
- Sigma-point transformations in high-dimensional Gaussian space
- Prediction vs correction animations for noisy trajectories
- Evolution of covariance ellipses during tracking
- Comparison of EKF vs UKF on nonlinear systems

<p align="center"> <img src="https://upload.wikimedia.org/wikipedia/commons/4/4b/Kalman_filter_animation.gif" width="600" alt="Kalman Filter Animation"> </p>

---
## 🧠 Learning Outcomes
After completing this repository, you will:
- Understand the mathematical core of Kalman filtering
- Visualize and interpret uncertainty propagation
- Apply filters to tracking, navigation, and time-series systems
- Compare the performance of linear, extended, and unscented filters
- Design your own sensor fusion pipelines

---
## 🧰 Tech Stack

- Python 3.11+
- Jupyter Notebooks
- NumPy, SciPy, Matplotlib, Plotly
- SymPy (for symbolic derivations)
- FilterPy / PyKalman (for reference implementations)
- Manim or matplotlib.animation for visualizations

---
## ⚙️ Installation
```bash
git clone https://github.com/AGasthya283/Kalman_Filters.git
cd kalman-filters-tutorial
pip install -r requirements.txt
```

---

## 📊 Example Usage
### Simulate a simple Kalman filter for a 1D motion model

```python
from kalman import LinearKalmanFilter

kf = LinearKalmanFilter(F=1, H=1, Q=0.1, R=0.2)
for z in [1.2, 0.9, 1.0, 1.1]:
    kf.predict()
    kf.update(z)
    print(kf.x, kf.P)
```

### Visualize Sigma Points (UKF)
```python
from visualization import plot_sigma_points
plot_sigma_points(mean=[0,0], cov=[[1,0.5],[0.5,1]])
```

---
## 📚 Suggested Reading

- Kalman (1960): A New Approach to Linear Filtering and Prediction Problems
- Julier & Uhlmann (1997): A New Extension of the Kalman Filter to Nonlinear Systems
- Simon (2006): Optimal State Estimation
- Maybeck (1979): Stochastic Models, Estimation, and Control
- Bar-Shalom et al. (2001): Estimation with Applications to Tracking and Navigation
- Van der Merwe (2004): Sigma-Point Kalman Filters for Probabilistic Inference

---
## 🚀 Roadmap
- Linear and Extended Kalman Filters
- Unscented and Ensemble Kalman Filters
- Square-Root and Information Filters
- Interactive Streamlit dashboard for real-time filtering demos
- Comparative visualizations of EKF vs UKF vs Particle Filter

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
Researcher in Estimation, Robotics, and Applied Probability

---
## 🧾 License

This repository is licensed under the MIT License.
Use freely for academic, educational, and research purposes.

---
## 🌟 Acknowledgements

Gratitude to the open-source community, researchers, and educators advancing the theory and practice of probabilistic estimation.

`“Kalman Filters are not just algorithms — they are the mathematics of trust under uncertainty.”`