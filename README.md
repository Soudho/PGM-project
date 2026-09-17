# Observation Resolution × Importance Sampling × Gradient Estimation

A research-grade probabilistic machine learning study of how **observation resolution**, **importance-sampling count**, and **gradient estimation** interact in variational image models.

The project implements and evaluates a fixed-architecture **VAE / IWAE / DReG-IWAE** pipeline on handwritten-image datasets. The central idea is to treat the observation model as an experimental factor rather than assuming that conclusions drawn from binarized MNIST automatically transfer to higher-resolution grayscale observations.

> **Study type:** Controlled empirical research / probabilistic generative modeling  
> **Primary metrics:** Test bits per dimension (bpd), convergence, stability, active latent units, runtime, and memory  
> **Main datasets:** MNIST and EMNIST-Letters in the reported results  
> **Framework:** PyTorch  
> **Primary reference:** `PGM project.pdf`

---

## 1. Project Overview

Importance-weighted variational objectives such as IWAE can produce tighter likelihood bounds as the number of importance samples `K` increases. However, larger `K` also changes the optimization problem and can make inference-network gradients harder to estimate. DReG (Doubly Reparameterized Gradient) was proposed to improve the inference-network gradient estimator.

This project asks whether those conclusions depend on **how the image likelihood is modeled**.

The same underlying 28×28 image is represented at three observation resolutions:

| Resolution `R` | Observation model | Interpretation |
|---:|---|---|
| `2` | Bernoulli | Binary image |
| `16` | Categorical(16) | 16-level grayscale |
| `256` | Discretized logistic | Full 8-bit grayscale |

The architecture is held fixed across these conditions as much as the likelihood heads permit:

```text
784
 │
 ├── Linear(200) + Tanh
 ├── Linear(200) + Tanh
 │
 └── Encoder → μ, log σ ∈ R²⁰
                  │
                  ▼
          z ~ N(μ, diag(σ²))
                  │
                  ▼
 ├── Linear(200) + Tanh
 ├── Linear(200) + Tanh
 │
 └── Resolution-specific likelihood head
```

For `K = 1`, the model is evaluated as the VAE baseline. For `K > 1`, the study compares standard IWAE and DReG-IWAE using the same forward samples and objective.

---

## 2. Research Questions

The paper frames four primary research questions.

### RQ1 — Observation complexity

Does moving from binary observations to 16-level and full 256-level observations systematically increase density-estimation complexity?

### RQ2 — IWAE vs. DReG

Does DReG improve the quality and optimization behavior of the importance-weighted objective compared with the standard IWAE gradient?

### RQ3 — Number of importance samples

Does increasing `K` consistently improve the bound, and when do the gains begin to diminish?

### RQ4 — Generalization across image content

Do the major effects observed on MNIST remain visible on a second handwritten-image dataset?

---

## 3. Experimental Design

The main design is fully crossed:

- Observation resolution: `R ∈ {2, 16, 256}`
- Training importance samples: `K ∈ {1, 5, 10, 20, 50}`
- Estimator for `K > 1`: `IWAE` or `DReG`
- Random seeds: `{1, 2, 3}`
- Evaluation importance samples: `K_eval = 1000`

Because `K = 1` is the VAE baseline and the IWAE/DReG distinction is only meaningful for `K > 1`, this produces:

```text
27 configurations × 3 seeds = 81 runs per dataset
```

The reported study therefore contains:

```text
81 MNIST main runs
81 EMNIST main runs
10 pilot runs
----------------
172 total experiment attempts
```

The reported paper states that all planned main-grid runs completed.

---

## 4. Model and Likelihoods

### Shared latent-variable model

The latent dimension is **hard-locked to 20** and is not treated as an experimental factor.

The encoder uses a diagonal Gaussian posterior:

```text
qφ(z | x) = N(μ(x), diag(σ²(x)))
```

with a standard normal prior:

```text
p(z) = N(0, I)
```

The decoder uses the same hidden-layer trunk for all resolutions; only the final likelihood head changes.

### Observation models

| `R` | Likelihood | Parameters per pixel | Decoder head |
|---:|---|---:|---|
| 2 | Bernoulli | 1 | Sigmoid/logit |
| 16 | Categorical(16) | 16 | Softmax logits |
| 256 | Discretized logistic | 2 | `tanh` location + `softplus` scale |

For the 256-level condition, the logistic density is integrated over each discrete intensity bin, producing a genuine probability mass for each observed pixel.

### Common evaluation metric

Because the three likelihoods operate on different alphabets, the project uses **bits per dimension (bpd)**:

```text
bpd = - log₂ p(x) / number_of_pixels
```

All configurations are evaluated using the same `K_eval = 1000` importance samples, with memory-safe chunking and exact `logsumexp`.

---

## 5. Training Protocol

The frozen training protocol includes:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Adam betas | `(0.9, 0.999)` |
| Adam epsilon | `1e-4` |
| Batch size | `128` |
| Epochs | `100` |
| Latent dimension | `20` |
| Weight initialization | Xavier uniform |
| Activation | `tanh` |
| Precision | float32 |
| TF32 | Disabled |
| Evaluation `K` | `1000` |
| Evaluation chunk | `50` |
| Bootstrap repetitions | `10,000` |
| Bootstrap seed | `424242` |
| Experimental seeds | `1, 2, 3` |

The notebook uses deterministic execution controls, seeds the relevant random-number generators, and records the software and benchmark environment for each run.

---

## 6. Pilot Gate and Reproducibility

The main grid is released only after a predeclared pilot gate passes.

The pilot checks:

1. Late-window convergence
2. Late-window validation stability
3. Gradient spikes
4. Finiteness of logged quantities

The gate is **fail-closed**: a completed pilot is not automatically considered successful.

The notebook also records and/or fingerprints:

- Source specification
- Training implementation
- Fixed training protocol
- Dataset bytes and split indices
- Software environment
- CUDA / GPU environment
- Benchmark environment

Completed runs can be resumed only when their metadata are compatible with the active scientific configuration. The experiment runner also supports sharding and recovery for long Kaggle sessions.

---

## 7. Dataset Handling

### MNIST

The reported study uses:

- 60,000 official training images
- Fixed split into 50,000 training and 10,000 validation images
- 10,000 held-out test images

The test set is kept sealed until checkpoint and configuration selection are complete.

### EMNIST

The notebook supports multiple EMNIST splits and explicitly records whether the chosen split conforms to the frozen protocol.

**Important reproducibility note:** the frozen notebook protocol specifies **EMNIST-Balanced**, but the reported experiments in the project paper were actually executed on **EMNIST-Letters**. The notebook flags this as a protocol deviation instead of silently treating the two as identical.

The README therefore describes the published result as:

> **consistency across MNIST digits and a harder set of handwritten characters**

rather than as a strict reproduction of the preregistered EMNIST-Balanced analysis.

---

## 8. Results Summary

### 8.1 Observation resolution has the largest effect

On MNIST, the best reported configurations reach:

| Resolution | Best reported test bpd |
|---:|---:|
| `R = 2` | `0.1324` |
| `R = 16` | `0.6028` |
| `R = 256` | `1.5787` |

The matched resolution gaps are approximately:

- `R = 2 → 16`: **+0.47 bpd**
- `R = 16 → 256`: **+0.98 bpd**

This effect is substantially larger than the estimator-side effects studied elsewhere in the project.

### 8.2 DReG generally improves the bound

Across the 24 matched `R × K` cells:

- DReG has lower bpd in **23 of 24** cells.
- The advantage is usually small, on the order of `10⁻³` bpd.
- The direction is much more consistent than the magnitude of the improvement.

The one clear reversal is the MNIST `R = 256, K = 10` cell, where IWAE performs better by approximately `1.51 × 10⁻³` bpd.

### 8.3 Increasing `K` helps, but gains diminish

The largest improvement occurs when moving from `K = 1` to the first multi-sample condition. Later gains are substantially smaller.

For MNIST, the improvement from the VAE baseline to DReG at `K = 50` is approximately:

| Resolution | Improvement |
|---:|---:|
| `R = 2` | `0.0072` bpd |
| `R = 16` | `0.0117` bpd |
| `R = 256` | `0.0462` bpd |

Thus, the absolute value of additional importance samples is much larger at fine observation resolution.

Using the project's predeclared saturation rule:

- `R = 2`: saturation around `K = 20`
- `R = 16`: saturation around `K = 20`
- `R = 256`: not saturated by `K = 50` on MNIST

### 8.4 Cross-dataset consistency

Across the matched contrast families for RQ1–RQ3:

```text
52 / 54 contrast pairs agree in sign
```

The resolution effect is especially consistent:

```text
18 / 18 agree in sign
18 / 18 are resolved on both datasets
```

The fine-grained three-way `R × K × estimator` interactions do **not** transfer reliably between the two datasets.

---

## 9. Stability and Computational Cost

DReG improves the achieved bound in most matched cells, but the stability story is resolution-dependent.

Across the main grid:

- DReG is more reproducible at `R = 2` and `R = 16`.
- DReG is less reproducible at `R = 256`.
- Neither estimator experienced a numerical-stability failure in the completed study.
- DReG required more wall-clock time to reach the selected checkpoint in all 24 matched comparisons.

The project reports DReG's additional training cost as roughly **5–59%**, depending on the condition.

Peak GPU memory is dominated by `K`. For the reported MNIST experiments, memory increased from roughly:

```text
82 MB at K = 1
→
522–561 MB at K = 50
```

The study therefore treats computational efficiency as a separate dimension from statistical quality.

---

## 10. Why the Results Matter

The central interpretation of the study is that the **observation model is not merely a preprocessing detail**.

Changing the observation likelihood changes the scale and dispersion of the importance weights. That, in turn, changes:

- How much tighter importance-weighted bounds can improve the objective
- How quickly additional importance samples pay off
- How concentrated the normalized importance weights become
- How difficult inference-network gradients are to estimate
- How much run-to-run variability appears at high resolution

This is why empirical conclusions calibrated only on binarized MNIST may underestimate the practical value of larger `K` for higher-resolution observations.

---

## 11. Repository Structure

A clean GitHub repository can be organized as:

```text
.
├── README.md
├── pgm_research_pilot_gate_fixed(1).ipynb
└── PGM project.pdf
```

The notebook is the executable research artifact; the PDF is the corresponding project report.

For a cleaner public repository, the files can optionally be renamed to:

```text
.
├── README.md
├── notebooks/
│   └── observation_resolution_iwae_dreg.ipynb
└── paper/
    └── PGM_Project_Report.pdf
```

Renaming is cosmetic and does not change the experiment.

---

## 12. Installation

The notebook uses the following Python ecosystem:

```text
Python 3.12.x
PyTorch
torchvision
NumPy
pandas
matplotlib
psutil
IPython / Jupyter
```

A minimal environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate

pip install torch torchvision numpy pandas matplotlib psutil ipython jupyter
```

For GPU execution, install the PyTorch build appropriate for the CUDA environment on the target machine.

The reported benchmark environment used:

- Python 3.12.3
- PyTorch 2.11
- CUDA 13.0
- NVIDIA RTX 4080 SUPER (16 GB)
- 28 logical CPU cores
- 31 GB RAM

Exact bitwise reproducibility is not guaranteed across different platforms, drivers, or software versions; the notebook therefore fingerprints the environment and treats those fingerprints as part of the scientific record.

---

## 13. Running the Notebook

### Recommended execution flow

1. Open `pgm_research_pilot_gate_fixed(1).ipynb` in Jupyter or Kaggle.
2. Review the **MASTER CONFIGURATION** cell.
3. Keep the frozen experimental ranges unless you intentionally want a different study.
4. Confirm the EMNIST split setting and understand that `letters` is the split used for the reported paper results.
5. Run the environment and dataset checks.
6. Execute the pilot gate.
7. Allow the main MNIST grid to complete.
8. Run the matched EMNIST experiment.
9. Run final aggregation, statistical analysis, visualizations, and the self-audit.
10. Inspect the generated CSV, PNG, JSON, checkpoint, and manifest artifacts.

The notebook deliberately withholds final scientific findings when the expected grid is incomplete or when execution is in debug mode.

---

## 14. Generated Artifacts

The experiment runner saves structured outputs under the configured results directory, including:

```text
latent_density_results/
└── research_grade_z20_v2/
    ├── experiments/
    └── sessions/
```

Depending on the executed phases, exported artifacts include:

- Configuration JSON files
- Environment snapshots
- Dataset statistics
- Frozen split indices
- Per-run configurations and checkpoints
- Per-run histories
- Aggregated result tables
- Resolution-effect comparisons
- IWAE vs. DReG comparisons
- Adjacent-`K` gain tables
- Saturation summaries
- Three-way interaction analyses
- Convergence plots
- Efficiency tables
- Best-configuration summaries
- SHA-256 artifact manifests
- Final self-audit information

---

## 15. Statistical Analysis

The project uses paired, descriptive bootstrap intervals rather than treating the study as a confirmatory hypothesis-testing exercise.

Key choices:

- 3 seeds per cell
- 10,000 bootstrap repetitions
- Seed `424242`
- Paired contrasts for matched configurations
- Explicit three-way `R × K × estimator` contrasts
- Mean ± standard deviation summaries
- A predeclared operational saturation rule

The paper explicitly treats these intervals as **exploratory summaries**.

---

## 16. Important Limitations

This repository should not be interpreted as establishing that resolution alone causally explains every observed difference.

### Resolution is coupled to the likelihood family

`R = 2`, `R = 16`, and `R = 256` use different observation likelihoods. The `R = 16` categorical head is also substantially larger than the Bernoulli head.

Therefore RQ1 estimates the effect of the **full observation-model condition**, not a likelihood-family-invariant causal effect of resolution alone.

### `K_eval = 1000` is still a lower bound

The evaluation metric is a finite-sample IWAE lower-bound estimate rather than exact log likelihood.

### Only three seeds are used per cell

The seed count is appropriate for descriptive paired uncertainty analysis but limits stronger statistical inference.

### The 100-epoch budget is binding

Many selected checkpoints occur late in training, especially at `R = 256`, so some high-resolution curves may not be fully converged.

### EMNIST protocol deviation

The frozen plan specifies EMNIST-Balanced, while the reported run used EMNIST-Letters. This is documented and should not be silently rewritten as a perfect preregistered replication.

### Limited architectural scope

The study uses one small MLP family, one latent dimension (`z = 20`), and one optimizer configuration. The findings should therefore be interpreted as controlled empirical evidence, not as a universal law for all generative models.

---

## 17. Key Takeaways

For a reader who only wants the main message:

1. **Observation complexity matters strongly.** Moving from binary to 16-level to full grayscale substantially increases the measured density-estimation difficulty.
2. **More importance samples help, especially at high resolution.** The benefit is strongly diminishing, but the high-resolution condition continues to benefit at larger `K`.
3. **DReG usually improves the bound, but the improvement is small.** Its stability advantage is not universal and reverses at the highest resolution.
4. **The major effects replicate across datasets.** The broad resolution, `K`, and estimator patterns are largely consistent between MNIST and the reported EMNIST character experiment.
5. **The observation model shapes importance-weight behavior.** That is the central probabilistic insight connecting the experimental factors.

---

## 18. References

The project builds on:

1. Kingma, D. P. & Welling, M. *Auto-Encoding Variational Bayes*. ICLR, 2014.
2. Burda, Y., Grosse, R. & Salakhutdinov, R. *Importance Weighted Autoencoders*. ICLR, 2016.
3. Rainforth, T. et al. *Tighter Variational Bounds Are Not Necessarily Better*. ICML, 2018.
4. Tucker, G., Lawson, D., Gu, S. & Maddison, C. J. *Doubly Reparameterized Gradient Estimators for Monte Carlo Objectives*. ICLR, 2019.
5. Roeder, G., Wu, Y. & Duvenaud, D. *Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference*. NeurIPS, 2017.
6. Theis, L., van den Oord, A. & Bethge, M. *A Note on the Evaluation of Generative Models*. ICLR, 2016.
7. Salimans, T. et al. *PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications*. ICLR, 2017.
8. Cohen, G. et al. *EMNIST: Extending MNIST to Handwritten Letters*. IJCNN, 2017.

See `PGM project.pdf` for the full methodology, tables, figures, discussion, limitations, and team contribution statement.

---

## 19. Authors

The project report credits:

- Mohammed Ali Hossain
- Md. Sudmun Hakim
- Ahtesham Ibne Mostafa
- Ipshita Bonhi Upoma — Faculty Advisor

Department of Computer Science and Engineering  
BRAC University  
Dhaka, Bangladesh
