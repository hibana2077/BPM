# Document 1 — **Training & Validation Workflow**

> Project: **BPM — Batch Prototype Matching for Ultra-Fine-Grained Visual Classification (Ultra-FGVC)**

## 1. Overview

BPM learns to be invariant to *batch-level shared structure* by explicitly constructing a **batch prototype** (mean or low-rank subspace), subtracting it from each sample to extract a **residual view**, and aligning predictions/embeddings across original vs. residual views. A teacher–student self-distillation branch (EMA teacher) and variance/decorr regularization avoid collapse.

```
Images → Augment → (batch mean / PCA subspace) →
   ├─ Original view X ─────────────┐
   ├─ Prototype  X̄ or X̃          │
   └─ Residual  X−X̄ or X−X̃      │
                                  ▼
                fθ / hθ / gθ  + teacher ϕ (EMA)
                     │
        CE + Invariance + Uniformity + Self-Distill + Var&Decorr
                     │
                 Backprop (θ) & EMA update (ϕ)
```

Notation: model $f_\theta$ (backbone + head), embedding $h_\theta$, classifier $g_\theta$, prediction $p_\theta=\operatorname{softmax}(g_\theta(h_\theta(\cdot)))$. Teacher parameters $\phi$ updated by EMA.

## 2. Data pipeline

* **Input size**: e.g., 224×224 (align with baselines).
* **Augmentations** (applied per view): random resized crop, horizontal flip, color jitter, grayscale, Gaussian blur, random erase/cutout; keep identical strength as baselines for fairness.
* **Normalization**: dataset mean/std.
* **Batching**: shuffled, class-balanced sampler if available.

## 3. Batch prototype construction

Given a mini-batch $B=\{X_i,y_i\}_{i=1}^m$:

* **Mean prototype**: $\bar{X}=\tfrac{1}{m}\sum_i X_i$.
* **Momentum prototype (optional)**: global EMA $\mu\leftarrow \tau\mu+(1-\tau)\bar{X}$; use $\mu$ as the prototype to reduce batch noise.
* **Subspace prototype (optional)**: compute k-dim PCA subspace P on the batch (or on momentum buffer); project $\tilde{X}=PP^\top X$. Use residual $X-\tilde{X}$.

## 4. View generation

For each image:

* **Original view**: $v_A=t_a(X)$ (strong augmentation).
* **Residual view**: $v_B=t_b(X-\bar{X})$ or $t_b(X-\tilde{X})$.
* **Prototype view** (no label): optionally feed $t_c(\bar{X})$ or $t_c(\tilde{X})$ to enforce uniform predictions.

## 5. Forward pass (per iteration)

* **Student** (trainable): process $v_A$ and $v_B$ → embeddings and logits.
* **Teacher** (stop-grad): process $v_A$ (or a different augmentation of X) for self-distillation target.

## 6. Losses

Let $u$ be the uniform distribution over K classes.

* **Supervised**: $\mathcal{L}_{\text{ce}}=\mathrm{CE}(p_\theta(v_A),y)+\lambda_{\text{ce}}\,\mathrm{CE}(p_\theta(v_B),y)$.
* **Invariance** (prediction or embedding level): $\mathcal{L}_{\text{inv}}=\mathrm{KL}(p_\theta(v_A)\Vert p_\theta(v_B))$ or $\lVert h_\theta(v_A)-h_\theta(v_B)\rVert_2^2$.
* **Uniformity on prototype**: $\mathcal{L}_{\text{uni}}=\mathrm{KL}(u\Vert p_\theta(t_c(\bar{X})))$ (or with $\tilde{X}$).
* **Self-distillation (BYOL/SimSiam-style)**: $\mathcal{L}_{\text{sd}}=1-\cos(q_\theta(h_\theta(v_B)),\,\operatorname{sg}[h_\phi(v_A)])$.
* **Variance & Decorrelation** (VICReg/Barlow flavor) on paired embeddings: per-dim variance lower bound + off-diagonal covariance penalty.
* **Total**: $\mathcal{L}=\mathcal{L}_{\text{ce}}+\alpha\mathcal{L}_{\text{inv}}+\beta\mathcal{L}_{\text{uni}}+\gamma\mathcal{L}_{\text{sd}}+\mathcal{R}_{\text{Var\&Decorr}}$.

## 7. Optimization

* **Backprop** on $\theta$ with SGD or AdamW; cosine schedule + warmup.
* **EMA teacher**: $\phi\leftarrow m\,\phi+(1-m)\,\theta$ (e.g., m∈\[0.99,0.999]).
* **Mixed precision**, gradient clipping.

## 8. Validation protocol

* **Views**: use the **original image** only; optionally test-time augmentation (TTA). (Do **not** subtract prototype at test unless explicitly evaluated.)
* **Metrics**: Top-1/Top-5, macro-F1, per-class F1, calibration (ECE), confusion matrix. Report by subset (e.g., SoyAgeing, SoyGene, SoyLocal, SoyGlobal, Cotton80) and overall.
* **Model selection**: best on validation Top-1; ties broken by macro-F1 and ECE.

## 9. Checkpointing & logging

* Save $\theta$, $\phi$, optimizer, scheduler, RNG states.
* Log losses, metrics, learning-rate, gradient norms, EMA decay, batch prototype norms, variance/decorr stats.
* Periodically **visualize prototypes** ($\bar{X}/\tilde{X}$) and **CAM** differences between X vs. residual.

## 10. Reproducibility

* Fixed seeds (torch, numpy, cudnn), deterministic ops where possible.
* Config files (YAML) for all hyper-parameters and ablations.

---

# Document 2 — **Planned Experiments & Data to Collect**

## A. Baselines & main comparisons

1. **Backbones**: ResNet-50, Swin-B (or similar ViT).
2. **Training regimes**: Supervised only; Supervised + standard SSL (BYOL/SimCLR); **BPM** with mean prototype; BPM with momentum prototype; BPM with PCA subspace (k∈{4,8,16}).
3. **Datasets / splits**: Follow the official Ultra-FGVC protocol with its five subsets; maintain identical preprocessing to existing baselines.

## B. Ablations (one-factor-at-a-time unless noted)

* Remove each loss: $\mathcal{L}_{\text{inv}}$, $\mathcal{L}_{\text{uni}}$, $\mathcal{L}_{\text{sd}}$, $\mathcal{R}_{\text{Var\&Decorr}}$.
* Prototype type: mean vs. momentum vs. PCA subspace; PCA rank k sweep.
* Where to enforce invariance: logits KL vs. embedding L2.
* Batch size effect (since prototype quality is batch-dependent).
* Teacher momentum m sweep.
* With/without TTA at validation.
* Residual computed at pixel space vs. shallow feature space (optional variant).

## C. Robustness & generalization

* **Label noise** (symmetric/asymmetric).
* **Class imbalance** (long-tail splits).
* **Distribution shift**: cross-subset training/testing where appropriate.
* **Augmentation sensitivity**: weaken/strengthen augmentation policies.

## D. Efficiency & scaling

* Compute overhead of prototype and PCA (wall time/step, throughput, memory).
* Compare training cost vs. accuracy gains.

## E. Visual analyses

* Epoch-wise visualization of $\bar{X}/\tilde{X}$ and CAM heatmaps for X vs. residual.
* t-SNE/UMAP of embeddings: separation by class and by subset.
* Per-class error taxonomy: confusion with nearest fine-grained neighbors.

## F. Metrics to **collect & log**

**Per run** (store in CSV/JSON):

* Top-1/Top-5 accuracy, macro/micro-F1; per-class F1.
* Calibration: ECE, NLL; Brier score (optional).
* Loss components: CE, INV, UNI, SD, Var, Decorr (averages & curves).
* Prototype stats: $\lVert\bar{X}\rVert_2$, $\lVert X-\bar{X}\rVert_2$, PCA explained variance ratio.
* Efficiency: images/sec, step time, GPU memory peak.
* EMA coefficient m, gradient norms, parameter norms.

## G. Statistical testing & reporting

* 3 seeds per config; report mean±std.
* Paired t-test (or Wilcoxon) vs. strongest baseline on each subset.
* Consolidated tables per subset + overall; highlight best/second-best.

## H. Templates

* **Main results table**: Backbone × Method × Subset Top-1 / Macro-F1.
* **Ablation table**: component on/off, k, m, batch size → Top-1.
* **Cost table**: throughput, memory, train hours → Top-1.

---

# Document 3 — **Key Algorithms (Pseudocode)**

We use clear, implementation-ready pseudocode. `⊙` denotes element-wise ops; `@` denotes matrix multiply.

### Alg. 1: Mean prototype & residualization (per batch)

```
function MEAN_PROTOTYPE(B):
    # B = {X_i, y_i}_{i=1..m}, X_i ∈ R^{C×H×W}
    X_bar = (1/m) * sum_i X_i
    Residuals = { X_i - X_bar for i=1..m }
    return X_bar, Residuals
```

### Alg. 2: Momentum prototype buffer (optional)

```
state μ ← zeros_like(image_shape)
function MOMENTUM_PROTOTYPE(B, τ):
    X_bar, _ = MEAN_PROTOTYPE(B)
    μ ← τ * μ + (1-τ) * X_bar
    Residuals = { X_i - μ for i=1..m }
    return μ, Residuals
```

### Alg. 3: PCA subspace prototype (batch-wise)

```
function PCA_PROTOTYPE(B, k):
    # Flatten spatial dims; center by batch mean
    X_bar, _ = MEAN_PROTOTYPE(B)
    Z = [vec(X_i - X_bar) for i=1..m]  # shape m×D
    # Compute top-k eigenvectors of (1/m) Z^T Z  (use randomized SVD in practice)
    P = TOPK_EIGENVECTORS(cov=Z^T Z / m, k)
    # Project each sample onto subspace and compute residuals
    Proj = { reshape( P @ (P^T @ vec(X_i)) ) for i=1..m }
    Residuals = { X_i - Proj[i] for i=1..m }
    return P, Proj, Residuals
```

### Alg. 4: One training step of **BPM** (student θ, teacher ϕ)

```
input: batch B = {(X_i, y_i)}_{i=1..m}, params θ, teacher ϕ, coeffs (λ_ce, α, β, γ),
       proto_mode ∈ {mean, momentum(τ), pca(k)}

# 1) Build prototype & residuals
if proto_mode == mean:
    X_proto, Resid = MEAN_PROTOTYPE(B)
elif proto_mode == momentum(τ):
    X_proto, Resid = MOMENTUM_PROTOTYPE(B, τ)
else:  # pca(k)
    P, Proj, Resid = PCA_PROTOTYPE(B, k); X_proto = Proj_mean(ProJ)

# 2) Two views per sample
VA = { t_a(X_i) for X_i in images(B) }                  # original view
VB = { t_b(R_i) for R_i in Resid }                      # residual view
VP = t_c(X_proto)                                       # prototype view (unlabeled)

# 3) Forward
sA = model_student_forward(θ, VA)  # logits_A, emb_A
sB = model_student_forward(θ, VB)  # logits_B, emb_B
with no_grad:
    tA = model_teacher_forward(ϕ, VA)  # emb_A_teacher

# 4) Losses
L_ce  = CE(softmax(sA.logits), y) + λ_ce * CE(softmax(sB.logits), y)
L_inv = KL(softmax(sA.logits) || softmax(sB.logits))  # or L2(emb_A, emb_B)
L_uni = KL(uniform || softmax(model_student_forward(θ, VP).logits))
L_sd  = 1 - cosine(qθ(sB.emb), stopgrad(tA.emb))
L_var, L_decorr = VICReg_Barlow(emb_pairs=(sA.emb, sB.emb))
L = L_ce + α*L_inv + β*L_uni + γ*L_sd + (L_var + L_decorr)

# 5) Backprop & updates
optimizer.zero_grad(); L.backward(); optimizer.step()
ϕ ← m * ϕ + (1-m) * θ
return L
```

### Alg. 5: Validation step

```
function VALIDATE(batch, θ):
    VA = { t_val(X_i) for X_i in images(batch) }   # standard val augment
    logits = model_student_forward(θ, VA).logits
    preds = argmax(softmax(logits), axis=1)
    update_metrics(preds, y)
```

### Alg. 6: Inference

```
# Default: use original image only. Optional TTA.
function INFER(X, θ, TTA=False):
    if not TTA:
        return softmax(model_student_forward(θ, t_test(X)).logits)
    else:
        probs = average_over_augs( softmax(model_student_forward(θ, a(X)).logits) for a in TTA_AUGS )
        return probs
```

**Complexity notes.** Mean prototype is O(m·C·H·W). PCA adds SVD/Top-k cost; use randomized SVD or patch/feature-space PCA for efficiency.

---

# Document 4 — **Mathematical Derivations**

## Notation

* Batch $B=\{(X_i,y_i)\}_{i=1}^m$, $X_i\in\mathbb{R}^{C\times H\times W}$, $y_i\in\{1,\dots,K\}$.
* Prototype (mean): $\bar{X}=\tfrac{1}{m}\sum_i X_i$. Subspace prototype: let $P\in\mathbb{R}^{D\times k}$ have orthonormal columns; projection $\tilde{X}=\operatorname{reshape}(PP^\top \operatorname{vec}(X))$.
* Residuals: $R(X)=X-\bar{X}$ or $R(X)=X-\tilde{X}$.
* Model: embeddings $z=h_\theta(\cdot)$, logits $\ell=g_\theta(z)$, predictions $p_\theta=\operatorname{softmax}(\ell)$.

## 1) Objective function

$$
\mathcal{L}=\underbrace{\mathrm{CE}(p_\theta(X),y)+\lambda_{\mathrm{ce}}\,\mathrm{CE}(p_\theta(R(X)),y)}_{\mathcal{L}_{\mathrm{ce}}}
+\,\alpha\,\underbrace{\mathrm{KL}\big(p_\theta(X)\Vert p_\theta(R(X))\big)}_{\mathcal{L}_{\mathrm{inv}}}
+\,\beta\,\underbrace{\mathrm{KL}\big(u\Vert p_\theta(\text{Proto}(B))\big)}_{\mathcal{L}_{\mathrm{uni}}}
+\,\gamma\,\underbrace{\big(1-\cos(q_\theta(h_\theta(R(X))),\,\operatorname{sg}[h_\phi(X)]\big)}_{\mathcal{L}_{\mathrm{sd}}}
+\,\underbrace{\mathcal{R}_{\mathrm{Var}}+\mathcal{R}_{\mathrm{Decorr}}}_{\mathcal{R}_{\mathrm{V\&C}}}.
$$

## 2) Uniformity term equivalence

With K classes and $u_k=1/K$:

$$
\mathrm{KL}(u\Vert p)=\sum_{k=1}^K \frac{1}{K}\log\frac{1/K}{p_k}=\log K-\frac{1}{K}\sum_k\log p_k=\log K- H_{\mathrm{CE}}(u,p).
$$

Minimizing $\mathrm{KL}(u\Vert p)$ ⇔ **maximizing entropy** $H(p)$. Hence $\mathcal{L}_{\mathrm{uni}}$ encourages prototype predictions to be **uninformative**.

## 3) Subspace residualization

Let $P$ span the prototype subspace (shared structure). The **orthogonal projector** is $\Pi=PP^\top$. Residual operator $\mathcal{R}=I-\Pi$ is idempotent and symmetric: $\mathcal{R}^2=\mathcal{R}=\mathcal{R}^\top$. Then $\tilde{X}=\Pi X$, $R(X)=\mathcal{R}X$, and $\langle R(X), \tilde{X}\rangle=0$. This decomposes each image into shared vs. distinctive components.

## 4) Invariance and decision boundary

Consider embedding invariance penalty $\lVert h_\theta(X)-h_\theta(R(X))\rVert_2^2$. Assume $h_\theta$ is L-Lipschitz. First-order Taylor expansion around $X$ with $\Delta=-\Pi X$ yields

$$
\lVert h_\theta(X)-h_\theta(X+\Delta)\rVert_2\ \lesssim\ L\,\lVert\Delta\rVert_2 = L\,\lVert\Pi X\rVert_2.
$$

Minimizing $\mathcal{L}_{\mathrm{inv}}$ drives **insensitivity to perturbations within span(P)**, encouraging the classifier to place decision boundaries using features in the **orthogonal complement** (fine-grained details).

## 5) Collapse avoidance via variance & decorrelation

Let a batch of paired embeddings be $Z=[z_A, z_B]\in\mathbb{R}^{d\times 2m}$, centered. The penalties

$$
\mathcal{R}_{\mathrm{Var}}=\sum_{j=1}^d \max(0, \sigma_0-\mathrm{Std}(Z_j))^2,\qquad
\mathcal{R}_{\mathrm{Decorr}}=\sum_{i\neq j} \mathrm{Cov}(Z)_ {ij}^2
$$

ensure (i) per-dimension spread $\ge\sigma_0$ and (ii) near-diagonal covariance, preventing representational collapse despite uniformity pressure on prototypes and the no-negatives self-distillation.

## 6) Teacher–student dynamics

EMA update $\phi\leftarrow m\phi+(1-m)\theta$ yields

$$
\phi_t= m^t\phi_0 + (1-m)\sum_{s=1}^t m^{t-s}\,\theta_s,
$$

a low-pass filter over student iterates, supplying a smoother target for $\mathcal{L}_{\mathrm{sd}}$.

## 7) Efficiency considerations

* Mean prototype adds negligible cost (one reduction).
* PCA cost is dominated by top-k SVD on centered batch features; use randomized SVD, patch-wise, or feature-space PCA to keep overhead sublinear in image dimension.

## 8) Evaluation mapping

* Validation without residualization tests whether the **learned invariance** transfers to standard evaluation.
* Reporting macro-F1 and ECE complements Top-1 on highly fine-grained, imbalanced subsets.

---

**Suggested default hyper-parameters** (tune per dataset): $\lambda_{\mathrm{ce}}=1$, $\alpha\in[0.5,1.0]$, $\beta\in[0.1,0.5]$, $\gamma=1$; EMA $m\in[0.99,0.999]$; PCA rank $k\in\{4,8,16\}$. Backbone: ResNet‑50 and Swin‑B.
