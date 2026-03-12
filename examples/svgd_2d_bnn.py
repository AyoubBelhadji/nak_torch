"""
SVGD (Stein Variational Gradient Descent) for Bayesian Neural Networks
Datasets: Two Moons, Two Circles, Checkerboard/XOR, Spirals
Features:
  - Real SVGD particle updates
  - Predictive uncertainty heatmap (entropy)
  - Individual decision boundaries per particle
  - Mean prediction + epistemic uncertainty
  - OOD uncertainty visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import make_moons, make_circles

np.random.seed(42)

# ── Palette ────────────────────────────────────────────────────────────────────
BG    = "#0D0F18"
C0    = "#FF6B6B"   # class 0 – coral
C1    = "#7B61FF"   # class 1 – violet
C2    = "#00D4AA"   # teal – mean boundary
C4    = "#FFD166"   # gold
WHITE = "#EAEAEA"
GRAY  = "#2E3140"
LGRAY = "#555970"

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASETS
# ══════════════════════════════════════════════════════════════════════════════
def make_checkerboard(n=300, noise=0.05):
    X = np.random.rand(n, 2) * 2 - 1
    y = ((np.floor(X[:,0]*2) + np.floor(X[:,1]*2)) % 2).astype(int)
    X += np.random.randn(n, 2) * noise
    return X.astype(np.float32), y

def make_spirals(n=300, noise=0.4):
    def one_spiral(n, delta):
        t = np.linspace(0, 4*np.pi, n)
        r = t / (4*np.pi)
        x = r * np.cos(t + delta) + np.random.randn(n)*noise*0.1
        y = r * np.sin(t + delta) + np.random.randn(n)*noise*0.1
        return np.stack([x, y], axis=1)
    X = np.vstack([one_spiral(n//2, 0), one_spiral(n//2, np.pi)])
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X.astype(np.float32), y

X_moons,  y_moons  = make_moons(n_samples=300, noise=0.15)
X_circles,y_circles= make_circles(n_samples=300, noise=0.10, factor=0.4)
X_check,  y_check  = make_checkerboard(300)
X_spiral, y_spiral = make_spirals(300)

# Normalise all datasets to [-1.5, 1.5]
def normalise(X):
    X = X - X.mean(0)
    X = X / (X.std(0) + 1e-8)
    return X.astype(np.float32)

datasets = [
    ("Two Moons",    normalise(X_moons),   y_moons),
    ("Two Circles",  normalise(X_circles), y_circles),
    ("Checkerboard", normalise(X_check),   y_check),
    ("Spirals",      normalise(X_spiral),  y_spiral),
]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  BNN  (2 → 16 → 16 → 1,  sigmoid output)
# ══════════════════════════════════════════════════════════════════════════════
ARCH = [2, 32, 32, 1]

def unpack(flat):
    """flat weight vector → list of (W, b) per layer"""
    params, idx = [], 0
    for i in range(len(ARCH)-1):
        ni, no = ARCH[i], ARCH[i+1]
        W = flat[idx:idx+ni*no].reshape(no, ni); idx += ni*no
        b = flat[idx:idx+no];                    idx += no
        params.append((W, b))
    return params

D = sum(ARCH[i]*ARCH[i+1] + ARCH[i+1] for i in range(len(ARCH)-1))  # 337

def forward(flat, X):
    """X: (N,2) → (N,) logits"""
    h = X
    params = unpack(flat)
    for k, (W, b) in enumerate(params):
        h = h @ W.T + b
        if k < len(params)-1:
            h = np.tanh(h)
    return h.squeeze()

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

def log_likelihood(flat, X, y, sigma=1.0):
    logits = forward(flat, X)
    p = sigmoid(logits)
    return np.sum(y*np.log(p+1e-9) + (1-y)*np.log(1-p+1e-9))

def log_prior(flat, sigma=1.0):
    return -0.5 * np.sum(flat**2) / sigma**2

def log_posterior(flat, X, y):
    return log_likelihood(flat, X, y) + log_prior(flat)

def grad_log_posterior(flat, X, y):
    """Analytical backprop: grad of log p(w|D) = grad log-likelihood + grad log-prior."""
    params = unpack(flat)
    N = len(X)
    # forward pass
    h = X
    cache = []
    for k, (W, b) in enumerate(params):
        z = h @ W.T + b
        a = np.tanh(z) if k < len(params)-1 else z
        cache.append((h, z, a, W, b))
        h = a
    logits = h.squeeze()
    p = sigmoid(logits)
    # backward
    delta = (p - y).reshape(-1, 1) / N
    grads = []
    for k in reversed(range(len(params))):
        h_in, z, a, W, b = cache[k]
        dW = delta.T @ h_in
        db = delta.sum(0)
        grads.insert(0, (dW, db))
        if k > 0:
            _, _, a_prev, _, _ = cache[k-1]
            delta = (delta @ W) * (1 - a_prev**2)
    g_flat = np.concatenate([np.concatenate([dW.ravel(), db.ravel()])
                              for dW, db in grads])
    g_flat = -N * g_flat       # grad of log-likelihood
    g_flat -= flat             # grad of log-prior (sigma=1)
    return g_flat


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SVGD
# ══════════════════════════════════════════════════════════════════════════════
def rbf_kernel(X):
    """X: (N,D) → K (N,N), dK (N,N,D)"""
    diff  = X[:,None,:] - X[None,:,:]          # (N,N,D)
    sq    = (diff**2).sum(-1)                  # (N,N)
    h     = np.median(sq) / (2*np.log(len(X)+1)) + 1e-6
    K     = np.exp(-sq / h)
    dK    = -2/h * diff * K[:,:,None]          # (N,N,D) repulsive grad
    return K, dK

def svgd_step(particles, X, y, lr=0.05):
    scores = np.array([grad_log_posterior(p, X, y) for p in particles])
    K, dK  = rbf_kernel(particles)
    N      = len(particles)
    phi    = (K[:,:,None]*scores[None,:,:] + dK).mean(1)   # (N,D)
    return particles + lr * phi


def run_svgd(X, y, n_particles=20, n_iter=120, lr=0.05):
    particles = np.random.randn(n_particles, D) * 0.5
    for t in range(n_iter):
        # anneal lr slightly
        particles = svgd_step(particles, X, y, lr=lr*(0.995**t))
    return particles


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PREDICTION GRID
# ══════════════════════════════════════════════════════════════════════════════
GRID_RES = 80

def make_grid(X, margin=0.5):
    x1 = np.linspace(X[:,0].min()-margin, X[:,0].max()+margin, GRID_RES)
    x2 = np.linspace(X[:,1].min()-margin, X[:,1].max()+margin, GRID_RES)
    G1, G2 = np.meshgrid(x1, x2)
    Xg = np.stack([G1.ravel(), G2.ravel()], axis=1).astype(np.float32)
    return G1, G2, Xg

def predict_ensemble(particles, Xg):
    """Returns mean_prob (N_grid,) and entropy (N_grid,)"""
    probs = np.array([sigmoid(forward(p, Xg)) for p in particles])  # (P, N_grid)
    mean_p = probs.mean(0)
    entropy = -(mean_p*np.log(mean_p+1e-9) + (1-mean_p)*np.log(1-mean_p+1e-9))
    return mean_p, entropy, probs


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAIN ALL DATASETS
# ══════════════════════════════════════════════════════════════════════════════
print("Training SVGD on all datasets...")
results = []
CONFIGS = {
    "Two Moons":    dict(n_particles=20, n_iter=150, lr=0.05),
    "Two Circles":  dict(n_particles=20, n_iter=150, lr=0.05),
    "Checkerboard": dict(n_particles=30, n_iter=300, lr=0.04),
    "Spirals":      dict(n_particles=30, n_iter=500, lr=0.05),
}
for name, X, y in datasets:
    cfg = CONFIGS[name]
    N_PARTICLES = cfg['n_particles']
    print(f"  {name}  (particles={cfg['n_particles']}, iter={cfg['n_iter']})...", flush=True)
    particles = run_svgd(X, y, n_particles=cfg['n_particles'],
                         n_iter=cfg['n_iter'], lr=cfg['lr'])
    G1, G2, Xg = make_grid(X)
    mean_p, entropy, probs = predict_ensemble(particles, Xg)
    results.append(dict(
        name=name, X=X, y=y,
        G1=G1, G2=G2,
        mean_p=mean_p.reshape(GRID_RES, GRID_RES),
        entropy=entropy.reshape(GRID_RES, GRID_RES),
        probs=probs,   # (P, N_grid)
        particles=particles,
    ))
    print(f"    done, acc={((sigmoid(np.array([forward(p,X) for p in particles])).mean(0)>0.5)==y).mean():.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PLOT
# ══════════════════════════════════════════════════════════════════════════════
cmap_prob = LinearSegmentedColormap.from_list(
    "prob", [C0, "#1a1030", C1], N=256)
cmap_unc  = LinearSegmentedColormap.from_list(
    "unc",  [BG, "#1a1f3a", "#2d1b6e", C4, "#FFFFFF"], N=256)

fig = plt.figure(figsize=(22, 18), facecolor=BG)
fig.patch.set_facecolor(BG)

# 4 datasets × 3 panels (mean pred | uncertainty | individual boundaries)
outer = gridspec.GridSpec(4, 1, figure=fig,
                          hspace=0.08, left=0.04, right=0.98,
                          top=0.94, bottom=0.03)

for row, res in enumerate(results):
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[row], wspace=0.06)

    G1, G2   = res['G1'], res['G2']
    X, y     = res['X'], res['y']
    mean_p   = res['mean_p']
    entropy  = res['entropy']
    probs    = res['probs']

    # --- (a) Mean predictive probability ─────────────────────────────────────
    ax0 = fig.add_subplot(inner[0])
    ax0.set_facecolor(BG)
    ax0.contourf(G1, G2, mean_p, levels=50, cmap=cmap_prob,
                 vmin=0, vmax=1, alpha=0.92)
    ax0.contour(G1, G2, mean_p, levels=[0.5], colors=[WHITE],
                linewidths=1.8, alpha=0.9)
    # scatter
    ax0.scatter(X[y==0,0], X[y==0,1], c=C0, s=18, alpha=0.85,
                edgecolors='white', linewidths=0.3, zorder=5)
    ax0.scatter(X[y==1,0], X[y==1,1], c=C1, s=18, alpha=0.85,
                edgecolors='white', linewidths=0.3, zorder=5)
    ax0.set_xticks([]); ax0.set_yticks([])
    for sp in ax0.spines.values(): sp.set_edgecolor(GRAY)
    if row == 0:
        ax0.set_title("Mean prediction  p(y=1|x)", color=WHITE,
                      fontsize=11, fontweight='bold', pad=6)
    ax0.text(0.03, 0.96, res['name'], transform=ax0.transAxes,
             color=WHITE, fontsize=10, fontweight='bold', va='top',
             bbox=dict(facecolor=GRAY, alpha=0.6, pad=3, edgecolor='none'))

    # --- (b) Epistemic uncertainty (entropy) ─────────────────────────────────
    ax1 = fig.add_subplot(inner[1])
    ax1.set_facecolor(BG)
    im = ax1.contourf(G1, G2, entropy, levels=40, cmap=cmap_unc, alpha=0.93)
    ax1.contour(G1, G2, mean_p, levels=[0.5], colors=[WHITE],
                linewidths=1.2, alpha=0.5, linestyles='--')
    ax1.scatter(X[y==0,0], X[y==0,1], c=C0, s=18, alpha=0.7,
                edgecolors='none', zorder=5)
    ax1.scatter(X[y==1,0], X[y==1,1], c=C1, s=18, alpha=0.7,
                edgecolors='none', zorder=5)
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values(): sp.set_edgecolor(GRAY)
    if row == 0:
        ax1.set_title("Epistemic uncertainty  H[p(y|x)]", color=WHITE,
                      fontsize=11, fontweight='bold', pad=6)

    # --- (c) Individual particle decision boundaries ─────────────────────────
    ax2 = fig.add_subplot(inner[2])
    ax2.set_facecolor(BG)
    # faint individual boundaries
    for i, p_vec in enumerate(probs):
        pm = p_vec.reshape(GRID_RES, GRID_RES)
        ax2.contour(G1, G2, pm, levels=[0.5],
                    colors=[C2], linewidths=0.6, alpha=0.25)
    # bold mean boundary
    ax2.contour(G1, G2, mean_p, levels=[0.5],
                colors=[C4], linewidths=2.2, alpha=0.95)
    ax2.scatter(X[y==0,0], X[y==0,1], c=C0, s=18, alpha=0.85,
                edgecolors='none', zorder=5)
    ax2.scatter(X[y==1,0], X[y==1,1], c=C1, s=18, alpha=0.85,
                edgecolors='none', zorder=5)
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values(): sp.set_edgecolor(GRAY)
    n_p = len(res['particles'])
    if row == 0:
        ax2.set_title(f"Particle boundaries  (N={n_p})", color=WHITE,
                      fontsize=11, fontweight='bold', pad=6)

    # label teal/gold legend once
    if row == 3:
        from matplotlib.lines import Line2D
        els = [Line2D([0],[0], color=C2, lw=1.2, alpha=0.6, label='Particle boundary'),
               Line2D([0],[0], color=C4, lw=2.2,             label='Mean boundary')]
        ax2.legend(handles=els, framealpha=0.25, facecolor=GRAY,
                   edgecolor='none', labelcolor=WHITE, fontsize=8,
                   loc='lower right')

fig.suptitle(
    "SVGD  ·  Bayesian Neural Network  ·  Four 2D Datasets",
    color=WHITE, fontsize=15, fontweight='bold', y=0.975)

plt.savefig("/mnt/user-data/outputs/svgd_bnn_2d.pdf", dpi=160,
            bbox_inches='tight', facecolor=BG)
plt.savefig("/mnt/user-data/outputs/svgd_bnn_2d.png", dpi=160,
            bbox_inches='tight', facecolor=BG)
print("Saved!")

