"""
SVGD  vs  SVGD + Diversity Bonus  —  Bayesian Neural Network on Two Moons
=========================================================================
Panels
  A  Decision-boundary spread        (SVGD | Ours)
  B  Predictive-uncertainty heatmap  (SVGD | Ours)
  C  Particle trajectories in PCA weight-space
  D  Particle diversity over training (avg pairwise distance)
  E  Calibration curve               (SVGD | Ours)

Swap-in point: replace `ours_step()` with your own update rule.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from sklearn.datasets import make_moons
from sklearn.calibration import calibration_curve

np.random.seed(0)

# ── palette ───────────────────────────────────────────────────────────────────
BG    = "#0D0F18"
SVGD_C  = "#FF6B6B"   # coral  – SVGD
OURS_C  = "#00D4AA"   # teal   – Ours
C_C0    = "#A78BFA"   # class 0
C_C1    = "#FCD34D"   # class 1
WHITE   = "#EAEAEA"
GRAY    = "#2E3140"
LGRAY   = "#4B5060"

def sax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(BG)
    if title:  ax.set_title(title, color=WHITE, fontsize=10, fontweight='bold', pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=WHITE, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=WHITE, fontsize=8)
    ax.tick_params(colors=LGRAY, labelcolor=WHITE, labelsize=7.5)
    for sp in ax.spines.values(): sp.set_edgecolor(GRAY)
    ax.grid(True, color=GRAY, alpha=0.35, linewidth=0.5)

# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
X_raw, y = make_moons(n_samples=300, noise=0.15, random_state=1)
X = (X_raw - X_raw.mean(0)) / X_raw.std(0)
X = X.astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# BNN  (2 → 32 → 32 → 1)
# ══════════════════════════════════════════════════════════════════════════════
ARCH = [2, 32, 32, 1]
D = sum(ARCH[i]*ARCH[i+1] + ARCH[i+1] for i in range(len(ARCH)-1))

def unpack(flat):
    params, idx = [], 0
    for i in range(len(ARCH)-1):
        ni, no = ARCH[i], ARCH[i+1]
        W = flat[idx:idx+ni*no].reshape(no, ni); idx += ni*no
        b = flat[idx:idx+no];                    idx += no
        params.append((W, b))
    return params

def forward(flat, X):
    h = X
    params = unpack(flat)
    for k, (W, b) in enumerate(params):
        z = h @ W.T + b
        h = np.tanh(z) if k < len(params)-1 else z
    return h.squeeze()

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))

def grad_log_post(flat, X, y, prior_sigma=1.0):
    """Analytical backprop gradient of log p(w|D)."""
    params = unpack(flat)
    N = len(X)
    h, cache = X, []
    for k, (W, b) in enumerate(params):
        z = h @ W.T + b
        a = np.tanh(z) if k < len(params)-1 else z
        cache.append((h, z, a, W, b)); h = a
    p = sigmoid(h.squeeze())
    delta = (p - y).reshape(-1, 1) / N
    grads = []
    for k in reversed(range(len(params))):
        h_in, z, a, W, b = cache[k]
        dW = delta.T @ h_in; db = delta.sum(0)
        grads.insert(0, (dW, db))
        if k > 0:
            _, _, a_prev, _, _ = cache[k-1]
            delta = (delta @ W) * (1 - a_prev**2)
    g = np.concatenate([np.concatenate([dW.ravel(), db.ravel()]) for dW, db in grads])
    return -N*g - flat/prior_sigma**2

# ══════════════════════════════════════════════════════════════════════════════
# SVGD kernel
# ══════════════════════════════════════════════════════════════════════════════
def rbf_kernel(P):
    diff  = P[:,None,:] - P[None,:,:]
    sq    = (diff**2).sum(-1)
    h     = np.median(sq) / (2*np.log(len(P)+1)) + 1e-6
    K     = np.exp(-sq/h)
    dK    = -2/h * diff * K[:,:,None]
    return K, dK, h

# ══════════════════════════════════════════════════════════════════════════════
# SVGD step
# ══════════════════════════════════════════════════════════════════════════════
def svgd_step(P, X, y, lr):
    scores = np.array([grad_log_post(p, X, y) for p in P])
    K, dK, _ = rbf_kernel(P)
    phi = (K[:,:,None]*scores[None,:,:] + dK).mean(1)
    return P + lr*phi

# ══════════════════════════════════════════════════════════════════════════════
# OUR METHOD  — SVGD + explicit repulsion bonus
# -----------------------------------------------------------------------
# ▶▶ SWAP-IN POINT ◀◀
# Replace this function with your own update rule.
# Signature: ours_step(P, X, y, lr) → P_new  where P is (N_particles, D)
# ══════════════════════════════════════════════════════════════════════════════
def ours_step(P, X, y, lr, repulsion_strength=5.0):
    """
    SVGD + extra repulsion term that explicitly pushes particles apart
    in weight space, preventing mode collapse.

    The extra term adds  lambda * sum_j  (xi - xj) / ||xi - xj||^2
    (truncated at close range) to each particle's velocity.
    """
    scores = np.array([grad_log_post(p, X, y) for p in P])
    K, dK, h = rbf_kernel(P)

    # ── standard SVGD ──────────────────────────────────────────────────────
    phi_svgd = (K[:,:,None]*scores[None,:,:] + dK).mean(1)

    # ── diversity bonus: stronger repulsion via inverse-distance kernel ─────
    diff  = P[:,None,:] - P[None,:,:]           # (N,N,D)
    sq    = (diff**2).sum(-1, keepdims=True)     # (N,N,1)
    # normalised repulsion force (capped to avoid explosion)
    rep   = diff / (sq + h*0.1)                 # (N,N,D)
    # zero out self-interaction
    mask = np.eye(len(P), dtype=bool)
    rep[mask] = 0.0
    phi_rep = repulsion_strength * rep.mean(1)  # (N,D)

    return P + lr*(phi_svgd + phi_rep)

# ══════════════════════════════════════════════════════════════════════════════
# Training loop — records diversity history
# ══════════════════════════════════════════════════════════════════════════════
N_PARTICLES = 25
N_ITER      = 200
LR          = 0.04

def avg_pairwise_dist(P):
    diff = P[:,None,:] - P[None,:,:]
    sq   = (diff**2).sum(-1)
    N = len(P)
    return np.sqrt(sq[np.triu_indices(N,k=1)]).mean()

def run(step_fn, label):
    P = np.random.randn(N_PARTICLES, D) * 0.5
    diversity_hist = []
    snapshots = {}
    for t in range(N_ITER):
        lr_t = LR * (0.998**t)
        P = step_fn(P, X, y, lr_t)
        diversity_hist.append(avg_pairwise_dist(P))
        if t in [0, N_ITER//4, N_ITER//2, 3*N_ITER//4, N_ITER-1]:
            snapshots[t] = P.copy()
    acc = ((sigmoid(np.array([forward(p,X) for p in P])).mean(0)>0.5)==y).mean()
    print(f"  [{label}]  acc={acc:.2f}  final_diversity={diversity_hist[-1]:.3f}")
    return P, diversity_hist, snapshots

print("Training…")
P_svgd, div_svgd, snap_svgd = run(svgd_step, "SVGD")
P_ours, div_ours, snap_ours = run(ours_step,  "Ours")

# ══════════════════════════════════════════════════════════════════════════════
# Prediction helpers
# ══════════════════════════════════════════════════════════════════════════════
GRID = 120
mg  = 0.6
x1g = np.linspace(X[:,0].min()-mg, X[:,0].max()+mg, GRID)
x2g = np.linspace(X[:,1].min()-mg, X[:,1].max()+mg, GRID)
G1, G2 = np.meshgrid(x1g, x2g)
Xg = np.stack([G1.ravel(), G2.ravel()], 1).astype(np.float32)

def ensemble(P, Xg):
    probs = np.array([sigmoid(forward(p, Xg)) for p in P])
    mu  = probs.mean(0)
    ent = -(mu*np.log(mu+1e-9) + (1-mu)*np.log(1-mu+1e-9))
    return mu, ent, probs

mu_s, ent_s, probs_s = ensemble(P_svgd, Xg)
mu_o, ent_o, probs_o = ensemble(P_ours, Xg)

MU_S  = mu_s .reshape(GRID,GRID)
MU_O  = mu_o .reshape(GRID,GRID)
ENT_S = ent_s.reshape(GRID,GRID)
ENT_O = ent_o.reshape(GRID,GRID)

# PCA over all particles from both methods
all_P = np.vstack([P_svgd, P_ours])
mu_all = all_P.mean(0)
C = ((all_P-mu_all).T @ (all_P-mu_all)) / len(all_P)
evals, evecs = np.linalg.eigh(C)
pc1, pc2 = evecs[:,-1], evecs[:,-2]
def proj(P): c = P-mu_all; return c@pc1, c@pc2

# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════
cmap_unc  = LinearSegmentedColormap.from_list("unc",
    [BG,"#1a1f3a","#2d1b6e","#7B61FF","#FFD166","#FFFFFF"])
cmap_prob = LinearSegmentedColormap.from_list("prob",
    [C_C0, "#1a1030", C_C1])

fig = plt.figure(figsize=(20, 17), facecolor=BG)
fig.patch.set_facecolor(BG)

# ── layout ────────────────────────────────────────────────────────────────────
#   Row 0:  [A_svgd | A_ours  | C_pca            ]
#   Row 1:  [B_svgd | B_ours  | D_diversity      ]
#   Row 2:  [E_cal_svgd       | E_cal_ours        ]

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.38, wspace=0.28,
                       left=0.06, right=0.97, top=0.93, bottom=0.05)

# ── A: Decision boundaries ────────────────────────────────────────────────────
for col, (P, probs, mu, color, label) in enumerate([
        (P_svgd, probs_s, MU_S, SVGD_C, "SVGD"),
        (P_ours, probs_o, MU_O, OURS_C, "Ours")]):

    ax = fig.add_subplot(gs[0, col])
    sax(ax, f"{label}  —  Decision Boundaries")
    ax.set_facecolor("#080b12")

    # individual particle boundaries
    for p_vec in probs:
        pm = p_vec.reshape(GRID,GRID)
        ax.contour(G1, G2, pm, levels=[0.5], colors=[color],
                   linewidths=0.7, alpha=0.25)
    # mean boundary
    ax.contour(G1, G2, mu, levels=[0.5], colors=[WHITE],
               linewidths=2.2, alpha=0.95)

    ax.scatter(X[y==0,0], X[y==0,1], c=C_C0, s=22, alpha=0.9,
               edgecolors='white', linewidths=0.3, zorder=6)
    ax.scatter(X[y==1,0], X[y==1,1], c=C_C1, s=22, alpha=0.9,
               edgecolors='white', linewidths=0.3, zorder=6)
    ax.set_xticks([]); ax.set_yticks([])

    # diversity annotation
    spread = np.std([probs[i].reshape(GRID,GRID) for i in range(len(P))], axis=0).mean()
    ax.text(0.04, 0.05, f"Boundary spread: {spread:.3f}",
            transform=ax.transAxes, color=color, fontsize=8.5,
            bbox=dict(facecolor=GRAY, alpha=0.6, pad=3, edgecolor='none'))

    els = [Line2D([0],[0], color=color, lw=0.9, alpha=0.5,
                  label=f'Particle ({N_PARTICLES})'),
           Line2D([0],[0], color=WHITE, lw=2.0, label='Mean boundary')]
    ax.legend(handles=els, framealpha=0.2, facecolor=GRAY,
              edgecolor='none', labelcolor=WHITE, fontsize=7.5, loc='upper right')

# ── B: Uncertainty heatmaps ───────────────────────────────────────────────────
for col, (ent, mu, color, label) in enumerate([
        (ENT_S, MU_S, SVGD_C, "SVGD"),
        (ENT_O, MU_O, OURS_C, "Ours")]):

    ax = fig.add_subplot(gs[1, col])
    sax(ax, f"{label}  —  Epistemic Uncertainty  H[p(y|x)]")

    cf = ax.contourf(G1, G2, ent, levels=50, cmap=cmap_unc,
                     vmin=0, vmax=np.log(2), alpha=0.95)
    ax.contour(G1, G2, mu, levels=[0.5], colors=[WHITE],
               linewidths=1.6, alpha=0.7, linestyles='--')

    ax.scatter(X[y==0,0], X[y==0,1], c=C_C0, s=18, alpha=0.8,
               edgecolors='none', zorder=6)
    ax.scatter(X[y==1,0], X[y==1,1], c=C_C1, s=18, alpha=0.8,
               edgecolors='none', zorder=6)
    ax.set_xticks([]); ax.set_yticks([])

    mean_unc = ent.mean()
    ax.text(0.04, 0.05, f"Mean H: {mean_unc:.3f}",
            transform=ax.transAxes, color=color, fontsize=8.5,
            bbox=dict(facecolor=GRAY, alpha=0.6, pad=3, edgecolor='none'))

# shared uncertainty colorbar
cb_ax = fig.add_axes([0.445, 0.365, 0.008, 0.175])
sm = plt.cm.ScalarMappable(cmap=cmap_unc,
                            norm=plt.Normalize(0, np.log(2)))
cb = plt.colorbar(sm, cax=cb_ax)
cb.ax.yaxis.set_tick_params(color=WHITE, labelcolor=WHITE, labelsize=7)
cb.set_label("H (nats)", color=WHITE, fontsize=7, rotation=90)

# ── C: PCA particle trajectories ─────────────────────────────────────────────
axC = fig.add_subplot(gs[0, 2])
sax(axC, "Particle Trajectories in ℝᴰ  (PCA)", "PC₁", "PC₂")

snap_keys = sorted(snap_svgd.keys())
alphas_t  = np.linspace(0.2, 1.0, len(snap_keys))

for i, t in enumerate(snap_keys):
    # SVGD
    px, py = proj(snap_svgd[t])
    axC.scatter(px, py, c=SVGD_C, s=14+10*i, alpha=alphas_t[i],
                edgecolors='none', zorder=3)
    # Ours
    px, py = proj(snap_ours[t])
    axC.scatter(px, py, c=OURS_C, s=14+10*i, alpha=alphas_t[i],
                edgecolors='none', zorder=3)

# draw convex hull / ellipses for final state
from matplotlib.patches import Ellipse
for P_fin, col in [(P_svgd, SVGD_C), (P_ours, OURS_C)]:
    px, py = proj(P_fin)
    cx, cy = px.mean(), py.mean()
    sx, sy = px.std()*2.5, py.std()*2.5
    el = Ellipse((cx,cy), width=2*sx, height=2*sy,
                 fill=False, color=col, lw=1.5, ls='--', alpha=0.7)
    axC.add_patch(el)

var_exp = evals[-2:] / evals.sum() * 100
axC.set_xlabel(f"PC₁ ({var_exp[1]:.1f}% var)", color=WHITE, fontsize=8)
axC.set_ylabel(f"PC₂ ({var_exp[0]:.1f}% var)", color=WHITE, fontsize=8)

els = [Line2D([0],[0], marker='o', color=SVGD_C, ls='none', ms=6, label='SVGD'),
       Line2D([0],[0], marker='o', color=OURS_C,  ls='none', ms=6, label='Ours'),
       Line2D([0],[0], color=WHITE, lw=0, alpha=0, label='(dark=init, bright=final)')]
axC.legend(handles=els, framealpha=0.2, facecolor=GRAY,
           edgecolor='none', labelcolor=WHITE, fontsize=8)

# ── D: Diversity over training ────────────────────────────────────────────────
axD = fig.add_subplot(gs[1, 2])
sax(axD, "Particle Diversity Over Training",
    "Iteration", "Avg pairwise ‖wᵢ − wⱼ‖")

iters = np.arange(N_ITER)
axD.plot(iters, div_svgd, color=SVGD_C, lw=2.2, label='SVGD', zorder=4)
axD.plot(iters, div_ours, color=OURS_C,  lw=2.2, label='Ours',  zorder=4)

axD.fill_between(iters, div_svgd, div_ours,
                 where=np.array(div_ours)>np.array(div_svgd),
                 color=OURS_C, alpha=0.12, label='Diversity gain')

# annotation arrows
mid = N_ITER//2
axD.annotate('', xy=(mid, div_ours[mid]), xytext=(mid, div_svgd[mid]),
             arrowprops=dict(arrowstyle='<->', color=WHITE, lw=1.2))
gap = div_ours[mid] - div_svgd[mid]
axD.text(mid+4, (div_ours[mid]+div_svgd[mid])/2,
         f'+{gap:.1f}', color=WHITE, fontsize=8.5)

axD.legend(framealpha=0.2, facecolor=GRAY, edgecolor='none',
           labelcolor=WHITE, fontsize=8.5)

# ── E: Calibration curves ─────────────────────────────────────────────────────
axE = fig.add_subplot(gs[2, :])
sax(axE, "Calibration  —  Reliability Diagram  (closer to diagonal = better calibrated)",
    "Mean predicted probability", "Fraction of positives")

axE.plot([0,1],[0,1], color=WHITE, lw=1.2, ls=':', alpha=0.4, label='Perfect calibration')

for mu_flat, color, label in [(mu_s, SVGD_C, "SVGD"), (mu_o, OURS_C, "Ours")]:
    # predict on training data for calibration
    P_use = P_svgd if color == SVGD_C else P_ours
    probs_tr = np.array([sigmoid(forward(p, X)) for p in P_use]).mean(0)
    frac_pos, mean_pred = calibration_curve(y, probs_tr, n_bins=10)
    axE.plot(mean_pred, frac_pos, 'o-', color=color, lw=2.2,
             ms=7, label=label, zorder=5)
    axE.fill_between(mean_pred, frac_pos, mean_pred,
                     color=color, alpha=0.10)

# ECE
def ece(probs, y, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    ece_val = 0
    for lo,hi in zip(bins[:-1],bins[1:]):
        mask = (probs>=lo)&(probs<hi)
        if mask.sum()>0:
            acc = y[mask].mean()
            conf = probs[mask].mean()
            ece_val += mask.sum()*abs(acc-conf)
    return ece_val/len(y)

mu_s_tr = np.array([sigmoid(forward(p, X)) for p in P_svgd]).mean(0)
mu_o_tr = np.array([sigmoid(forward(p, X)) for p in P_ours]).mean(0)
ece_s = ece(mu_s_tr, y); ece_o = ece(mu_o_tr, y)
axE.text(0.72, 0.12,
         f"ECE  SVGD: {ece_s:.3f}   Ours: {ece_o:.3f}",
         transform=axE.transAxes, color=WHITE, fontsize=9.5,
         bbox=dict(facecolor=GRAY, alpha=0.7, pad=5, edgecolor='none'))

axE.set_xlim(0,1); axE.set_ylim(0,1)
axE.legend(framealpha=0.2, facecolor=GRAY, edgecolor='none',
           labelcolor=WHITE, fontsize=9)

# ── Supertitle & method labels ────────────────────────────────────────────────
fig.text(0.14, 0.955, "SVGD", color=SVGD_C, fontsize=13,
         fontweight='bold', ha='center')
fig.text(0.22, 0.955, "vs", color=LGRAY, fontsize=12, ha='center')
fig.text(0.30, 0.955, "Ours  (SVGD + Diversity)", color=OURS_C, fontsize=13,
         fontweight='bold', ha='center')
fig.text(0.65, 0.955,
         "Bayesian Neural Network  ·  Two Moons  ·  N=25 particles",
         color=LGRAY, fontsize=10, ha='center')

plt.savefig("svgd_vs_ours.pdf", dpi=170,
            bbox_inches='tight', facecolor=BG)
plt.savefig("svgd_vs_ours.png", dpi=170,
            bbox_inches='tight', facecolor=BG)
print("Saved!")
