"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         Modular BNN Particle Comparison Framework                           ║
║                                                                              ║
║  HOW TO ADD YOUR METHOD:                                                     ║
║  1. Define a function with signature:                                        ║
║       def my_method(P, grad_fn, t, cfg) -> P_new                            ║
║         P       : (N, D) current particles                                  ║
║         grad_fn : callable, grad_fn(P) -> (N, D) score gradients            ║
║         t       : int, current iteration                                     ║
║         cfg     : dict, your hyperparameters                                 ║
║                                                                              ║
║  2. Register it:                                                              ║
║       register_method("My Method", my_method, cfg={...}, color="#00D4AA")   ║
║                                                                              ║
║  3. Run:  python framework.py                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.datasets import make_moons, make_circles
from sklearn.calibration import calibration_curve

np.random.seed(0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def make_checkerboard(n=300, noise=0.05):
    X = np.random.rand(n, 2) * 2 - 1
    y = ((np.floor(X[:,0]*2) + np.floor(X[:,1]*2)) % 2).astype(int)
    X += np.random.randn(n, 2) * noise
    return X.astype(np.float32), y

def make_spirals(n=300, noise=0.08):
    def one(n, d):
        t = np.linspace(0, 4*np.pi, n)
        r = t / (4*np.pi)
        return np.stack([r*np.cos(t+d) + noise*np.random.randn(n),
                         r*np.sin(t+d) + noise*np.random.randn(n)], 1)
    X = np.vstack([one(n//2, 0), one(n//2, np.pi)])
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X.astype(np.float32), y

def normalise(X):
    return ((X - X.mean(0)) / (X.std(0) + 1e-8)).astype(np.float32)

DATASETS = {
    "two_moons":   lambda: (normalise(make_moons(300,   noise=0.15, random_state=1)[0]),
                             make_moons(300,   noise=0.15, random_state=1)[1]),
    "two_circles": lambda: (normalise(make_circles(300, noise=0.10, factor=0.4, random_state=1)[0]),
                             make_circles(300, noise=0.10, factor=0.4, random_state=1)[1]),
    "checkerboard":lambda: make_checkerboard(300),
    "spirals":     lambda: make_spirals(300),
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BAYESIAN NEURAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════

class BNN:
    """
    Small BNN with analytical backprop.
    Architecture: input_dim → hidden → hidden → 1  (tanh activations, sigmoid output)
    """
    def __init__(self, input_dim=2, hidden=32, prior_sigma=1.0):
        self.arch = [input_dim, hidden, hidden, 1]
        self.prior_sigma = prior_sigma
        self.D = sum(self.arch[i]*self.arch[i+1] + self.arch[i+1]
                     for i in range(len(self.arch)-1))

    def unpack(self, flat):
        params, idx = [], 0
        for i in range(len(self.arch)-1):
            ni, no = self.arch[i], self.arch[i+1]
            W = flat[idx:idx+ni*no].reshape(no, ni); idx += ni*no
            b = flat[idx:idx+no];                    idx += no
            params.append((W, b))
        return params

    def forward(self, flat, X):
        h, params = X, self.unpack(flat)
        for k, (W, b) in enumerate(params):
            z = h @ W.T + b
            h = np.tanh(z) if k < len(params)-1 else z
        return h.squeeze()

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

    def predict_proba(self, flat, X):
        return self.sigmoid(self.forward(flat, X))

    def grad_log_posterior(self, flat, X, y):
        """Gradient of log p(w|D) via backprop — O(N·D) not O(N·D²)."""
        params = self.unpack(flat)
        N = len(X)
        h, cache = X, []
        for k, (W, b) in enumerate(params):
            z = h @ W.T + b
            a = np.tanh(z) if k < len(params)-1 else z
            cache.append((h, z, a, W, b)); h = a

        p = self.sigmoid(h.squeeze())
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

        g = np.concatenate([np.concatenate([dW.ravel(), db.ravel()])
                             for dW, db in grads])
        return -N * g - flat / self.prior_sigma**2  # grad log-likelihood + grad log-prior


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUILT-IN METHODS
# ══════════════════════════════════════════════════════════════════════════════

def _rbf_kernel(P, h=None):
    """RBF kernel + gradient.  Returns K (N,N), dK (N,N,D), h (scalar)."""
    diff = P[:,None,:] - P[None,:,:]
    sq   = (diff**2).sum(-1)
    if h is None:
        h = np.median(sq) / (2 * np.log(len(P) + 1)) + 1e-6
    K  = np.exp(-sq / h)
    dK = -2/h * diff * K[:,:,None]
    return K, dK, h


def svgd(P, grad_fn, t, cfg):
    """
    Standard SVGD (Liu & Wang 2016).
    cfg keys:
        lr          : float, base learning rate  (default 0.04)
        lr_decay    : float, per-step decay      (default 0.998)
    """
    lr = cfg.get("lr", 0.04) * cfg.get("lr_decay", 0.998)**t
    scores    = grad_fn(P)                        # (N, D)
    K, dK, _  = _rbf_kernel(P)
    phi       = (K[:,:,None] * scores[None,:,:] + dK).mean(1)
    return P + lr * phi


def svgd_diversity(P, grad_fn, t, cfg):
    """
    SVGD + explicit inverse-distance repulsion bonus.
    cfg keys:
        lr                 : float  (default 0.04)
        lr_decay           : float  (default 0.998)
        repulsion_strength : float  (default 5.0)
    """
    lr     = cfg.get("lr", 0.04) * cfg.get("lr_decay", 0.998)**t
    rep_s  = cfg.get("repulsion_strength", 5.0)

    scores   = grad_fn(P)
    K, dK, h = _rbf_kernel(P)
    phi_svgd = (K[:,:,None] * scores[None,:,:] + dK).mean(1)

    diff = P[:,None,:] - P[None,:,:]
    sq   = (diff**2).sum(-1, keepdims=True)
    rep  = diff / (sq + h * 0.1)
    rep[np.eye(len(P), dtype=bool)] = 0.0
    phi_rep = rep_s * rep.mean(1)

    return P + lr * (phi_svgd + phi_rep)


# ── placeholder so users can see the exact signature ─────────────────────────
def your_method_template(P, grad_fn, t, cfg):
    """
    ▶ COPY THIS FUNCTION AND RENAME IT.

    Parameters
    ----------
    P        : np.ndarray, shape (N_particles, D)
               Current particle positions in weight space.
    grad_fn  : callable  →  (N_particles, D)
               grad_fn(P) returns ∇_w log p(w|D) for every particle at once.
    t        : int
               Current iteration index (0-based). Useful for lr schedules.
    cfg      : dict
               Your hyperparameters, passed from register_method(..., cfg={}).

    Returns
    -------
    P_new    : np.ndarray, shape (N_particles, D)
               Updated particle positions.
    """
    lr = cfg.get("lr", 0.04) * cfg.get("lr_decay", 0.998)**t

    # Example: gradient ascent on log-posterior (no interaction)
    scores = grad_fn(P)
    return P + lr * scores


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — METHOD REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

_REGISTRY = []   # list of dicts: {name, fn, cfg, color}

def register_method(name, fn, cfg=None, color=None):
    """
    Register a particle-update method for comparison.

    Parameters
    ----------
    name  : str   — display name on the figure
    fn    : callable(P, grad_fn, t, cfg) → P_new
    cfg   : dict  — hyperparameters forwarded to fn
    color : str   — hex color for this method's plots
    """
    default_colors = ["#FF6B6B", "#00D4AA", "#7B61FF", "#FFD166",
                      "#4FC3F7", "#F97316", "#A78BFA"]
    color = color or default_colors[len(_REGISTRY) % len(default_colors)]
    _REGISTRY.append(dict(name=name, fn=fn,
                          cfg=cfg or {}, color=color))


# ── Register the built-in methods ─────────────────────────────────────────────
register_method("SVGD",
                svgd,
                cfg={"lr": 0.04, "lr_decay": 0.998},
                color="#FF6B6B")

register_method("SVGD + Diversity",
                svgd_diversity,
                cfg={"lr": 0.04, "lr_decay": 0.998,
                     "repulsion_strength": 5.0},
                color="#00D4AA")

# ══════════════════════════════════════════════════════════════════════════════
# ▼▼▼  ADD YOUR METHOD HERE  ▼▼▼
# ══════════════════════════════════════════════════════════════════════════════
#
# def my_method(P, grad_fn, t, cfg):
#     ...
#     return P_new
#
# register_method("My Method", my_method, cfg={"lr": 0.04}, color="#FFD166")
#
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class Experiment:
    def __init__(self, dataset="two_moons", n_particles=25, n_iter=200,
                 hidden=32, prior_sigma=1.0, grid_res=100):
        self.dataset    = dataset
        self.n_particles = n_particles
        self.n_iter     = n_iter
        self.grid_res   = grid_res

        X, y = DATASETS[dataset]()
        self.X, self.y  = X, y
        self.bnn        = BNN(input_dim=2, hidden=hidden,
                               prior_sigma=prior_sigma)

        # prediction grid
        m = 0.6
        x1 = np.linspace(X[:,0].min()-m, X[:,0].max()+m, grid_res)
        x2 = np.linspace(X[:,1].min()-m, X[:,1].max()+m, grid_res)
        self.G1, self.G2 = np.meshgrid(x1, x2)
        self.Xg = np.stack([self.G1.ravel(),
                             self.G2.ravel()], 1).astype(np.float32)

    def _batch_grad(self, P):
        """Vectorised score: grad log p(wᵢ|D) for all particles."""
        return np.array([self.bnn.grad_log_posterior(p, self.X, self.y)
                         for p in P])

    def run_method(self, entry):
        """Train one registered method, return results dict."""
        fn, cfg = entry["fn"], entry["cfg"]
        P       = np.random.randn(self.n_particles, self.bnn.D) * 0.5

        diversity_hist = []
        snapshots      = {}
        snap_iters     = {0, self.n_iter//4, self.n_iter//2,
                          3*self.n_iter//4, self.n_iter-1}

        for t in range(self.n_iter):
            P = fn(P, self._batch_grad, t, cfg)
            # avg pairwise distance
            diff = P[:,None,:] - P[None,:,:]
            sq   = (diff**2).sum(-1)
            idx  = np.triu_indices(len(P), k=1)
            diversity_hist.append(np.sqrt(sq[idx]).mean())
            if t in snap_iters:
                snapshots[t] = P.copy()

        # ensemble predictions on grid
        probs_g = np.array([self.bnn.predict_proba(p, self.Xg) for p in P])
        mu_g    = probs_g.mean(0)
        ent_g   = -(mu_g*np.log(mu_g+1e-9) + (1-mu_g)*np.log(1-mu_g+1e-9))

        # predictions on training data (for calibration)
        probs_tr = np.array([self.bnn.predict_proba(p, self.X) for p in P])
        mu_tr    = probs_tr.mean(0)
        acc      = ((mu_tr > 0.5) == self.y).mean()

        print(f"  [{entry['name']:20s}]  acc={acc:.2f}"
              f"  diversity={diversity_hist[-1]:.2f}")

        return dict(
            name         = entry["name"],
            color        = entry["color"],
            P            = P,
            snapshots    = snapshots,
            diversity    = diversity_hist,
            probs_g      = probs_g,
            mu_g         = mu_g.reshape(self.grid_res, self.grid_res),
            ent_g        = ent_g.reshape(self.grid_res, self.grid_res),
            mu_tr        = mu_tr,
            acc          = acc,
        )

    def run_all(self):
        print(f"\nDataset: {self.dataset}  |  "
              f"particles={self.n_particles}  iters={self.n_iter}")
        return [self.run_method(e) for e in _REGISTRY]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

BG    = "#0D0F18"
CC0   = "#A78BFA"
CC1   = "#FCD34D"
WHITE = "#EAEAEA"
GRAY  = "#2E3140"
LGRAY = "#4B5060"

cmap_unc  = LinearSegmentedColormap.from_list("unc",
    [BG,"#1a1f3a","#2d1b6e","#7B61FF","#FFD166","#FFFFFF"])
cmap_prob = LinearSegmentedColormap.from_list("prob", [CC0,"#1a1030",CC1])

def _sax(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(BG)
    if title:  ax.set_title(title, color=WHITE, fontsize=9.5,
                             fontweight='bold', pad=5)
    if xlabel: ax.set_xlabel(xlabel, color=WHITE, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=WHITE, fontsize=8)
    ax.tick_params(colors=LGRAY, labelcolor=WHITE, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(GRAY)
    ax.grid(True, color=GRAY, alpha=0.3, linewidth=0.5)


def plot_comparison(exp, results, save_path="comparison.png"):
    """
    Generate the full comparison figure for all registered methods.
    Layout scales automatically with the number of methods.
    """
    M    = len(results)       # number of methods
    G1, G2 = exp.G1, exp.G2
    X, y    = exp.X, exp.y

    # ── figure layout: rows = [boundaries, uncertainty, PCA+diversity, calib]
    fig = plt.figure(figsize=(7*M, 22), facecolor=BG)
    fig.patch.set_facecolor(BG)
    gs_outer = gridspec.GridSpec(4, 1, figure=fig,
                                 hspace=0.38, left=0.06, right=0.97,
                                 top=0.94, bottom=0.04,
                                 height_ratios=[1, 1, 1, 0.85])

    # row 0: decision boundaries  (one col per method)
    gs0 = gridspec.GridSpecFromSubplotSpec(1, M, subplot_spec=gs_outer[0],
                                            wspace=0.08)
    # row 1: uncertainty heatmaps
    gs1 = gridspec.GridSpecFromSubplotSpec(1, M, subplot_spec=gs_outer[1],
                                            wspace=0.08)
    # row 2: PCA trajectories + diversity curve (side by side)
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[2],
                                            wspace=0.28)
    # row 3: calibration
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_outer[3])

    # ── PCA over all final particles ─────────────────────────────────────────
    all_P   = np.vstack([r["P"] for r in results])
    mu_all  = all_P.mean(0)
    C       = ((all_P-mu_all).T @ (all_P-mu_all)) / len(all_P)
    evals, evecs = np.linalg.eigh(C)
    pc1, pc2     = evecs[:,-1], evecs[:,-2]
    var_exp      = evals[-2:] / evals.sum() * 100

    def proj(P):
        c = P - mu_all
        return c @ pc1, c @ pc2

    # ── Row 0: Decision boundaries ───────────────────────────────────────────
    for col, r in enumerate(results):
        ax = fig.add_subplot(gs0[col])
        _sax(ax, f"{r['name']}  —  Boundaries")
        ax.set_facecolor("#080b12")

        for p_vec in r["probs_g"]:
            pm = p_vec.reshape(exp.grid_res, exp.grid_res)
            ax.contour(G1, G2, pm, levels=[0.5],
                       colors=[r["color"]], linewidths=0.7, alpha=0.25)
        ax.contour(G1, G2, r["mu_g"], levels=[0.5],
                   colors=[WHITE], linewidths=2.0, alpha=0.95)
        ax.scatter(X[y==0,0], X[y==0,1], c=CC0, s=20, alpha=0.9,
                   edgecolors='white', linewidths=0.3, zorder=6)
        ax.scatter(X[y==1,0], X[y==1,1], c=CC1, s=20, alpha=0.9,
                   edgecolors='white', linewidths=0.3, zorder=6)
        ax.set_xticks([]); ax.set_yticks([])

        spread = np.std([p.reshape(exp.grid_res, exp.grid_res)
                         for p in r["probs_g"]], axis=0).mean()
        ax.text(0.04, 0.05, f"Spread: {spread:.3f}",
                transform=ax.transAxes, color=r["color"], fontsize=8.5,
                bbox=dict(facecolor=GRAY, alpha=0.6, pad=3, edgecolor='none'))

        els = [Line2D([0],[0], color=r["color"], lw=0.9, alpha=0.5,
                      label=f'Particles ({exp.n_particles})'),
               Line2D([0],[0], color=WHITE, lw=2.0, label='Mean boundary')]
        ax.legend(handles=els, framealpha=0.2, facecolor=GRAY,
                  edgecolor='none', labelcolor=WHITE, fontsize=7.5,
                  loc='upper right')

    # ── Row 1: Uncertainty heatmaps ──────────────────────────────────────────
    vmax_ent = np.log(2)
    for col, r in enumerate(results):
        ax = fig.add_subplot(gs1[col])
        _sax(ax, f"{r['name']}  —  Uncertainty H[p(y|x)]")
        ax.contourf(G1, G2, r["ent_g"], levels=50,
                    cmap=cmap_unc, vmin=0, vmax=vmax_ent, alpha=0.95)
        ax.contour(G1, G2, r["mu_g"], levels=[0.5],
                   colors=[WHITE], linewidths=1.4, alpha=0.6, linestyles='--')
        ax.scatter(X[y==0,0], X[y==0,1], c=CC0, s=16, alpha=0.8,
                   edgecolors='none', zorder=5)
        ax.scatter(X[y==1,0], X[y==1,1], c=CC1, s=16, alpha=0.8,
                   edgecolors='none', zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.04, 0.05, f"Mean H: {r['ent_g'].mean():.3f}",
                transform=ax.transAxes, color=r["color"], fontsize=8.5,
                bbox=dict(facecolor=GRAY, alpha=0.6, pad=3, edgecolor='none'))

    # ── Row 2a: PCA particle trajectories ────────────────────────────────────
    axC = fig.add_subplot(gs2[0])
    _sax(axC, "Particle Trajectories  (PCA of weight space)",
         f"PC₁  ({var_exp[1]:.1f}% var)", f"PC₂  ({var_exp[0]:.1f}% var)")

    snap_keys = sorted(results[0]["snapshots"].keys())
    alphas_t  = np.linspace(0.15, 1.0, len(snap_keys))

    for r in results:
        for i, t in enumerate(snap_keys):
            if t in r["snapshots"]:
                px, py = proj(r["snapshots"][t])
                axC.scatter(px, py, c=r["color"],
                            s=12 + 8*i, alpha=alphas_t[i],
                            edgecolors='none', zorder=3)
        # final ellipse
        px, py = proj(r["P"])
        from matplotlib.patches import Ellipse
        el = Ellipse((px.mean(), py.mean()),
                     width=2*2.5*px.std(), height=2*2.5*py.std(),
                     fill=False, color=r["color"], lw=1.5,
                     ls='--', alpha=0.75)
        axC.add_patch(el)

    legend_els = [Line2D([0],[0], marker='o', color=r["color"], ls='none',
                         ms=7, label=r["name"]) for r in results]
    legend_els += [Line2D([0],[0], color=WHITE, lw=0, alpha=0,
                          label='(dark→bright = init→final)')]
    axC.legend(handles=legend_els, framealpha=0.2, facecolor=GRAY,
               edgecolor='none', labelcolor=WHITE, fontsize=8)

    # ── Row 2b: Diversity over training ──────────────────────────────────────
    axD = fig.add_subplot(gs2[1])
    _sax(axD, "Particle Diversity Over Training",
         "Iteration", "Avg pairwise  ‖wᵢ − wⱼ‖₂")

    iters = np.arange(exp.n_iter)
    for r in results:
        axD.plot(iters, r["diversity"], color=r["color"],
                 lw=2.2, label=r["name"], zorder=4)

    # shade gap between first two methods
    if len(results) >= 2:
        d0 = np.array(results[0]["diversity"])
        d1 = np.array(results[1]["diversity"])
        axD.fill_between(iters, d0, d1,
                         where=d1 > d0, alpha=0.12,
                         color=results[1]["color"], label="Diversity gain")
        mid = exp.n_iter // 2
        gap = d1[mid] - d0[mid]
        if gap > 0:
            axD.annotate('', xy=(mid, d1[mid]), xytext=(mid, d0[mid]),
                         arrowprops=dict(arrowstyle='<->', color=WHITE, lw=1.2))
            axD.text(mid + exp.n_iter*0.02,
                     (d0[mid]+d1[mid])/2,
                     f'+{gap:.1f}', color=WHITE, fontsize=8.5)

    axD.legend(framealpha=0.2, facecolor=GRAY, edgecolor='none',
               labelcolor=WHITE, fontsize=8.5)

    # ── Row 3: Calibration ───────────────────────────────────────────────────
    axE = fig.add_subplot(gs3[0])
    _sax(axE, "Calibration  —  Reliability Diagram",
         "Mean predicted probability", "Fraction of positives")
    axE.plot([0,1],[0,1], color=WHITE, lw=1.2, ls=':',
             alpha=0.4, label='Perfect calibration')

    def ece(probs, y, n_bins=10):
        bins = np.linspace(0, 1, n_bins+1)
        e = 0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (probs>=lo) & (probs<hi)
            if m.sum() > 0:
                e += m.sum() * abs(y[m].mean() - probs[m].mean())
        return e / len(y)

    ece_strs = []
    for r in results:
        fp, mp = calibration_curve(y, r["mu_tr"], n_bins=10)
        axE.plot(mp, fp, 'o-', color=r["color"], lw=2.2, ms=7,
                 label=r["name"], zorder=5)
        axE.fill_between(mp, fp, mp, color=r["color"], alpha=0.09)
        e = ece(r["mu_tr"], y)
        ece_strs.append(f"{r['name']}: {e:.3f}")

    axE.text(0.98, 0.05, "ECE ↓\n" + "\n".join(ece_strs),
             transform=axE.transAxes, color=WHITE, fontsize=9,
             ha='right', va='bottom',
             bbox=dict(facecolor=GRAY, alpha=0.7, pad=5, edgecolor='none'))
    axE.set_xlim(0,1); axE.set_ylim(0,1)
    axE.legend(framealpha=0.2, facecolor=GRAY, edgecolor='none',
               labelcolor=WHITE, fontsize=9)

    # ── Supertitle ─────────────────────────────────────────────────────────--
    names = "  vs  ".join(r["name"] for r in results)
    fig.suptitle(
        f"{names}   ·   BNN   ·   {exp.dataset.replace('_',' ').title()}"
        f"   ·   N={exp.n_particles} particles",
        color=WHITE, fontsize=13, fontweight='bold', y=0.975)

    fig.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=BG)
    # also save pdf
    fig.savefig(save_path.replace(".png", ".pdf"), dpi=160,
                bbox_inches='tight', facecolor=BG)
    print(f"\nSaved → {save_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    exp     = Experiment(
        dataset     = "two_moons",   # "two_moons" | "two_circles" | "checkerboard" | "spirals"
        n_particles = 25,
        n_iter      = 200,
        hidden      = 32,
    )
    results = exp.run_all()
    plot_comparison(exp, results,
                    save_path="comparison.png")
