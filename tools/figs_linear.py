#!/usr/bin/env python3
"""Figures for 3-linear-models (prefix lin-). Run: python3 tools/figs_linear.py"""
import numpy as np

from mplfig import ACCENT, ACCENT2, HI, INK, MUTED, plt, save


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------- linear regression (L7)
rng = np.random.default_rng(364)
N1 = 25
x1 = np.linspace(-2, 6, N1)
y1 = 4 * x1 + 2 + 10 * rng.standard_normal(N1)
X1 = np.stack((x1, np.ones(N1)), axis=1)
m_star, b_star = np.linalg.solve(X1.T @ X1, X1.T @ y1)


def mse(m, b):
    return np.mean((m * x1 + b - y1) ** 2)


# residuals of a guess vs. the least-squares line
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
xx = np.linspace(-2.6, 6.6, 50)
for ax, (m, b, title) in zip(axs, [(2, 2, "guess $m=2,\\ b=2$"), (m_star, b_star, "least squares")]):
    yhat = m * x1 + b
    ax.vlines(x1, np.minimum(y1, yhat), np.maximum(y1, yhat), color=HI, lw=1.2, linestyles="dashed")
    ax.plot(xx, m * xx + b, color=ACCENT)
    ax.scatter(x1, y1, s=16, color=INK, zorder=3)
    ax.set_title(f"{title}: MSE = {mse(m, b):.1f}", fontsize=11)
    ax.set_xlabel("$x$")
    ax.grid(True)
axs[0].set_ylabel("$y$")
save(fig, "lin-residuals",
     "The MSE is the mean of the squared lengths of the dashed <b>residuals</b> <i>mx<sub>i</sub></i> + <i>b</i> − <i>y<sub>i</sub></i>. "
     "Left: a guessed line. Right: the least-squares line, the one with the smallest possible MSE.",
     alt="Scatter plot with two candidate lines and their vertical residuals")

# MSE bowl with the gradient-descent path, and the iterates vs. iteration
alpha, n_iter = 3e-2, 200
m, b = -2.0, -4.0
path = [(m, b)]
for _ in range(n_iter):
    r = m * x1 + b - y1
    m, b = m - alpha * np.mean(2 * x1 * r), b - alpha * np.mean(2 * r)
    path.append((m, b))
path = np.array(path)
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
M, B = np.meshgrid(np.linspace(-3, 8, 220), np.linspace(-6, 14, 220))
J = np.mean((M[..., None] * x1 + B[..., None] - y1) ** 2, axis=-1)
axs[0].contour(M, B, np.log(J), levels=14, colors=MUTED, linewidths=0.8)
axs[0].plot(path[:, 0], path[:, 1], color=ACCENT, lw=1.6, marker="o", ms=2.5, markevery=[0, 1, 2, 3, 5, 10, 20, 50, 100, 200])
axs[0].plot([m_star], [b_star], marker="*", ms=13, color=HI, ls="none")
axs[0].annotate("start $(-2,-4)$", path[0], xytext=(-0.5, -5.3), fontsize=9)
axs[0].annotate("$(m^*, b^*)$", (m_star, b_star), xytext=(m_star + 0.5, b_star + 1.5), fontsize=9)
axs[0].set_xlabel("slope $m$")
axs[0].set_ylabel("intercept $b$")
axs[0].set_title("contours of $\\log J(m,b)$", fontsize=11)
it = np.arange(n_iter + 1)
axs[1].plot(it, path[:, 0], color=ACCENT, label="$m^{(k)}$")
axs[1].plot(it, path[:, 1], color=ACCENT2, label="$b^{(k)}$")
axs[1].axhline(m_star, color=HI, ls="--", lw=1)
axs[1].axhline(b_star, color=HI, ls="--", lw=1)
axs[1].set_xlabel("iteration $k$")
axs[1].set_title("iterates vs. the closed form (dashed)", fontsize=11)
axs[1].legend(loc="lower right")
axs[1].grid(True)
save(fig, "lin-mse-bowl",
     "Left: the MSE of a line is a convex bowl over (<i>m</i>, <i>b</i>) with one minimum (star). Gradient descent "
     "(α = 0.03, as in Lecture 7) slides down the long, narrow valley. Right: <i>m</i> settles within a few "
     "dozen steps, <i>b</i> needs about 200; both approach the exact solution from the normal equations.",
     alt="Contour plot of the MSE with a gradient-descent path, and m, b versus iteration")

# ---------------------------------------------------------------- multiple regression: diabetes MSEs (L7 notebook output)
feats = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
mses = [5720.5470, 5918.8889, 3890.4566, 4774.1139, 5663.3156, 5750.2411, 5005.6616, 4831.1384, 4030.9987, 5062.3806]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.bar(feats, mses, color=ACCENT, width=0.65)
ax.axhline(5929.8849, color=MUTED, ls="--", lw=1.4, label="predict the mean: 5930")
ax.axhline(2859.6963, color=HI, ls="-", lw=1.6, label="all 10 features: 2860")
ax.set_ylabel("training MSE")
ax.set_ylim(0, 7600)
ax.legend(fontsize=9, loc="upper center", ncol=2)
ax.set_title("one feature at a time vs. all ten together", fontsize=11)
save(fig, "lin-diabetes-mse",
     "Lecture 7's diabetes data (442 patients, 10 features): the training MSE of a line fit on each single feature "
     "(bars), against the constant mean prediction and multiple linear regression on all ten features. Numbers are "
     "the notebook's printed output. BMI is the best single feature; sex alone is almost useless.",
     alt="Bar chart of single-feature MSEs with two horizontal reference lines")

# ---------------------------------------------------------------- basis functions (L8)
rng = np.random.default_rng(8)
N = 100
coef = np.array([0.7, -0.7, -0.8, -0.25, -0.3, 0.75])
x = rng.uniform(-2, 2, N)
Phi = np.vander(x, 6)
y = Phi @ coef + 3 * rng.standard_normal(N)
w = np.linalg.solve(Phi.T @ Phi, Phi.T @ y)
xs = np.linspace(-2, 2, 300)
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
axs[0].scatter(x, y, s=10, color=MUTED)
axs[0].plot(xs, np.vander(xs, 6) @ coef, color=ACCENT2, label="true quintic")
axs[0].plot(xs, np.vander(xs, 6) @ w, color=ACCENT, ls="--", label="fit, $\\Phi$ has 6 columns")
axs[0].set_title("$\\phi(x)=[x^5,x^4,x^3,x^2,x,1]$", fontsize=11)
axs[0].legend(fontsize=9)
axs[0].grid(True)
# sinusoidal features, as in HW3 Q1
w0 = 2.0
A = np.array([1.5, -0.8, 0.5])
x = rng.uniform(0, 2 * np.pi, 40)
S = lambda t: np.stack([np.sin(w0 * t), np.sin(2 * w0 * t), np.sin(3 * w0 * t)], axis=1)
y = S(x) @ A + 0.3 * rng.standard_normal(len(x))
Ahat = np.linalg.solve(S(x).T @ S(x), S(x).T @ y)
xs = np.linspace(0, 2 * np.pi, 400)
axs[1].scatter(x, y, s=12, color=MUTED)
axs[1].plot(xs, S(xs) @ A, color=ACCENT2, label="true")
axs[1].plot(xs, S(xs) @ Ahat, color=ACCENT, ls="--", label="fit")
axs[1].set_title("$\\phi(x)=[\\sin\\omega_0x,\\ \\sin2\\omega_0x,\\ \\sin3\\omega_0x]$", fontsize=11)
axs[1].legend(fontsize=9, loc="lower left")
axs[1].grid(True)
save(fig, "lin-basis-fits",
     "Both fits use the same formula <i>w</i>* = (Φ<sup>⊤</sup>Φ)<sup>−1</sup>Φ<sup>⊤</sup><i>y</i>; only the columns of Φ change. "
     "Left: a fifth-order polynomial (Lecture 8). Right: three sinusoids with known frequency ω<sub>0</sub>, so the "
     "unknowns are the amplitudes (HW3 Q1). The curves are nonlinear in <i>x</i> but the models are linear in <i>w</i>.",
     alt="Polynomial and sinusoidal least-squares fits to noisy data")

# ---------------------------------------------------------------- ridge (L8)
rng = np.random.default_rng(81)
N = 20
mt = np.array([0.4, 0.9, 0.0])
x = rng.uniform(-2, 2, N)
Phi = np.vander(x, 3)
y = Phi @ mt + 3 * rng.standard_normal(N)
out = rng.choice(N, 4, replace=False)
y[out] += 30
xs = np.linspace(x.min(), x.max(), 300)
fig, ax = plt.subplots(figsize=(6.2, 3.3))
ax.scatter(np.delete(x, out), np.delete(y, out), s=16, color=INK, zorder=3)
ax.scatter(x[out], y[out], s=34, facecolors="none", edgecolors=HI, lw=1.6, zorder=3, label="outliers (+30)")
ax.plot(xs, np.vander(xs, 3) @ mt, color=ACCENT2, label="true quadratic")
for lam, ls in [(0, "--"), (100, "-")]:
    wl = np.linalg.solve(Phi.T @ Phi + lam * np.eye(3), Phi.T @ y)
    ax.plot(xs, np.vander(xs, 3) @ wl, color=ACCENT if lam else MUTED, ls=ls, label=f"ridge, $\\lambda={lam}$")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True)
save(fig, "lin-ridge-outliers",
     "Lecture 8's outlier example: 20 points from a quadratic, four of them shifted up by 30. Plain least squares "
     "(λ = 0) bends to chase the outliers; with λ = 100 the large coefficients are penalized and the fit "
     "stays near the bulk of the data.",
     alt="Quadratic data with four outliers and two fitted curves")

rng = np.random.default_rng(89)
x = rng.uniform(-2, 2, 20)
y = 3 * x**2 + 2 * x + 1 + 3 * rng.standard_normal(20)
D = 10
Phi = np.vander(x, D + 1)
xs = np.linspace(x.min(), x.max(), 400)
fig, axs = plt.subplots(1, 3, figsize=(9.2, 2.9), sharey=True)
for ax, (lam, verdict) in zip(axs, [(0, "overfit: increase $\\lambda$"), (10, "about right"), (1e6, "underfit: decrease $\\lambda$")]):
    wl = np.linalg.solve(Phi.T @ Phi + lam * np.eye(D + 1), Phi.T @ y)
    ax.scatter(x, y, s=12, color=INK, zorder=3)
    ax.plot(xs, 3 * xs**2 + 2 * xs + 1, color=ACCENT2, lw=1.4)
    ax.plot(xs, np.vander(xs, D + 1) @ wl, color=ACCENT)
    ax.set_title("$\\lambda=" + {0: "0", 10: "10", 1e6: "10^6"}[lam] + f"$ ({verdict})", fontsize=10)
    ax.set_ylim(-15, 30)
    ax.grid(True)
save(fig, "lin-ridge-lambda",
     "A degree-10 polynomial (11 weights) fit to 20 noisy points from 3<i>x</i><sup>2</sup> + 2<i>x</i> + 1 (green). With λ = 0 the "
     "curve swings between and beyond the points; a moderate λ follows the trend; a huge λ "
     "shrinks every weight toward 0 and the curve goes flat. This is the “read λ off the plot” exam question.",
     alt="Three panels of degree-10 polynomial fits with increasing lambda")

# ---------------------------------------------------------------- cost (L8)
d = np.logspace(0, 4, 100)
Nn = 1e5
fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.loglog(d, Nn * d**2 + Nn * d + d**3 + d**2, color=HI, label="closed form $Nd^2+d^3$")
for k, c in [(10, ACCENT2), (100, ACCENT), (1000, MUTED)]:
    ax.loglog(d, k * (2 * Nn * d + Nn), color=c, ls="--", label=f"GD, $k={k}$: $k(2Nd+N)$")
ax.set_xlabel("number of features $d$")
ax.set_ylabel("multiply-adds")
ax.set_title("$N=10^5$ data points", fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, which="major")
save(fig, "lin-cost",
     "Operation counts from Lecture 8 for <i>N</i> = 10<sup>5</sup>. Gradient descent grows linearly in <i>d</i>; the closed form grows "
     "like <i>d</i><sup>2</sup> (then <i>d</i><sup>3</sup>), so it wins only when <i>d</i> is small compared with the number of iterations <i>k</i>.",
     alt="Log-log plot of operation counts versus d")

# ---------------------------------------------------------------- classification (L8, L9)
def fit_logistic(X, y, alpha, n_iter):
    w = np.zeros(X.shape[1])
    for _ in range(n_iter):
        mrg = y * (X @ w)
        w -= alpha * np.sum((-y * sigmoid(-mrg))[:, None] * X, axis=0)
    return w


rng = np.random.default_rng(9)
n = 50
xa = np.concatenate((rng.random(n) - 1, rng.random(n)))
ya = np.concatenate((-np.ones(n), np.ones(n)))
xb = np.concatenate((xa, rng.random(n) + 20))
yb = np.concatenate((ya, np.ones(n)))
fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.2), gridspec_kw={"width_ratios": [1, 1.5]})
for ax, (xd, yd, title) in zip(axs, [(xa, ya, "balanced: works"), (xb, yb, "50 extra positives at $x\\approx 20$")]):
    Xd = np.stack((xd, np.ones_like(xd)), axis=1)
    wl = np.linalg.solve(Xd.T @ Xd, Xd.T @ yd)
    wg = fit_logistic(Xd, yd, 1e-2, 3000)
    xs = np.linspace(xd.min() - 0.3, xd.max() + 0.3, 600)
    ax.scatter(xd[yd < 0], yd[yd < 0], marker="^", s=18, color=ACCENT, label="class $-1$")
    ax.scatter(xd[yd > 0], yd[yd > 0], marker="o", s=18, color=HI, label="class $+1$")
    ax.plot(xs, wl[0] * xs + wl[1], color=MUTED, ls="--", label="least-squares line")
    ax.plot(xs, 2 * sigmoid(wg[0] * xs + wg[1]) - 1, color=ACCENT2, label="logistic, $2\\sigma(z)-1$")
    ax.axvline(-wl[1] / wl[0], color=MUTED, lw=1, ls=":")
    ax.set_ylim(-1.6, 1.6)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("$x$")
    ax.grid(True)
axs[0].set_ylabel("label")
axs[1].legend(fontsize=8.5, loc="lower right")
save(fig, "lin-linreg-vs-logistic",
     "Classifying with sgn(<i>mx</i> + <i>b</i>) from least squares (Lecture 8). Left: balanced data, the line crosses 0 "
     "(dotted) between the classes. Right: the far-away positives drag the line, its zero crossing moves into the "
     "positive class and those points are misclassified, although the data is still separable. Logistic regression "
     "(green, rescaled to [−1, 1]) keeps its boundary at the gap (Lecture 9).",
     alt="Two panels comparing least-squares and logistic fits on 1-D labels")

z = np.linspace(-6, 6, 300)
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.0))
axs[0].plot(z, sigmoid(z), color=ACCENT)
axs[0].axhline(0.5, color=MUTED, lw=1, ls="--")
axs[0].axvline(0, color=MUTED, lw=1, ls="--")
axs[0].set_xlabel("score $z=w^\\top x$")
axs[0].set_ylabel("$\\sigma(z)=\\Pr\\{y=+1\\mid x\\}$")
axs[0].set_title("$\\sigma(z) \\geq 0.5 \\Leftrightarrow z \\geq 0$", fontsize=11)
axs[0].grid(True)
xx = np.linspace(-5, 5, 300)
for (mm, bb), c in zip([(1, 0), (4, 0), (10, -20), (-2, 1)], [ACCENT, HI, ACCENT2, MUTED]):
    axs[1].plot(xx, sigmoid(mm * xx + bb), color=c, label=f"$m={mm},\\ b={bb}$")
axs[1].set_xlabel("$x$")
axs[1].set_title("$\\sigma(mx+b)$: boundary at $x=-b/m$", fontsize=11)
axs[1].legend(fontsize=8.5, loc="center left")
axs[1].grid(True)
save(fig, "lin-sigmoid",
     "Left: the sigmoid maps any score to a probability; it is 0.5 exactly at <i>z</i> = 0. Right: in 1-D, |<i>m</i>| sets how "
     "sharp the transition is, −<i>b</i>/<i>m</i> sets where it happens, and <i>m</i> &lt; 0 flips which side is positive.",
     alt="Sigmoid curve and four sigmoids with different slopes and offsets")

# Fall 2025 Q6(a) data with the candidate boundaries
P = np.array([[0, -1], [1, -2], [0, -3]])
Q = np.array([[1, 1], [0, 2], [1, 3]])
fig, ax = plt.subplots(figsize=(5.0, 4.0))
ax.scatter(P[:, 0], P[:, 1], marker="o", s=40, color=HI, label="$+1$", zorder=3)
ax.scatter(Q[:, 0], Q[:, 1], marker="^", s=40, color=ACCENT, label="$-1$", zorder=3)
t = np.linspace(-2.5, 3.5, 10)
ax.plot(t, -t, color=MUTED, ls="--", label="$w_2=(-1,-1)$: $x_2=-x_1$")
ax.axhline(0, color=ACCENT2, label="$w_3=(0,-2)$: $x_2=0$")
ax.plot(t, t / 2, color=INK, ls=":", label="$w_5=(1,-2)$: $x_2=x_1/2$")
for (w, c) in [((0, -2), ACCENT2)]:
    ax.annotate("", xy=(2.6, -1.0), xytext=(2.6, 0), arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
ax.text(2.7, -0.6, "$w_3$", color=ACCENT2)
ax.set_xlim(-2.5, 3.5)
ax.set_ylim(-3.6, 3.6)
ax.set_aspect("equal")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.grid(True)
ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))
save(fig, "lin-fall25-boundaries",
     "Fall 2025 Q6(a): the three weight vectors that classify all six points. Each boundary is the line <i>w</i><sup>⊤</sup><i>x</i> = 0 "
     "through the origin (no bias), and <i>w</i> points to the positive side. <i>w</i><sub>3</sub> gives every point a margin "
     "<i>y<sub>i</sub></i><i>w</i><sup>⊤</sup><i>x<sub>i</sub></i> ≥ 2, the largest smallest margin, so it has the largest likelihood.",
     alt="Six labeled points and three linear decision boundaries")

# 2-D logistic regression probability contours
rng = np.random.default_rng(92)
Xc = np.concatenate((rng.standard_normal((50, 2)) + [-2, -2], rng.standard_normal((50, 2)) + [2, 2]))
yc = np.concatenate((-np.ones(50), np.ones(50)))
Xa = np.concatenate((Xc, np.ones((100, 1))), axis=1)
wc = fit_logistic(Xa, yc, 1e-2, 500)
g1, g2 = np.meshgrid(np.linspace(-5.5, 5.5, 200), np.linspace(-5.5, 5.5, 200))
pr = sigmoid(wc[0] * g1 + wc[1] * g2 + wc[2])
fig, ax = plt.subplots(figsize=(4.6, 4.0))
cs = ax.contour(g1, g2, pr, levels=[0.1, 0.5, 0.9], colors=[MUTED, INK, MUTED], linewidths=[1, 1.8, 1], linestyles=["--", "-", "--"])
ax.clabel(cs, fmt={0.1: "0.1", 0.5: "0.5", 0.9: "0.9"}, fontsize=8)
ax.scatter(Xc[yc < 0, 0], Xc[yc < 0, 1], marker="^", s=16, color=ACCENT)
ax.scatter(Xc[yc > 0, 0], Xc[yc > 0, 1], marker="o", s=16, color=HI)
ax.set_aspect("equal")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.grid(True)
save(fig, "lin-logistic-2d",
     "Logistic regression on two Gaussian clusters (Lecture 9 setup, 500 gradient-descent steps). The level sets of "
     "Pr{<i>y</i> = +1 | <i>x</i>} = σ(<i>m</i><sub>1</sub><i>x</i><sub>1</sub> + <i>m</i><sub>2</sub><i>x</i><sub>2</sub> + <i>b</i>) are parallel lines; the 0.5 line is the decision boundary.",
     alt="Two clusters with probability contour lines at 0.1, 0.5, 0.9")

# ---------------------------------------------------------------- BCE
f = np.linspace(0.005, 0.995, 300)
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.plot(f, -np.log(f), color=HI, label="$y=1$: $-\\log f$")
ax.plot(f, -np.log(1 - f), color=ACCENT, label="$y=0$: $-\\log(1-f)$")
ax.set_xlabel("predicted $f=\\Pr\\{y=1\\mid x\\}$")
ax.set_ylabel("loss")
ax.set_ylim(0, 5)
ax.legend()
ax.grid(True)
save(fig, "lin-bce",
     "Binary cross-entropy: only one of its two terms is active for each example. A confident wrong prediction "
     "costs a lot (−log <i>f</i> → ∞ as <i>f</i> → 0 when <i>y</i> = 1); a confident right one costs almost nothing.",
     alt="Two loss curves of binary cross-entropy")

# ---------------------------------------------------------------- softmax
zl = np.array([2.0, 0.0, -1.0])
p = np.exp(zl) / np.exp(zl).sum()
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
axs[0].bar(["1", "2", "3"], zl, color=ACCENT, width=0.55)
axs[0].axhline(0, color=INK, lw=0.8)
axs[0].set_title("logits $z$", fontsize=11)
for i, v in enumerate(zl):
    axs[0].text(i, v + (0.12 if v >= 0 else -0.35), f"{v:g}", ha="center", fontsize=10)
axs[0].set_ylim(-1.6, 2.6)
axs[1].bar(["1", "2", "3"], p, color=HI, width=0.55)
for i, v in enumerate(p):
    axs[1].text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=10)
axs[1].set_ylim(0, 1.05)
axs[1].set_title("softmax$(z)$", fontsize=11)
for ax in axs:
    ax.set_xlabel("class $k$")
save(fig, "lin-softmax",
     "Softmax exponentiates and normalizes: logits (2, 0, −1) become probabilities (0.844, 0.114, 0.042). "
     "The order is unchanged, so the predicted class is the same before and after softmax.",
     alt="Bar charts of three logits and their softmax probabilities")

# ---------------------------------------------------------------- multi-class regions (L9 toy)
rng = np.random.default_rng(93)
means = np.array([[3, 3], [-3, 3], [3, -3], [-3, -3]])
Xm = np.concatenate([rng.standard_normal((50, 2)) + mu for mu in means])
ym = np.repeat(np.arange(4), 50)
W = np.zeros((4, 2))
bm = np.zeros(4)
Y = np.eye(4)[ym]
for _ in range(500):
    Z = Xm @ W.T + bm
    Pm = np.exp(Z - Z.max(1, keepdims=True))
    Pm /= Pm.sum(1, keepdims=True)
    G = (Pm - Y) / len(ym)
    W -= 0.1 * G.T @ Xm
    bm -= 0.1 * G.sum(0)
g1, g2 = np.meshgrid(np.linspace(-6.5, 6.5, 260), np.linspace(-6.5, 6.5, 260))
cls = np.argmax(np.stack([W[k, 0] * g1 + W[k, 1] * g2 + bm[k] for k in range(4)]), axis=0)
cols = [ACCENT, HI, ACCENT2, MUTED]
mk = ["o", "^", "s", "D"]
fig, ax = plt.subplots(figsize=(4.6, 4.3))
ax.contourf(g1, g2, cls, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=cols, alpha=0.13)
ax.contour(g1, g2, cls, levels=[0.5, 1.5, 2.5], colors=INK, linewidths=0.8)
for k in range(4):
    ax.scatter(Xm[ym == k, 0], Xm[ym == k, 1], s=12, color=cols[k], marker=mk[k], label=f"class {k}")
    ax.annotate("", xy=W[k] * 1.6, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=cols[k], lw=1.8))
ax.set_aspect("equal")
ax.set_xlim(-6.5, 6.5)
ax.set_ylim(-6.5, 6.5)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend(fontsize=8, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.13), handletextpad=0.2, columnspacing=0.8)
save(fig, "lin-multiclass-regions",
     "Four-class logistic regression (the Lecture 9 toy problem), trained with cross-entropy. Each region is where "
     "one score <i>z<sub>k</sub></i> = <i>w<sub>k</sub></i><sup>⊤</sup><i>x</i> + <i>b<sub>k</sub></i> is the largest; the boundaries are where two scores tie, so they are straight "
     "lines. Arrows: the rows <i>w<sub>k</sub></i> of <i>W</i> (scaled), each pointing toward its own cluster.",
     alt="Four clusters, linear decision regions and the class weight vectors")

print("ok")
