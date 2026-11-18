"""
Stage 2: parametric PINN for thermocapillary migration of a Brinkman porous sphere
(Beavers-Joseph slip, stress jump, low Peclet number).  JAX implementation.

Same formulation as pinn_stage1_jax.py (Sections 3.1-3.7), extended so that one network
covers a range of the physical parameters theta = (lambda, eta, xi, beta):
  * every collocation point carries its own theta, sampled in the valid domain
    beta / (lambda sqrt(eta)) <= 1  (Eq. validity);
  * the interior network receives (r, zeta, exp(-lambda(1-r)), theta_n), the exterior network
    (s, zeta, theta_n), with theta_n the parameters scaled to [-1, 1];
  * the migration velocity becomes a small network U(theta) (free particle, C = 0), or the
    Stokeslet coefficient a network C(theta) (held particle, U = 0); their last layer starts
    at zero, so U = 0 (or C = 0) initially, as in Stage 1;
  * optional --param_weight lam4 multiplies ALL loss terms at a point by (1+lambda^2)^-2:
    it balances points of different lambda, while keeping the Stage 1 balance between the
    terms at any one lambda (unlike the interior-only rescaling of Section 5.1).
k only scales U_T and F_T by 1/(2+k) (Section 2.4) and is kept fixed.
The exact solution is used ONLY in selftest and eval.

Usage
  python pinn_stage2_param.py --mode selftest
  python pinn_stage2_param.py --mode lbfgs --iters 10000 --tag P1 --lam 0.5 16
  python pinn_stage2_param.py --mode eval  --tag P1 --lam 0.5 16
A parameter is varied when its two range values differ, e.g. --eta 0.25 4 --xi -1 1 --beta 0 2.
"""
import argparse, json, os, time
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

# =============================================================================
# 0. Configuration
# =============================================================================
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["selftest", "lbfgs", "eval", "atlas"], default="selftest")
ap.add_argument("--branch", choices=["free", "held"], default="free")
ap.add_argument("--lam", type=float, nargs=2, default=[0.5, 16.0])
ap.add_argument("--eta", type=float, nargs=2, default=[1.0, 1.0])
ap.add_argument("--xi", type=float, nargs=2, default=[0.5, 0.5])
ap.add_argument("--beta", type=float, nargs=2, default=[0.5, 0.5])
ap.add_argument("--k", type=float, default=1.0)
ap.add_argument("--lam_sampling", choices=["log", "uniform"], default="log")
ap.add_argument("--param_weight", choices=["none", "lam2", "lam4"], default="none")
ap.add_argument("--no_bl_feature", action="store_true")
ap.add_argument("--iters", type=int, default=10000)
ap.add_argument("--tag", default="param")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--w_if", type=float, default=10.0)
ap.add_argument("--n_int", type=int, default=2000)
ap.add_argument("--n_ext", type=int, default=1500)
ap.add_argument("--n_if", type=int, default=500)
ap.add_argument("--int_layers", default="48,48,48")
ap.add_argument("--ext_layers", default="32,32")
ap.add_argument("--u_layers", default="16,16")
ap.add_argument("--n_eval", type=int, default=33)
ap.add_argument("--ref", type=float, nargs=3, default=[1.0, 0.5, 0.5])   # (eta, xi, beta) of the lambda sweep and atlas slices
ap.add_argument("--n_atlas", type=int, default=9)                        # grid points per axis of each atlas slice
args = ap.parse_args()

RANGES = np.array([args.lam, args.eta, args.xi, args.beta], dtype=float)   # rows: lam, eta, xi, beta
VARIES = RANGES[:, 0] < RANGES[:, 1]
LOGSCALE = np.array([args.lam_sampling == "log" and RANGES[0, 0] > 0, RANGES[1, 0] > 0, False, False])
MARANG = 3.0 / (2.0 + args.k)
BL = not args.no_bl_feature

# =============================================================================
# 1. Parameters: sampling, scaling, exact reference
# =============================================================================
def sample_theta(n, rng):
    """n parameter vectors (lam, eta, xi, beta) in the ranges and the valid domain."""
    out = np.zeros((0, 4))
    while len(out) < n:
        m = 4 * n
        cols = []
        for j in range(4):
            a, b = RANGES[j]
            if a == b:
                cols.append(np.full(m, a))
            elif LOGSCALE[j]:
                cols.append(np.exp(rng.uniform(np.log(a), np.log(b), m)))
            else:
                cols.append(rng.uniform(a, b, m))
        th = np.stack(cols, 1)
        lam, eta, beta = th[:, 0], th[:, 1], th[:, 3]
        ok = (beta == 0) | ((lam > 0) & (eta > 0) & (beta <= lam * np.sqrt(np.maximum(eta, 0)) + 1e-12))
        out = np.concatenate([out, th[ok]])
    return out[:n]

def theta_n(th):
    """Parameters scaled to [-1, 1] (0 for a fixed parameter); log scale for lam and eta when positive."""
    cols = []
    for j in range(4):
        a, b = RANGES[j]
        if a == b:
            cols.append(0.0 * th[j])
        elif LOGSCALE[j]:
            cols.append(2 * (jnp.log(th[j]) - np.log(a)) / (np.log(b) - np.log(a)) - 1)
        else:
            cols.append(2 * (th[j] - a) / (b - a) - 1)
    return jnp.stack(cols)

def coefficients(th):
    """Interface coefficients for theta; safe at beta = 0 (incl. lam = 0 or eta = 0)."""
    lam, eta, xi, beta = th[0], th[1], th[2], th[3]
    sq = jnp.sqrt(eta)
    c_slip = beta / (lam * sq + (beta == 0))       # = 0 when beta = 0
    c_jump = lam * sq * xi
    return c_slip, c_jump

def exact(th):
    """Exact coefficients (A1, A2, B1, B2, U) for one parameter vector (numpy)."""
    L, eta, xi, beta = [float(v) for v in th]
    if L == 0:
        g0, g1, g2 = 1.0, 4.0, 12.0
    else:
        sh, ch = np.sinh(L), np.cosh(L)
        g0, g1, g2 = sh / L - ch, ch - sh / L - L * sh, -2 * ch + 2 * sh / L + L * sh - L**2 * ch
    cs = 0.0 if beta == 0 else beta / (L * np.sqrt(eta))
    cj = L * np.sqrt(eta) * xi
    SpA2 = g2 - 2 * g1 + 2 * g0
    if args.branch == "free":
        M = np.array([[1, g0, 0, 0, .5], [0, 0, 1, 1, .5], [-2, -g1, 1, -1 - 6 * cs, 0],
                      [0, -eta * SpA2, -cj, 6 + cj, 0], [0, 0, 1, 0, 0]])
        return np.linalg.solve(M, [0, 0, 0, -MARANG, 0])
    M = np.array([[1, g0, 0, 0], [0, 0, 1, 1], [-2, -g1, 1, -1 - 6 * cs], [0, -eta * SpA2, -cj, 6 + cj]])
    A1, A2, B1, B2 = np.linalg.solve(M, [0, 0, 0, -MARANG])
    return np.array([A1, A2, B1, B2, 0.0])

def g_exact(L, r):
    return r**4 if L == 0 else np.sinh(L * r) / (L * r) - np.cosh(L * r)

# =============================================================================
# 2. Networks
# =============================================================================
def init_mlp(key, sizes, zero_last=False):
    params = []
    for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, sub = jax.random.split(key)
        W = jax.random.normal(sub, (m, n)) * np.sqrt(2.0 / (m + n))
        if zero_last and i == len(sizes) - 2:
            W = jnp.zeros((m, n))
        params.append((W, jnp.zeros(n)))
    return params

def mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return (x @ W + b)[0]

def U_of(P, th):
    return mlp(P["u"], theta_n(th)) if args.branch == "free" else 0.0

def C_of(P, th):
    return mlp(P["u"], theta_n(th)) if args.branch == "held" else 0.0

# =============================================================================
# 3. Representation (Secs. 3.1-3.3, with theta)
# =============================================================================
def Phi_p(P, x, th):
    r = x[0]
    feats = [x[0:1], x[1:2]]
    if BL:
        feats.append(jnp.exp(-th[0] * (1.0 - r))[None])
    inp = jnp.concatenate(feats + [theta_n(th)])
    return r**2 * (-U_of(P, th) / 2 + (1 - r) * mlp(P["p"], inp))

def Phi_f_reg(P, x, th):
    s = x[0]
    inp = jnp.concatenate([x, theta_n(th)])
    return -U_of(P, th) / 2 * s - C_of(P, th) * s + s * (1 - s) * mlp(P["f"], inp)

def Phi_f(P, x, th):
    return Phi_f_reg(P, x, th) + C_of(P, th) / x[0]

# =============================================================================
# 4. Operators (identical to Stage 1, acting on functions of x)
# =============================================================================
def E2(fun):
    def out(x):
        H = jax.hessian(fun)(x)
        return H[0, 0] + (1 - x[1]**2) / x[0]**2 * H[1, 1]
    return out

def Ls(fun):
    def out(x):
        g = jax.grad(fun)(x); H = jax.hessian(fun)(x)
        return x[0]**2 * H[0, 0] + 2 * x[0] * g[0] + (1 - x[1]**2) * H[1, 1]
    return out

def r_int(psi, lam):
    e2 = E2(psi); e4 = E2(e2)
    return lambda x: x[0]**4 * (e4(x) - lam**2 * e2(x))

def r_ext(psi):
    l1 = Ls(psi); l2 = Ls(lambda x: x[0]**2 * l1(x))
    return lambda x: l2(x) / x[0]**2

def interface(PhiP, PhiF, z, th):
    c_slip, c_jump = coefficients(th)
    eta = th[1]
    x = jnp.array([1.0, z])
    gp, Hp, Pp = jax.grad(PhiP)(x), jax.hessian(PhiP)(x), PhiP(x)
    gf, Hf, Pf = jax.grad(PhiF)(x), jax.hessian(PhiF)(x), PhiF(x)
    pr, prr = gp[0], Hp[0, 0]
    fr, frr = -gf[0], Hf[0, 0] + 2 * gf[0]
    S = lambda Phi, Pr, Prr, Pz, Pzz: Prr - 2 * Pr + 2 * Phi + 4 * z * Pz - (1 - z**2) * Pzz
    Sp = S(Pp, pr, prr, gp[1], Hp[1, 1]); Sf = S(Pf, fr, frr, gf[1], Hf[1, 1])
    return fr - pr - c_slip * Sf, Sf - eta * Sp + MARANG - c_jump * fr

psi_of = lambda Phi: (lambda x: (1 - x[1]**2) * Phi(x))

# =============================================================================
# 5. Self-test: exact solutions at random theta through the same operators
# =============================================================================
if args.mode == "selftest":
    rng = np.random.default_rng(1)
    for th in sample_theta(4, rng):
        A1, A2, B1, B2, U = exact(th)
        L = th[0]
        gj = (lambda r: r**4) if L == 0 else (lambda r: jnp.sinh(L * r) / (L * r) - jnp.cosh(L * r))
        Pp = lambda x: A1 * x[0]**2 + A2 * gj(x[0]) + 0 * x[1]
        Pf = lambda x: B1 / x[0] + B2 * x[0] + 0 * x[1]
        Pf_reg = lambda x: (B1 + B2) * x[0] + 0 * x[1]
        xs = [jnp.array(v) for v in [(0.3, 0.2), (0.8, -0.6), (0.97, 0.9)]]
        ri = max(abs(float(r_int(psi_of(Pp), L)(x))) for x in xs)
        re = max(abs(float(r_ext(psi_of(Pf_reg))(x))) for x in xs)
        sl, st = interface(Pp, Pf, 0.3, jnp.array(th))
        print(f"theta = {np.round(th, 3)}  U = {U:.6f}  max|R_int| {ri:.1e}  max|R_ext| {re:.1e}  "
              f"slip {abs(float(sl)):.1e}  stress {abs(float(st)):.1e}")
    raise SystemExit

# =============================================================================
# 6. Collocation and loss
# =============================================================================
rng = np.random.default_rng(args.seed)
X_INT = jnp.array(np.stack([rng.uniform(0.02, 1.0, args.n_int), rng.uniform(-1, 1, args.n_int)], 1))
X_EXT = jnp.array(np.stack([rng.uniform(1e-3, 1.0, args.n_ext), rng.uniform(-1, 1, args.n_ext)], 1))
Z_IF = jnp.array(np.cos(rng.uniform(0, np.pi, args.n_if)))
TH_INT = jnp.array(sample_theta(args.n_int, rng))
TH_EXT = jnp.array(sample_theta(args.n_ext, rng))
TH_IF = jnp.array(sample_theta(args.n_if, rng))
WPOW = {"none": 0.0, "lam2": 1.0, "lam4": 2.0}[args.param_weight]       # weight (1+lambda^2)^-WPOW on all terms
wfun = lambda th: (1.0 + th[:, 0]**2) ** (-WPOW)
W_INT, W_EXT, W_IF = wfun(TH_INT), wfun(TH_EXT), wfun(TH_IF)

def point_int(P, x, th):
    return r_int(psi_of(lambda y: Phi_p(P, y, th)), th[0])(x)

def point_ext(P, x, th):
    return r_ext(psi_of(lambda y: Phi_f_reg(P, y, th)))(x)

def point_if(P, z, th):
    return interface(lambda y: Phi_p(P, y, th), lambda y: Phi_f(P, y, th), z, th)

def loss_terms(P):
    rp = jax.vmap(point_int, (None, 0, 0))(P, X_INT, TH_INT)
    rf = jax.vmap(point_ext, (None, 0, 0))(P, X_EXT, TH_EXT)
    sl, st = jax.vmap(point_if, (None, 0, 0))(P, Z_IF, TH_IF)
    return (jnp.mean(W_INT * rp**2), jnp.mean(W_EXT * rf**2), jnp.mean(W_IF * sl**2), jnp.mean(W_IF * st**2))

def total_loss(P):
    a, b, c, d = loss_terms(P)
    return a + b + args.w_if * (c + d)

kp, kf, ku = jax.random.split(jax.random.PRNGKey(args.seed), 3)
lay = lambda s_: [int(v) for v in s_.split(",")]
P0 = {"p": init_mlp(kp, [(3 if BL else 2) + 4] + lay(args.int_layers) + [1]),
      "f": init_mlp(kf, [2 + 4] + lay(args.ext_layers) + [1]),
      "u": init_mlp(ku, [4] + lay(args.u_layers) + [1], zero_last=True)}
flat0, unravel = ravel_pytree(P0)
vg = jax.jit(jax.value_and_grad(lambda w: total_loss(unravel(w))))
terms = jax.jit(lambda w: loss_terms(unravel(w)))

# =============================================================================
# 7. Training (L-BFGS, checkpointed; same protocol as Stage 1)
# =============================================================================
CK = f"ckpt_{args.tag}.npz"
w, it_lbfgs = flat0, 0
if os.path.exists(CK):
    d = np.load(CK); w, it_lbfgs = jnp.array(d["w"]), int(d["it_lbfgs"])

if args.mode == "lbfgs":
    t0 = time.time()
    def fn(wn):
        val, g = vg(jnp.array(wn))
        return float(val), np.asarray(g, dtype=np.float64)
    res = minimize(fn, np.array(w), jac=True, method="L-BFGS-B",
                   options={"maxiter": args.iters, "maxcor": 50, "ftol": 1e-18, "gtol": 1e-13})
    w = jnp.array(res.x); it_lbfgs += int(res.nit)
    t = [float(v) for v in terms(w)]
    print(f"lbfgs {it_lbfgs:6d} loss {res.fun:.3e} [int {t[0]:.1e} ext {t[1]:.1e} slip {t[2]:.1e} "
          f"stress {t[3]:.1e}]  ({time.time() - t0:.0f} s)")
    np.savez(CK, w=np.array(w), it_lbfgs=it_lbfgs)
    print(f"checkpoint {CK} saved")
    raise SystemExit

# =============================================================================
# 8. Evaluation: lambda sweep (mode eval) or two-parameter atlas slices (mode atlas)
# =============================================================================
P = unravel(w)
rg_in, rg_out, zg = np.linspace(0.02, 1.0, 60), np.linspace(1.0, 20.0, 80), np.linspace(-0.99, 0.99, 21)
R1, Z1 = np.meshgrid(rg_in, zg); R2, Z2 = np.meshgrid(rg_out, zg)
XI = jnp.array(np.stack([R1.ravel(), Z1.ravel()], 1)); XE = jnp.array(np.stack([1 / R2.ravel(), Z2.ravel()], 1))
field_p = jax.jit(lambda th: jax.vmap(lambda x: Phi_p(P, x, th))(XI))
field_f = jax.jit(lambda th: jax.vmap(lambda x: Phi_f(P, x, th))(XE))
resid_p = jax.jit(lambda th: jax.vmap(lambda x: point_int(P, x, th))(XI))
qfun = jax.jit(lambda th: U_of(P, th) if args.branch == "free" else C_of(P, th))

def force(th, R, n=1001):
    psi = lambda r, t: jnp.sin(t)**2 * Phi_f(P, jnp.array([1.0 / r, jnp.cos(t)]), th)
    d = lambda f, i: jax.grad(f, argnums=i)
    E2p = lambda r, t: d(d(psi, 0), 0)(r, t) + jnp.sin(t) / r**2 * d(lambda r_, t_: d(psi, 1)(r_, t_) / jnp.sin(t_), 1)(r, t)
    ur = lambda r, t: -d(psi, 1)(r, t) / (r**2 * jnp.sin(t))
    ut = lambda r, t: d(psi, 0)(r, t) / (r * jnp.sin(t))
    S = lambda r, t: r * d(lambda r_, t_: ut(r_, t_) / r_, 0)(r, t) + d(ur, 1)(r, t) / r
    tt = (jnp.arange(n) + 0.5) * jnp.pi / n
    ev = lambda f: jax.vmap(lambda t: f(R, t))(tt)
    dp = ev(lambda r, t: d(E2p, 0)(r, t) / jnp.sin(t))
    p = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dp) * (jnp.pi / n)])[:-1]
    integ = ((-p + 2 * ev(d(ur, 0))) * jnp.cos(tt) - ev(S) * jnp.sin(tt)) * jnp.sin(tt)
    return 2 * jnp.pi * R**2 * jnp.sum(integ) * (jnp.pi / n)
R_LIST = [1.0, 2.0, 5.0, 10.0]
force_j = {R: jax.jit(lambda th, R=R: force(th, R)) for R in R_LIST}

def valid(th):
    return th[3] == 0 or (th[0] > 0 and th[1] > 0 and th[3] <= th[0] * np.sqrt(th[1]) + 1e-12)

def evaluate(th):
    """All metrics at one parameter vector (lam, eta, xi, beta)."""
    co = exact(th); thj = jnp.array(th)
    q = float(qfun(thj))
    ex_q = co[4] if args.branch == "free" else co[2]
    ex_p = co[0] * R1**2 + co[1] * g_exact(th[0], R1); ex_f = co[2] * R2 + co[3] / R2
    fp = np.array(field_p(thj)).reshape(R1.shape); ff = np.array(field_f(thj)).reshape(R2.shape)
    l2 = lambda a_, b_: float(np.linalg.norm(a_ - b_) / np.linalg.norm(b_))
    rp = np.array(resid_p(thj))
    F = [float(force_j[R](thj)) for R in R_LIST]
    fprof = (max(abs(f) for f in F) / (6 * np.pi * abs(q)) if args.branch == "free"
             else max(abs(f / (8 * np.pi * q) - 1) for f in F))
    return dict(lam=float(th[0]), eta=float(th[1]), xi=float(th[2]), beta=float(th[3]),
                exact=float(ex_q), pinn=q, rel_err=abs(q - ex_q) / abs(ex_q), signed=(q - ex_q) / abs(ex_q),
                L2_interior=l2(fp, ex_p), L2_exterior=l2(ff, ex_f),
                zeta=float(max(np.max(np.std(fp, 0)) / np.max(np.abs(fp)), np.max(np.std(ff, 0)) / np.max(np.abs(ff)))),
                force_profile=float(fprof),
                resid_interior=float(np.sqrt(np.mean(rp**2)) / ((1 + th[0]**2)**2 * np.max(np.abs(fp)))))

eta0, xi0, beta0 = args.ref
la, lb = RANGES[0]
lam_axis = lambda n: (np.geomspace(la, lb, n) if (VARIES[0] and LOGSCALE[0]) else np.linspace(la, lb, n)) if VARIES[0] else np.array([la])

if args.mode == "eval":
    TH = [np.array([L, eta0, xi0, beta0]) for L in lam_axis(args.n_eval)]
    rows = [evaluate(th) for th in TH if valid(th)]
    errs = np.array([r_["rel_err"] for r_ in rows]); lams = np.array([r_["lam"] for r_ in rows])
    summary = dict(median_err=float(np.median(errs)), max_err=float(errs.max()),
                   max_err_lam_le_10=float(errs[lams <= 10].max()) if (lams <= 10).any() else None,
                   max_force_profile=float(max(r_["force_profile"] for r_ in rows)),
                   max_resid_interior=float(max(r_["resid_interior"] for r_ in rows)))
    json.dump(dict(config=vars(args), iterations=it_lbfgs, summary=summary, grid=rows),
              open(f"results_{args.tag}.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"{'lam':>7} {'exact':>9} {'pinn':>9} {'rel.err':>8} {'force':>8} {'res_int':>8}")
    for r_ in rows:
        print(f"{r_['lam']:7.3f} {r_['exact']:9.5f} {r_['pinn']:9.5f} {r_['rel_err']:8.1e} {r_['force_profile']:8.1e} {r_['resid_interior']:8.1e}")

if args.mode == "atlas":
    n = args.n_atlas
    axes = {"beta": (3, np.linspace(RANGES[3, 0], RANGES[3, 1], n)),
            "eta": (1, np.geomspace(RANGES[1, 0], RANGES[1, 1], n) if RANGES[1, 0] > 0 else np.linspace(*RANGES[1], n)),
            "xi": (2, np.linspace(RANGES[2, 0], RANGES[2, 1], n))}
    rows = []
    for name, (j, vals) in axes.items():
        if not VARIES[j]:
            continue
        for L in lam_axis(n):
            for v in vals:
                th = np.array([L, eta0, xi0, beta0]); th[j] = v
                if valid(th):
                    r_ = evaluate(th); r_["slice"] = f"lam-{name}"; rows.append(r_)
        sl = [r_ for r_ in rows if r_["slice"] == f"lam-{name}"]
        e = np.array([r_["rel_err"] for r_ in sl]); lm = np.array([r_["lam"] for r_ in sl])
        print(f"slice lam-{name}: {len(sl)} points, median err {np.median(e):.1e}, max err {e.max():.1e}, "
              f"max err (lam<=10) {e[lm <= 10].max():.1e}, max force {max(r_['force_profile'] for r_ in sl):.1e}")
    json.dump(dict(config=vars(args), iterations=it_lbfgs, atlas=rows), open(f"atlas_{args.tag}.json", "w"), indent=2)
    print(f"written atlas_{args.tag}.json ({len(rows)} points)")
