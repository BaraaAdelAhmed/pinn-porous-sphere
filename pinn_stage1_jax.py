"""
Stage 1 PINN: thermocapillary migration of a Brinkman porous sphere with
Beavers-Joseph slip and stress jump (low Peclet number).  JAX implementation.

Each block below corresponds to a subsection of Section 3 of the article:
  3.1  stream-function formulation  -> representation of psi_i = (1 - z^2) Phi_i
  3.2  two-domain architecture      -> separate MLPs M_p, M_f; hard interface value
  3.3  unbounded exterior           -> s = 1/r, decaying exterior, s^-2-scaled residual
  3.4  force-free condition         -> branch 'free' (U learned, C = 0)
                                       branch 'held' (U = 0, C learned, F_T = 8 pi C)
  3.5  loss                         -> r^4-weighted interior residual, interface weight
  3.7  metrics                      -> U_T / F_T error, field L2 errors, zeta diagnostic

The exact solution is used ONLY in `selftest` and `eval`, never in training.

Usage (each call resumes from the checkpoint of the same --tag):
  python pinn_stage1_jax.py --mode selftest
  python pinn_stage1_jax.py --mode lbfgs --iters 1000 --tag base
  python pinn_stage1_jax.py --mode lbfgs --iters 1000 --tag base      # continue
  python pinn_stage1_jax.py --mode eval --tag base
  python pinn_stage1_jax.py --lam 8 --mode lbfgs --iters 1000 --tag lam8
Optional Adam warm-up:  --mode adam --iters 1500 --lr 2e-3

Status: free branch tested at (lam, beta) = (2, 0.5), (8, 0.5), (2, 2.0);
        held branch implemented but NOT yet tested.
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
ap.add_argument("--lam", type=float, default=2.0)    # permeability parameter lambda
ap.add_argument("--eta", type=float, default=1.0)    # viscosity ratio
ap.add_argument("--xi", type=float, default=0.5)     # stress-jump coefficient
ap.add_argument("--beta", type=float, default=0.5)   # Beavers-Joseph slip coefficient
ap.add_argument("--k", type=float, default=1.0)      # thermal-conductivity ratio
ap.add_argument("--branch", choices=["free", "held"], default="free")
ap.add_argument("--mode", choices=["selftest", "adam", "lbfgs", "rar", "eval", "evalres"], default="selftest")
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--tag", default="run")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--w_if", type=float, default=10.0)  # interface loss weight
ap.add_argument("--n_int", type=int, default=800)
ap.add_argument("--n_ext", type=int, default=800)
ap.add_argument("--n_if", type=int, default=64)
ap.add_argument("--int_layers", default="32,32,32")
ap.add_argument("--ext_layers", default="24,24")
ap.add_argument("--n_cand", type=int, default=10000)  # RAR: candidate points per domain
ap.add_argument("--n_add", type=int, default=200)     # RAR: points added per domain
# ---- ablation options (Section 4.3); defaults = baseline formulation ----
ap.add_argument("--act", choices=["tanh", "sin", "silu"], default="tanh")         # activation function
ap.add_argument("--no_r4", action="store_true")                                  # drop the r^4 interior weight
ap.add_argument("--exterior", choices=["decay", "nondecay"], default="decay")    # Eq. (Nf-final) vs earlier form
ap.add_argument("--resid_ext", choices=["scaled", "unscaled"], default="scaled") # s^-2 scaling of exterior residual
ap.add_argument("--int_sampling", choices=["uniform", "interface"], default="uniform")  # interior r sampling
# ---- Stage 2 remedies for the screening layer (Section 5); defaults = Stage 1 formulation ----
ap.add_argument("--resid_scale", choices=["none", "lambda4"], default="none")  # divide R_p by (1+lam^2)^2
ap.add_argument("--bl_feature", action="store_true")                         # extra interior input exp(-lam(1-r))
# ---- Section 5.5: inverse demonstration -- infer beta from a measured U_T ----
ap.add_argument("--inverse_beta", action="store_true")   # beta unknown; --beta is the TRUE value (generates the data)
ap.add_argument("--noise", type=float, default=0.0)      # relative error of the measurement: U_meas = U_T (1 + noise)
ap.add_argument("--w_data", type=float, default=100.0)   # weight of the data term ((U - U_meas)/U_meas)^2
args = ap.parse_args()

# Dimensionless interface coefficients. These reproduce Eq. (42) of the base paper
# exactly (verified symbolically), i.e. they use sqrt(K)/a = sqrt(eta)/lambda.
LAM, ETA, XI, BETA, KAPPA = args.lam, args.eta, args.xi, args.beta, args.k
if BETA == 0:                          # velocity continuity: no slip term (also allows lam = 0 or eta = 0)
    C_SLIP = 0.0
elif LAM > 0 and ETA > 0:
    C_SLIP = BETA / (LAM * np.sqrt(ETA))   # dimensionless slip length
else:
    raise SystemExit("beta > 0 needs lam > 0 and eta > 0: the slip length beta/(lam sqrt(eta)) would be infinite")
C_JUMP = LAM * np.sqrt(ETA) * XI       # dimensionless stress-jump coefficient
MARANG = 3.0 / (2.0 + KAPPA)           # Marangoni forcing amplitude (conduction solution)

# =============================================================================
# 1. Exact reference (evaluation only)
# =============================================================================
def exact_reference():
    """Solve the 4x4 interface system for f_p = A1 r^2 + A2 g(r), f_f = B1 r + B2/r."""
    L = LAM
    if L == 0:                                         # lam = 0 (viscous drop): regular interior solution r^4
        g0, g1, g2 = 1.0, 4.0, 12.0
    else:
        sh, ch = np.sinh(L), np.cosh(L)
        g0 = sh / L - ch                               # g(1)
        g1 = ch - sh / L - L * sh                      # g'(1)
        g2 = -2 * ch + 2 * sh / L + L * sh - L**2 * ch # g''(1)
    # rows: fp(1)+U/2=0, ff(1)+U/2=0, slip, stress ; unknown order depends on branch
    # S = f'' - 2 f' + 2 f at r = 1 ;  S_f = 6 B2 ; S_p = A1*2 + A2*(g2 - 2 g1 + 2 g0)
    Sp_A1, Sp_A2 = 2.0 - 4.0 + 2.0, g2 - 2 * g1 + 2 * g0
    # columns: A1, A2, B1, B2, X  (X = U for free, X = unused for held)
    if args.branch == "free":
        M = np.array([
            [1, g0, 0, 0, 0.5],
            [0, 0, 1, 1, 0.5],
            [-2, -g1, 1, -1 - 6 * C_SLIP, 0],
            [-ETA * Sp_A1, -ETA * Sp_A2, -C_JUMP, 6 + C_JUMP, 0],
            [0, 0, 1, 0, 0]])                          # B1 = 0 (force-free)
        rhs = np.array([0, 0, 0, -MARANG, 0])
        A1, A2, B1, B2, U = np.linalg.solve(M, rhs)
    else:
        M = np.array([
            [1, g0, 0, 0],
            [0, 0, 1, 1],
            [-2, -g1, 1, -1 - 6 * C_SLIP],
            [-ETA * Sp_A1, -ETA * Sp_A2, -C_JUMP, 6 + C_JUMP]])
        rhs = np.array([0, 0, 0, -MARANG])
        A1, A2, B1, B2 = np.linalg.solve(M, rhs)
        U = 0.0
    return dict(A1=A1, A2=A2, B1=B1, B2=B2, U=U, F=8 * np.pi * B1)

def eq42_UT():
    """Closed-form migration velocity, Eq. (42)-(43) of the base paper (cross-check)."""
    L, e, b, x = LAM, ETA, BETA, XI
    DU = (3 * L * (12 * b * np.sqrt(e) + L * (-6 + 6 * e + L**2 * e + L * np.sqrt(e) * (2 * b - x)))) * np.cosh(L) \
       + (6 * L * (3 + L**2) - 18 * (2 + L**2) * b * np.sqrt(e) - 9 * L * (2 + L**2) * e
          + L**2 * (3 + L**2) * np.sqrt(e) * x) * np.sinh(L)
    return 6 * L * ((3 + L**2) * np.sinh(L) - 3 * L * np.cosh(L)) / ((2 + KAPPA) * DU)

EX = exact_reference()
g_exact = (lambda r: r**4) if LAM == 0 else (lambda r: np.sinh(LAM * r) / (LAM * r) - np.cosh(LAM * r))
fp_exact = lambda r: EX["A1"] * r**2 + EX["A2"] * g_exact(r)
ff_exact = lambda r: EX["B1"] * r + EX["B2"] / r

# =============================================================================
# 2. Networks (Sec. 3.2)
# =============================================================================
def init_mlp(key, sizes):
    params = []
    for m, n in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        params.append((jax.random.normal(sub, (m, n)) * np.sqrt(2.0 / (m + n)), jnp.zeros(n)))  # Glorot
    return params

ACT = {"tanh": jnp.tanh, "sin": jnp.sin, "silu": jax.nn.silu}[args.act]

def mlp(params, x):
    for W, b in params[:-1]:
        x = ACT(x @ W + b)
    W, b = params[-1]
    return (x @ W + b)[0]

def interior_features(x):
    """Inputs of the interior network: (r, zeta), plus exp(-lam(1-r)) with --bl_feature (Section 5 remedy).
    The feature resolves the screening layer of width 1/lam at the interface; it equals 1 for lam = 0."""
    if args.bl_feature:
        return jnp.array([x[0], x[1], jnp.exp(-LAM * (1.0 - x[0]))])
    return x

N_IN_P = 3 if args.bl_feature else 2

def branch_scalars(P):
    """Sec. 3.4, Eq. (branches): which of U and C is learned."""
    if args.branch == "free":
        return P["q"], 0.0
    return 0.0, P["q"]

# =============================================================================
# 3. Representation (Secs. 3.1-3.3)
# =============================================================================
def Phi_p(P, x):                       # x = (r, z);  Phi_p = r^2 N_p
    r = x[0]
    U, _ = branch_scalars(P)
    return r**2 * (-U / 2 + (1 - r) * mlp(P["p"], interior_features(x)))

def Phi_f_reg(P, x):                   # x = (s, z); exterior WITHOUT the Stokeslet term
    s = x[0]
    U, C = branch_scalars(P)
    if args.exterior == "nondecay":    # earlier form: -U/2 + C(1/s - 1) + (1-s) M_f, Stokeslet C/s removed here
        return -U / 2 - C + (1 - s) * mlp(P["f"], x)
    return -U / 2 * s - C * s + s * (1 - s) * mlp(P["f"], x)

def Phi_f(P, x):                       # full exterior amplitude, Eq. (Nf-final)
    _, C = branch_scalars(P)
    return Phi_f_reg(P, x) + C / x[0]

# =============================================================================
# 4. Operators
# =============================================================================
def E2(fun):                           # interior, coordinates (r, z)
    def out(x):
        H = jax.hessian(fun)(x)
        return H[0, 0] + (1 - x[1]**2) / x[0]**2 * H[1, 1]
    return out

def Ls(fun):                           # exterior, coordinates (s, z): d/ds(s^2 d/ds) + (1-z^2) d2/dz2
    def out(x):
        g = jax.grad(fun)(x)
        H = jax.hessian(fun)(x)
        return x[0]**2 * H[0, 0] + 2 * x[0] * g[0] + (1 - x[1]**2) * H[1, 1]
    return out

def res_int_unscaled(psi):             # r^4 [E^2 (E^2 - lam^2) psi]   (Sec. 3.5)
    e2 = E2(psi); e4 = E2(e2)
    if args.no_r4:
        return lambda x: e4(x) - LAM**2 * e2(x)
    return lambda x: x[0]**4 * (e4(x) - LAM**2 * e2(x))

RP_SCALE = 1.0 / (1.0 + LAM**2)**2 if args.resid_scale == "lambda4" else 1.0   # ~ lam^-4 for large lam, 1 at lam = 0

def res_int(psi):                      # optionally rescaled interior residual (Section 5 remedy)
    r0 = res_int_unscaled(psi)
    return lambda x: RP_SCALE * r0(x)

def res_ext(psi):                      # s^-2 L_s(s^2 L_s psi)          (Sec. 3.3, Eq. Rf)
    l1 = Ls(psi)
    l2 = Ls(lambda x: x[0]**2 * l1(x))
    if args.resid_ext == "unscaled":
        return lambda x: l2(x)
    return lambda x: l2(x) / x[0]**2

def interface_terms(PhiP, PhiF, z, c_slip=None):
    """Slip and tangential-stress residuals at r = 1, divided by sqrt(1 - z^2)."""
    x = jnp.array([1.0, z])
    gp, Hp, Pp = jax.grad(PhiP)(x), jax.hessian(PhiP)(x), PhiP(x)
    gf, Hf, Pf = jax.grad(PhiF)(x), jax.hessian(PhiF)(x), PhiF(x)
    pr, prr = gp[0], Hp[0, 0]
    fr, frr = -gf[0], Hf[0, 0] + 2 * gf[0]      # d/dr = -s^2 d/ds ; d2/dr2 = s^4 d2/ds2 + 2 s^3 d/ds (s = 1)
    S = lambda Phi, Pr, Prr, Pz, Pzz: Prr - 2 * Pr + 2 * Phi + 4 * z * Pz - (1 - z**2) * Pzz
    Sp = S(Pp, pr, prr, gp[1], Hp[1, 1])
    Sf = S(Pf, fr, frr, gf[1], Hf[1, 1])
    slip = fr - pr - (C_SLIP if c_slip is None else c_slip) * Sf
    stress = Sf - ETA * Sp + MARANG - C_JUMP * fr
    return slip, stress

# =============================================================================
# 5. Self-test: exact solution through the same operators (expect ~1e-12 or less)
# =============================================================================
if args.mode == "selftest":
    print(f"exact U = {EX['U']:.12f}   Eq.(42) U_T = {eq42_UT():.12f}   F = {EX['F']:.6e}")
    gj = (lambda r: r**4) if LAM == 0 else (lambda r: jnp.sinh(LAM * r) / (LAM * r) - jnp.cosh(LAM * r))
    Pp = lambda x: EX["A1"] * x[0]**2 + EX["A2"] * gj(x[0])
    Pf = lambda x: EX["B1"] / x[0] + EX["B2"] * x[0] + 0 * x[1]       # exact Phi_f = B1 r + B2 s
    # regular part used in the residual: Phi_f - B1/s + B1 s = (B1 + B2) s   (cf. Phi_f_reg)
    Pf_reg = lambda x: (EX["B1"] + EX["B2"]) * x[0] + 0 * x[1]
    psi = lambda Phi: (lambda x: (1 - x[1]**2) * Phi(x))
    pts = [jnp.array(v) for v in [(0.3, 0.2), (0.7, -0.6), (0.95, 0.9)]]
    print("interior residual :", [f"{float(res_int(psi(Pp))(x)):.1e}" for x in pts])
    print("exterior residual :", [f"{float(res_ext(psi(Pf_reg))(x)):.1e}" for x in pts])
    print("interface (slip, stress) at z=0.3:", [f"{float(v):.1e}" for v in interface_terms(Pp, Pf, 0.3)])
    raise SystemExit

# =============================================================================
# 5b. Inverse demonstration (Section 5.5)
# =============================================================================
if args.inverse_beta:
    if args.branch != "free" or BETA <= 0 or LAM <= 0 or ETA <= 0:
        raise SystemExit("--inverse_beta needs the free branch and lam, eta, beta > 0")
    U_MEAS = EX["U"] * (1.0 + args.noise)

def slip_coeff(P):
    """Dimensionless slip length beta/(lam sqrt(eta)): learned (sigmoid, in (0,1)) or fixed."""
    return jax.nn.sigmoid(P["b"]) if args.inverse_beta else C_SLIP

def U_of_slip(l):
    """Exact U_T for dimensionless slip length l (other parameters as given)."""
    L = LAM; sh, ch = np.sinh(L), np.cosh(L)
    g0, g1, g2 = sh / L - ch, ch - sh / L - L * sh, -2 * ch + 2 * sh / L + L * sh - L**2 * ch
    M = np.array([[1, g0, 0, 0, .5], [0, 0, 1, 1, .5], [-2, -g1, 1, -1 - 6 * l, 0],
                  [0, -ETA * (g2 - 2 * g1 + 2 * g0), -C_JUMP, 6 + C_JUMP, 0], [0, 0, 1, 0, 0]])
    return np.linalg.solve(M, [0, 0, 0, -MARANG, 0])[4]

def exact_inverse_slip(U_target):
    """The slip length that reproduces U_target exactly (bisection; U_T decreases with l)."""
    a, b = 0.0, 1.0
    if not (U_of_slip(b) <= U_target <= U_of_slip(a)):
        return float("nan")
    for _ in range(200):
        m = 0.5 * (a + b)
        a, b = (m, b) if U_of_slip(m) > U_target else (a, m)
    return 0.5 * (a + b)

# =============================================================================
# 6. Collocation and loss (Secs. 3.5-3.6)
# =============================================================================
rng = np.random.default_rng(args.seed)
_u = rng.uniform(0.0, 1.0, args.n_int)
_r = 0.02 + 0.98 * _u if args.int_sampling == "uniform" else 1.0 - 0.98 * _u**2   # "interface": density ~ 1/sqrt(1-r)
X_INT = jnp.array(np.stack([_r, rng.uniform(-1, 1, args.n_int)], 1))
X_EXT = jnp.array(np.stack([rng.uniform(1e-3, 1.0, args.n_ext), rng.uniform(-1, 1, args.n_ext)], 1))
Z_IF = jnp.array(np.cos(np.linspace(0, np.pi, args.n_if)))
CK = f"ckpt_{args.tag}.npz"
if os.path.exists(CK) and "x_int_add" in np.load(CK):        # points added by residual-based refinement
    _d = np.load(CK)
    X_INT = jnp.concatenate([X_INT, jnp.array(_d["x_int_add"])])
    X_EXT = jnp.concatenate([X_EXT, jnp.array(_d["x_ext_add"])])

def loss_terms(P):
    psi_p = lambda x: (1 - x[1]**2) * Phi_p(P, x)
    psi_f_reg = lambda x: (1 - x[1]**2) * Phi_f_reg(P, x)   # Stokeslet is an exact solution: excluded
    rp = jax.vmap(res_int(psi_p))(X_INT)
    rf = jax.vmap(res_ext(psi_f_reg))(X_EXT)
    sl, st = jax.vmap(lambda z: interface_terms(lambda x: Phi_p(P, x), lambda x: Phi_f(P, x), z, slip_coeff(P)))(Z_IF)
    return jnp.mean(rp**2), jnp.mean(rf**2), jnp.mean(sl**2), jnp.mean(st**2)

def total_loss(P):
    a, b, c, d = loss_terms(P)
    data = args.w_data * ((P["q"] - U_MEAS) / U_MEAS)**2 if args.inverse_beta else 0.0
    return a + b + args.w_if * (c + d) + data

key_p, key_f = jax.random.split(jax.random.PRNGKey(args.seed))
sizes = lambda s: [2] + [int(v) for v in s.split(",")] + [1]
P0 = {"p": init_mlp(key_p, [N_IN_P] + sizes(args.int_layers)[1:]),
      "f": init_mlp(key_f, sizes(args.ext_layers)),
      "q": jnp.array(0.0)}                 # U (free) or C (held), initialized at zero
if args.inverse_beta:
    P0["b"] = jnp.array(0.0)             # slip length l = sigmoid(b) in (0, 1): starts at 0.5, mid valid range
flat0, unravel = ravel_pytree(P0)
vg = jax.jit(jax.value_and_grad(lambda w: total_loss(unravel(w))))
terms = jax.jit(lambda w: loss_terms(unravel(w)))

# =============================================================================
# 7. Training with checkpoints
# =============================================================================
x_int_add, x_ext_add = np.zeros((0, 2)), np.zeros((0, 2))
if os.path.exists(CK):
    d = np.load(CK)
    if "x_int_add" in d:
        x_int_add, x_ext_add = d["x_int_add"], d["x_ext_add"]
    w, m, v = jnp.array(d["w"]), jnp.array(d["m"]), jnp.array(d["v"])
    it_adam, it_lbfgs = int(d["it_adam"]), int(d["it_lbfgs"])
else:
    w, m, v = flat0, jnp.zeros_like(flat0), jnp.zeros_like(flat0)
    it_adam = it_lbfgs = 0

target = EX["U"] if args.branch == "free" else EX["B1"]
def report(tag, i, w):
    val = float(vg(w)[0]); t = [float(x) for x in terms(w)]; q = float(unravel(w)["q"])
    name = "U" if args.branch == "free" else "C"
    if args.inverse_beta:
        bl = float(jax.nn.sigmoid(unravel(w)["b"])) * LAM * np.sqrt(ETA)
        print(f"{tag} {i:6d} beta {bl:.5f} (true {BETA:g}, rel.err {abs(bl - BETA) / BETA:.2e})", flush=True)
    print(f"{tag} {i:6d} loss {val:.3e} [int {t[0]:.1e} ext {t[1]:.1e} slip {t[2]:.1e} stress {t[3]:.1e}] "
          f"{name} {q:.6f} rel.err {abs(q - target) / abs(target):.2e}", flush=True)

t0 = time.time()
if args.mode == "adam":
    b1, b2, eps = 0.9, 0.999, 1e-8
    for it in range(it_adam + 1, it_adam + args.iters + 1):
        _, g = vg(w)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        w = w - args.lr * (m / (1 - b1**it)) / (jnp.sqrt(v / (1 - b2**it)) + eps)
        if it % 250 == 0:
            report("adam ", it, w)
    it_adam += args.iters
elif args.mode == "lbfgs":
    def fn(wn):
        val, g = vg(jnp.array(wn))
        return float(val), np.asarray(g, dtype=np.float64)
    res = minimize(fn, np.array(w), jac=True, method="L-BFGS-B",
                   options={"maxiter": args.iters, "maxcor": 50, "ftol": 1e-18, "gtol": 1e-13})
    w = jnp.array(res.x); it_lbfgs += int(res.nit)
    report("lbfgs", it_lbfgs, w); print(res.message)
elif args.mode == "rar":
    # Residual-based adaptive refinement: add the n_add candidates with the largest residual
    P = unravel(w)
    cr = np.random.default_rng(args.seed + 1000 + len(x_int_add))
    ci = np.stack([cr.uniform(0.02, 1.0, args.n_cand), cr.uniform(-1, 1, args.n_cand)], 1)
    ce = np.stack([cr.uniform(1e-3, 1.0, args.n_cand), cr.uniform(-1, 1, args.n_cand)], 1)
    ri = np.abs(np.array(jax.jit(jax.vmap(res_int(lambda x: (1 - x[1]**2) * Phi_p(P, x))))(jnp.array(ci))))
    re = np.abs(np.array(jax.jit(jax.vmap(res_ext(lambda x: (1 - x[1]**2) * Phi_f_reg(P, x))))(jnp.array(ce))))
    x_int_add = np.concatenate([x_int_add, ci[np.argsort(ri)[-args.n_add:]]])
    x_ext_add = np.concatenate([x_ext_add, ce[np.argsort(re)[-args.n_add:]]])
    print(f"RAR: added {args.n_add} + {args.n_add} points; interior candidates max|R| {ri.max():.2e} "
          f"(median {np.median(ri):.2e}), exterior max|R| {re.max():.2e} (median {np.median(re):.2e})")
    print("  added interior points: r quartiles", np.round(np.percentile(x_int_add[:, 0], [25, 50, 75]), 3),
          " exterior points: s quartiles", np.round(np.percentile(x_ext_add[:, 0], [25, 50, 75]), 3))
if args.mode in ("adam", "lbfgs", "rar"):
    np.savez(CK, w=np.array(w), m=np.array(m), v=np.array(v), it_adam=it_adam, it_lbfgs=it_lbfgs,
             x_int_add=x_int_add, x_ext_add=x_ext_add)
    print(f"checkpoint {CK} saved ({time.time() - t0:.0f}s)")
    raise SystemExit

# =============================================================================
# 8. Evaluation (Sec. 3.7)
# =============================================================================
def residual_diagnostics(P):
    """Reference-free residual diagnostics on the evaluation grids (Section 5.1).
    Always the Stage 1 residual forms -- r^4 E^2(E^2 - lam^2) psi_p and s^-2 L_s(s^2 L_s psi_f) --
    whatever scaling or variant was used in training, normalized by the field amplitude
    (and (1+lam^2)^2 for the interior), so values are comparable across formulations."""
    rg_in, rg_out, zg = np.linspace(0.02, 1.0, 120), np.linspace(1.0, 20.0, 160), np.linspace(-0.99, 0.99, 41)
    R1, Z1 = np.meshgrid(rg_in, zg); R2, Z2 = np.meshgrid(rg_out, zg)
    Xi = jnp.array(np.stack([R1.ravel(), Z1.ravel()], 1)); Xe = jnp.array(np.stack([1 / R2.ravel(), Z2.ravel()], 1))
    psi_p = lambda x: (1 - x[1]**2) * Phi_p(P, x)
    e2 = E2(psi_p); e4 = E2(e2)
    rp = lambda x: x[0]**4 * (e4(x) - LAM**2 * e2(x))
    psi_f = lambda x: (1 - x[1]**2) * Phi_f_reg(P, x)
    l1 = Ls(psi_f); l2 = Ls(lambda x: x[0]**2 * l1(x))
    rf = lambda x: l2(x) / x[0]**2
    Rp = np.array(jax.jit(jax.vmap(rp))(Xi)); Rf = np.array(jax.jit(jax.vmap(rf))(Xe))
    Pp = np.array(jax.vmap(lambda x: Phi_p(P, x))(Xi)); Pf = np.array(jax.vmap(lambda x: Phi_f(P, x))(Xe))
    rms = lambda a: float(np.sqrt(np.mean(a**2)))
    return dict(resid_interior=float(rms(Rp) / ((1 + LAM**2)**2 * np.abs(Pp).max())),
                resid_exterior=float(rms(Rf) / np.abs(Pf).max()))

if args.mode == "evalres":          # add the residual diagnostics to an existing results file, nothing else
    P = unravel(w); rd = residual_diagnostics(P)
    fn = f"results_{args.tag}.json"
    if os.path.exists(fn):
        d = json.load(open(fn)); d.update(rd); json.dump(d, open(fn, "w"), indent=2); print(f"updated {fn}:", rd)
    else:
        print("no results file; diagnostics:", rd)
    raise SystemExit

def force_profile(PhiF_sz, R, n=2001):
    """Net axial force exerted by the fluid on everything inside the sphere r = R
    (units mu U0 a). Pressure from dp/dtheta = (1/sin th) d(E^2 psi)/dr, up to a constant
    that exerts no net force. Verified: Stokeslet B1 r sin^2 -> 8 pi B1 at every R;
    potential dipole -> 0; rigid translating sphere -> -6 pi U (Stokes drag)."""
    psi = lambda r, th: jnp.sin(th)**2 * PhiF_sz(jnp.array([1.0 / r, jnp.cos(th)]))
    d = lambda f, i: jax.grad(f, argnums=i)
    E2psi = lambda r, th: d(d(psi, 0), 0)(r, th) + jnp.sin(th) / r**2 * d(lambda r_, t_: d(psi, 1)(r_, t_) / jnp.sin(t_), 1)(r, th)
    ur = lambda r, th: -d(psi, 1)(r, th) / (r**2 * jnp.sin(th))
    ut = lambda r, th: d(psi, 0)(r, th) / (r * jnp.sin(th))
    S = lambda r, th: r * d(lambda r_, t_: ut(r_, t_) / r_, 0)(r, th) + d(ur, 1)(r, th) / r
    dpdth = lambda r, th: d(E2psi, 0)(r, th) / jnp.sin(th)
    th = (np.arange(n) + 0.5) * np.pi / n                       # midpoint rule in theta
    ev = lambda f: np.array(jax.jit(jax.vmap(lambda t: f(R, t)))(jnp.array(th)))
    p = np.concatenate([[0.0], np.cumsum(ev(dpdth)) * (np.pi / n)])[:-1]
    integrand = ((-p + 2 * ev(d(ur, 0))) * np.cos(th) - ev(S) * np.sin(th)) * np.sin(th)
    return float(2 * np.pi * R**2 * np.sum(integrand) * (np.pi / n))

P = unravel(w)
q = float(P["q"])
rg_in, rg_out, zg = np.linspace(0.02, 1.0, 120), np.linspace(1.0, 20.0, 160), np.linspace(-0.99, 0.99, 41)
R1, Z1 = np.meshgrid(rg_in, zg); R2, Z2 = np.meshgrid(rg_out, zg)
Phip = np.array(jax.vmap(lambda x: Phi_p(P, x))(jnp.array(np.stack([R1.ravel(), Z1.ravel()], 1)))).reshape(R1.shape)
Phif = np.array(jax.vmap(lambda x: Phi_f(P, x))(jnp.array(np.stack([1 / R2.ravel(), Z2.ravel()], 1)))).reshape(R2.shape)
ex_p, ex_f = fp_exact(R1), ff_exact(R2)
l2 = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
out = {"params": dict(lam=LAM, eta=ETA, xi=XI, beta=BETA, k=KAPPA, branch=args.branch),
       "iterations": dict(adam=it_adam, lbfgs=it_lbfgs)}
if args.branch == "free":
    out.update(U_exact=EX["U"], U_pinn=q, U_rel_err=abs(q - EX["U"]) / abs(EX["U"]))
    if args.inverse_beta:
        sq = LAM * np.sqrt(ETA)
        b_pinn = float(jax.nn.sigmoid(P["b"])) * sq
        b_ideal = exact_inverse_slip(U_MEAS) * sq          # what a perfect solver would infer from the noisy datum
        out.update(inverse=dict(beta_true=BETA, beta_pinn=b_pinn, beta_exact_inverse=b_ideal,
                                noise=args.noise, U_meas=U_MEAS,
                                beta_rel_err=abs(b_pinn - BETA) / BETA,
                                beta_rel_err_vs_exact_inverse=abs(b_pinn - b_ideal) / b_ideal))
else:
    out.update(F_exact=EX["F"], F_pinn=8 * np.pi * q, F_rel_err=abs(q - EX["B1"]) / abs(EX["B1"]))
out.update(residual_diagnostics(P))
out.update(L2_interior=l2(Phip, ex_p), L2_exterior=l2(Phif, ex_f),
           # reference-free: spurious angular variation relative to the PINN's own field
           zeta_spurious_interior=float(np.max(np.std(Phip, 0)) / np.max(np.abs(Phip))),
           zeta_spurious_exterior=float(np.max(np.std(Phif, 0)) / np.max(np.abs(Phif))),
           loss_terms=dict(zip(["interior", "exterior", "slip", "stress"], [float(t) for t in terms(w)])))
R_LIST = [1.0, 1.5, 2.0, 5.0, 10.0, 20.0]
F_R = [force_profile(lambda x: Phi_f(P, x), R) for R in R_LIST]
if args.branch == "free":     # should vanish at every R; normalized by the Stokes drag scale 6 pi |U|
    out["force_profile"] = {f"R={R:g}": F / (6 * np.pi * abs(q)) for R, F in zip(R_LIST, F_R)}
    out["force_profile_note"] = "F(R)/(6 pi |U|); zero at every R for a force-free particle"
else:                         # should equal 8 pi C at every R
    out["force_profile"] = {f"R={R:g}": F / (8 * np.pi * q) for R, F in zip(R_LIST, F_R)}
    out["force_profile_note"] = "F(R)/(8 pi C); one at every R"
print(json.dumps(out, indent=2))
json.dump(out, open(f"results_{args.tag}.json", "w"), indent=2)
np.savez(f"fields_{args.tag}.npz", rg_in=rg_in, rg_out=rg_out, zg=zg, Phip=Phip, Phif=Phif, ex_p=ex_p, ex_f=ex_f)
