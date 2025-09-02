#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V&V harness (all-green):
 - MMS (manufactured solution) for u_t = D Δu - k u + f
   * Space order: FIXED tiny dt across meshes to isolate spatial truncation
   * Time order: refine dt on a fine, fixed mesh
 - Gaussian diffusion sanity + mass bound
 - PK two-compartment: numeric vs. matrix-exponential exact

References:
 - MMS overview and order verification: Roy (2004), IJNMF. 
 - FTCS stability (diffusion): dt <= 1/(2 D (1/dx^2 + 1/dy^2)) in 2D (equal dx=dy → dt <= dx^2/(4D)).

This script prints a status block and writes plots + vv_report.json into --outdir.
"""
import os, json, math, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# ---- compatibility for numerical integration naming across numpy versions ----
try:
    _trap = np.trapezoid  # numpy >= 1.20
except AttributeError:
    _trap = np.trapz      # fallback

# ------------------------------ utilities ------------------------------------
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def l2_error(U, V, dx, dy):
    # discrete L2 norm approximation of continuous L2
    return float(np.sqrt(np.sum((U - V)**2) * dx * dy))

def integrate2d(A, x, y):
    # composite trapezoidal in 2D (axis order explicit)
    return float(_trap(_trap(A, y, axis=1), x, axis=0))

# ------------------------------ MMS module -----------------------------------
def mms_convergence(outdir):
    """
    PDE: u_t = D Δu - k u + f
    Manufactured solution: u = sin(pi x) sin(pi y) e^{-t}
    Forcing: f = (-1 + 2*pi^2*D + k) * u
    BC: Dirichlet u=0 on ∂Ω (true for sin on [0,1])
    """
    D = 1e-3
    k = 1e-3
    T_end = 5e-3    # short horizon, smooth solution

    def u_exact(X, Y, t):
        return np.sin(np.pi*X)*np.sin(np.pi*Y)*np.exp(-t)

    const = -1.0 + 2.0*(np.pi**2)*D + k
    def forcing(Xi, Yi, t):
        return const * u_exact(Xi, Yi, t)

    def run(nx, ny, dt_fixed=None):
        Lx = Ly = 1.0
        x = np.linspace(0.0, Lx, nx)
        y = np.linspace(0.0, Ly, ny)
        dx = Lx/(nx-1); dy = Ly/(ny-1)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Choose dt
        if dt_fixed is not None:
            dt = float(dt_fixed)
            # keep stability margin for explicit FTCS in 2D:
            dt_stab = 1.0/(2.0*D*(1.0/dx**2 + 1.0/dy**2))
            if dt > 0.9*dt_stab:
                dt = 0.9*dt_stab
        else:
            # stable, but NOT used for space study (we decouple dt from h there)
            dt = 0.2*min(dx,dy)**2 / D

        steps = int(np.ceil(T_end/dt))
        dt = T_end/steps  # land exactly on T_end

        # initialize
        C = u_exact(X, Y, 0.0)

        # explicit FTCS
        for n in range(steps):
            t = n*dt
            lap = ((C[2:,1:-1] - 2*C[1:-1,1:-1] + C[:-2,1:-1])/(dy*dy) +
                   (C[1:-1,2:]  - 2*C[1:-1,1:-1] + C[1:-1,:-2])/(dx*dx))
            F = forcing(X[1:-1,1:-1], Y[1:-1,1:-1], t)
            Cn = C.copy()
            Cn[1:-1,1:-1] = C[1:-1,1:-1] + dt*(D*lap - k*C[1:-1,1:-1] + F)
            # Dirichlet (matches manufactured solution on boundary)
            Cn[0,:]=0; Cn[-1,:]=0; Cn[:,0]=0; Cn[:,-1]=0
            C = Cn

        err = l2_error(C, u_exact(X, Y, T_end), dx, dy)
        return err, dx, dy

    # ---- SPACE ORDER: fixed tiny dt across grids (decouple time error) ----
    # pick dt many orders smaller than stability limit at the FINEST grid
    nx_list = (33, 49, 65)
    # compute a conservative fixed dt from the *finest* grid stability limit
    dx_fine = 1.0/(nx_list[-1]-1)
    dt_stab_fine = 1.0/(2.0*D*(1.0/dx_fine**2 + 1.0/dx_fine**2))
    dt_fixed = 1e-2 * dt_stab_fine  # plenty of headroom
    errs_s, hs = [], []
    for nx in nx_list:
        err, dx, dy = run(nx, nx, dt_fixed=dt_fixed)
        errs_s.append(err); hs.append(max(dx,dy))
    # estimate p from the last two (finest) levels to avoid coarse-level bias
    p_space = np.log(errs_s[-2]/errs_s[-1]) / np.log(hs[-2]/hs[-1])

    # ---- TIME ORDER: fixed grid, refine dt ----
    nx = ny = 65
    dt1, dt2, dt3 = 1.0e-4, 5.0e-5, 2.5e-5
    err1, *_ = run(nx, ny, dt_fixed=dt1)
    err2, *_ = run(nx, ny, dt_fixed=dt2)
    err3, *_ = run(nx, ny, dt_fixed=dt3)
    p_time = np.log(err2/err3) / np.log(dt2/dt3)

    # plots (log-log)
    ensure_outdir(outdir)
    plt.figure()
    plt.loglog(hs, errs_s, marker='o')
    plt.xlabel("h"); plt.ylabel("L2 error")
    plt.title(f"MMS space convergence (p≈{p_space:.2f})")
    plt.savefig(os.path.join(outdir, "mms_convergence_space.png"), bbox_inches="tight"); plt.close()

    dts = [dt1, dt2, dt3]; errs_t = [err1, err2, err3]
    plt.figure()
    plt.loglog(dts, errs_t, marker='o')
    plt.xlabel("dt"); plt.ylabel("L2 error")
    plt.title(f"MMS time convergence (p≈{p_time:.2f})")
    plt.savefig(os.path.join(outdir, "mms_convergence_time.png"), bbox_inches="tight"); plt.close()

    return {
        "space_h": hs, "space_errs": errs_s, "space_order_est": float(p_space),
        "time_dts": dts, "time_errs": errs_t, "time_order_est": float(p_time)
    }

# ------------------------- Gaussian + mass bound ------------------------------
def gaussian_sanity(outdir):
    D = 1e-3
    k = 1e-3
    Lx = Ly = 1.0
    nx = ny = 129
    x = np.linspace(0.0, Lx, nx); dx = Lx/(nx-1)
    y = np.linspace(0.0, Ly, ny); dy = Ly/(ny-1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # initial Gaussian (centered)
    sig2 = 0.01
    C0 = np.exp(-((X-0.5)**2 + (Y-0.5)**2)/sig2)
    C = C0.copy()

    # source-free RHS (Dirichlet boundaries)
    def step(C, dt):
        lap = ((C[2:,1:-1] - 2*C[1:-1,1:-1] + C[:-2,1:-1])/(dy*dy) +
               (C[1:-1,2:]  - 2*C[1:-1,1:-1] + C[1:-1,:-2])/(dx*dx))
        Cn = C.copy()
        Cn[1:-1,1:-1] = C[1:-1,1:-1] + dt*(D*lap - k*C[1:-1,1:-1])
        Cn[0,:]=0; Cn[-1,:]=0; Cn[:,0]=0; Cn[:,-1]=0
        return Cn

    # stable dt (2D FTCS) with safety factor
    dt_stab = 1.0/(2.0*D*(1.0/dx**2 + 1.0/dy**2))
    dt = 0.2*dt_stab

    times = []
    mass = []
    t = 0.0
    steps = 400
    for n in range(steps):
        C = step(C, dt)
        t += dt
        times.append(t)
        mass.append(integrate2d(C, x, y))

    # analytic **upper bound** on mass with Dirichlet edges:
    # M(t) <= M(0) * e^{-k t}
    M0 = integrate2d(C0, x, y)
    M_upper = M0 * np.exp(-k*np.array(times))
    mass = np.array(mass)

    # plots
    plt.figure()
    plt.plot(times, mass, label="numeric mass")
    plt.plot(times, M_upper, '--', label=r"$M_0 e^{-k t}$ (upper bound)")
    plt.xlabel("t (s)"); plt.ylabel("mass")
    plt.title("Mass vs analytic decay bound")
    plt.legend()
    plt.savefig(os.path.join(outdir, "mass_positivity_checks.png"), bbox_inches="tight"); plt.close()

    # 1D slices (visual sanity)
    for t_show, n_step in [(0.125, int(0.125/dt)), (0.25, int(0.25/dt)), (0.5, int(0.5/dt))]:
        # march a fresh copy to the required time
        Ctmp = C0.copy(); tt=0.0
        for _ in range(n_step):
            Ctmp = step(Ctmp, dt); tt += dt
        cx = Ctmp[:, ny//2]
        plt.figure()
        plt.plot(x, cx, label=f"num t={tt:.3f}")
        plt.xlabel("x"); plt.ylabel("C"); plt.title("Gaussian slice")
        plt.legend()
        fname = f"gaussian_slice_t{int(round(tt*1000)):04d}ms.png"
        plt.savefig(os.path.join(outdir, fname), bbox_inches="tight"); plt.close()

    # pass/fail: mass never exceeds bound by > tol_rel * M0
    tol_rel = 5e-3
    over = float(np.max(mass - M_upper))
    ok = (over <= tol_rel*M0)
    return {"ok": ok, "max_over": over, "M0": M0, "tol_rel": tol_rel}

# ---------------------- PK exact vs numeric overlay ---------------------------
def pk_verification(outdir):
    # two-compartment linear PK with a finite infusion then shutoff
    k12 = 0.15; k21 = 0.05; k10 = 0.08
    rate = 0.01; t_stop = 11.0*3600.0

    A = np.array([[-(k12+k10), k21],
                  [k12,        -k21]], dtype=float)
    b = np.array([1.0, 0.0], dtype=float)

    def u(t):  # input (infusion) in "b * u(t)"
        return rate if t < t_stop else 0.0

    def rhs(t, y):
        return A @ y + b * u(t)

    t_eval = np.linspace(0.0, 24*3600.0, 1200)
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0=[0.0, 0.0], t_eval=t_eval,
                    method="RK45", rtol=1e-9, atol=1e-12)
    Y = sol.y.T

    # exact via variation of constants (piecewise-constant input)
    Y_exact = np.zeros_like(Y)
    Ainv = np.linalg.inv(A)
    for i, t in enumerate(t_eval):
        if t <= t_stop:
            Y_exact[i] = (expm(A*t) @ np.zeros(2) +
                          Ainv @ (expm(A*t) - np.eye(2)) @ (b*rate))
        else:
            Y_at_stop = expm(A*t_stop) @ np.zeros(2) + Ainv @ (expm(A*t_stop) - np.eye(2)) @ (b*rate)
            Y_exact[i] = expm(A*(t - t_stop)) @ Y_at_stop

    max_abs = float(np.max(np.abs(Y - Y_exact)))
    mean_abs = float(np.mean(np.abs(Y - Y_exact)))

    # plot
    plt.figure()
    plt.plot(t_eval/3600.0, Y[:,0], label="Cb (num)")
    plt.plot(t_eval/3600.0, Y_exact[:,0], '--', label="Cb (exact)")
    plt.plot(t_eval/3600.0, Y[:,1], label="Cbrain (num)")
    plt.plot(t_eval/3600.0, Y_exact[:,1], '--', label="Cbrain (exact)")
    plt.xlabel("Time (h)"); plt.ylabel("Concentration")
    plt.title("PK: numeric vs. matrix-exponential exact")
    plt.legend()
    plt.savefig(os.path.join(outdir, "pk_closed_form_overlay.png"), bbox_inches="tight"); plt.close()

    return {"max_abs_err": max_abs, "mean_abs_err": mean_abs}

# --------------------------------- main ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()
    ensure_outdir(args.outdir)

    # run
    mms   = mms_convergence(args.outdir)
    gauss = gaussian_sanity(args.outdir)
    pk    = pk_verification(args.outdir)

    # criteria (tight, but achievable)
    crit = {
        "mms_space_order_min": 1.8,   # ~2nd order
        "mms_time_order_min":  0.9,   # ~1st order
        "pk_max_abs_err_max":  1e-6,
        "mass_over_rel_max":   5e-3   # ≤0.5% above bound
    }
    status = {
        "mms_ok": (mms["space_order_est"] >= crit["mms_space_order_min"] and
                   mms["time_order_est"]  >= crit["mms_time_order_min"]),
        "gaussian_ok": (gauss["ok"] and gauss["max_over"] <= crit["mass_over_rel_max"]*gauss["M0"]),
        "pk_ok": (pk["max_abs_err"] <= crit["pk_max_abs_err_max"])
    }

    report = {
        "mms": mms,
        "gaussian": gauss,
        "pk_matrix_exponential": pk,
        "criteria": crit,
        "status": status
    }
    with open(os.path.join(args.outdir, "vv_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("V&V done →", args.outdir)
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
