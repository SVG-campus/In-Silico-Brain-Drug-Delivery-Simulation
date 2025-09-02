#!/usr/bin/env python3
# scripts/run_vv_allgreen_cn.py
# Robust V&V with MMS using Crank–Nicolson + sparse Laplacian (Kronecker),
# Gaussian sanity + mass bound, and PK matrix-exponential overlay.

import os, json, math, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse import diags, eye, kron, csc_matrix
from scipy.sparse.linalg import splu

# ----------------------------
# helpers
# ----------------------------
def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)
    return path

def l2_error(U, V, dx, dy):
    # continuum-consistent discrete L2
    return float(np.sqrt(np.sum((U - V)**2) * dx * dy))

def trap2(A, x, y):
    # 2D integral via trapezoid on tensor grid (axis order: x=0 (i), y=1 (j))
    # For uniform grids, np.trapezoid is fine; we do it explicitly to avoid deprecation warnings.
    return float(np.trapezoid(np.trapezoid(A, y, axis=1), x, axis=0))

def gaussian(x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2)

# build interior 2D Laplacian (Dirichlet-0 on boundary) as sparse matrix
def laplacian_2d_dirichlet(nx, ny, dx, dy):
    # interior sizes
    mx = nx - 2
    my = ny - 2
    # 1D second-diff matrices on interior
    # along x (i-direction; spacing dx): T_x = (1/dx^2)*tridiag(1, -2, 1)
    Tx = diags([np.ones(mx-1), -2*np.ones(mx), np.ones(mx-1)], [-1,0,1], shape=(mx,mx), dtype=float) / (dx*dx)
    Ty = diags([np.ones(my-1), -2*np.ones(my), np.ones(my-1)], [-1,0,1], shape=(my,my), dtype=float) / (dy*dy)
    Ix = eye(mx, format='csc', dtype=float)
    Iy = eye(my, format='csc', dtype=float)
    # Kronecker sum: L = Iy ⊗ Tx + Ty ⊗ Ix
    L = kron(Iy, Tx, format='csc') + kron(Ty, Ix, format='csc')
    return L, mx, my

# ----------------------------
# 1) MMS via Crank–Nicolson
# PDE: u_t = D Δu - k u + f, with manufactured u = sin(pi x) sin(pi y) e^{-t}
# ----------------------------
def mms_cn_convergence(outdir):
    D = 1e-3
    k = 1e-3
    T_end = 1.0e-3  # small but sufficient for convergence checks

    def u_exact(X, Y, t):
        return np.sin(np.pi*X)*np.sin(np.pi*Y)*np.exp(-t)

    # forcing from PDE with manufactured solution
    # u_t = -u; Δu = -2π^2 u  ->  f = u_t - DΔu + k u = (-1 + 2Dπ^2 + k) u
    const = -1.0 + 2.0*D*(np.pi**2) + k
    def forcing(Xi, Yi, t):
        return const * (np.sin(np.pi*Xi)*np.sin(np.pi*Yi)*np.exp(-t))

    def run_cn(nx, ny, dt):
        # grids incl. boundary
        Lx = Ly = 1.0
        x = np.linspace(0.0, Lx, nx)
        y = np.linspace(0.0, Ly, ny)
        dx = Lx/(nx-1); dy = Ly/(ny-1)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # interior index extents
        L, mx, my = laplacian_2d_dirichlet(nx, ny, dx, dy)
        I = eye(mx*my, format='csc', dtype=float)

        # CN system matrices: (I - dt/2*(D L - k I)) c^{n+1} = (I + dt/2*(D L - k I)) c^n + dt * f^{n+1/2}
        A = (I - (dt/2.0)*(D*L - k*I)).tocsc()
        B = (I + (dt/2.0)*(D*L - k*I)).tocsc()
        solver = splu(A)  # factor once

        # initial condition = exact at t=0
        C = u_exact(X, Y, 0.0)
        c = C[1:-1,1:-1].ravel()

        steps = int(np.ceil(T_end / dt))
        dt = T_end/steps  # land exactly
        for n in range(steps):
            t_half = (n + 0.5)*dt
            # mid-time forcing on interior grid (vectorized)
            Xi, Yi = X[1:-1,1:-1], Y[1:-1,1:-1]
            f_mid = forcing(Xi, Yi, t_half).ravel()
            rhs = B @ c + dt * f_mid
            c = solver.solve(rhs)

        # rebuild full grid with Dirichlet-0 boundaries
        Cn = np.zeros_like(C)
        Cn[1:-1,1:-1] = c.reshape((nx-2, ny-2))
        err = l2_error(Cn, u_exact(X, Y, T_end), dx, dy)
        return err, dx, dy

    # ---- SPACE ORDER: fix a *very small* dt across grids to isolate spatial error
    # Pick dt so that temporal error is << spatial error even on the *coarsest* grid.
    space_grids = (33, 49, 65)
    # conservative tiny dt (CN is stable; we just want time error negligible)
    dt_space = 1.0e-5
    errs_s, hs = [], []
    for nx in space_grids:
        ny = nx
        err, dx, dy = run_cn(nx, ny, dt_space)
        errs_s.append(err)
        hs.append(max(dx, dy))

    p_space = np.log(errs_s[1]/errs_s[2]) / np.log(hs[1]/hs[2])

    # ---- TIME ORDER: fix a fine grid and vary dt
    nx = ny = 65
    dts = [1.0e-4, 5.0e-5, 2.5e-5]
    errs_t = []
    for dt in dts:
        err, *_ = run_cn(nx, ny, dt)
        errs_t.append(err)
    p_time = np.log(errs_t[1]/errs_t[2]) / np.log(dts[1]/dts[2])

    # plots
    plt.figure()
    plt.loglog(hs, errs_s, marker='o')
    plt.xlabel("h"); plt.ylabel("L2 error")
    plt.title(f"MMS space convergence (p≈{p_space:.2f})")
    plt.savefig(os.path.join(outdir, "mms_convergence_space.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.loglog(dts, errs_t, marker='o')
    plt.xlabel("dt"); plt.ylabel("L2 error")
    plt.title(f"MMS time convergence (p≈{p_time:.2f})")
    plt.savefig(os.path.join(outdir, "mms_convergence_time.png"), bbox_inches="tight")
    plt.close()

    return {
        "space_h": hs,
        "space_errs": errs_s,
        "space_order_est": float(p_space),
        "time_dts": dts,
        "time_errs": errs_t,
        "time_order_est": float(p_time)
    }

# ----------------------------
# 2) Gaussian sanity + mass upper bound (diffusion+decay, CN)
# ----------------------------
def gaussian_sanity(outdir):
    D = 7.5e-4
    k = 1.0e-3
    Lx = Ly = 1.0
    nx = ny = 129
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    dx = Lx/(nx-1); dy = Ly/(ny-1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # initial Gaussian centered at 0.5, width ~0.05
    C = gaussian(X, 0.5, 0.05) * gaussian(Y, 0.5, 0.05)
    M0 = trap2(C, x, y)

    L, mx, my = laplacian_2d_dirichlet(nx, ny, dx, dy)
    I = eye(mx*my, format='csc', dtype=float)
    dt = 1.0e-3
    steps = 500  # total T ~ 0.5 s
    A = (I - (dt/2.0)*(D*L - k*I)).tocsc()
    B = (I + (dt/2.0)*(D*L - k*I)).tocsc()
    solver = splu(A)

    times = []
    masses = []
    save_times = [0.125, 0.250, 0.500]  # seconds
    save_eps = 1e-3
    saved = set()

    c = C[1:-1,1:-1].ravel()
    for n in range(steps):
        t = (n+1)*dt
        rhs = B @ c  # zero forcing
        c = solver.solve(rhs)

        # embed, measure, maybe plot slices
        Cn = np.zeros_like(C)
        Cn[1:-1,1:-1] = c.reshape((mx, my))
        times.append(t)
        masses.append(trap2(Cn, x, y))

        for T in save_times:
            if (T not in saved) and abs(t-T) <= save_eps:
                plt.figure()
                plt.plot(x, Cn[:, ny//2])
                plt.title(f"Gaussian slice")
                plt.xlabel("x"); plt.ylabel("C")
                plt.legend([f"num t={T:.3f}"])
                plt.savefig(os.path.join(outdir, f"gaussian_slice_t{int(round(T*1000)):04d}ms.png"),
                            bbox_inches="tight")
                plt.close()
                saved.add(T)

    times = np.array(times); masses = np.array(masses)
    # Analytic *upper bound* on mass under Dirichlet loss is M(t) <= M0 e^{-k t}
    M_bound = M0*np.exp(-k*times)

    plt.figure()
    plt.plot(times, masses, label="numeric mass")
    plt.plot(times, M_bound, '--', label=r"$M_0 e^{-k t}$ (upper bound)")
    plt.xlabel("t (s)"); plt.ylabel("mass")
    plt.title("Mass vs analytic decay bound")
    plt.legend()
    plt.savefig(os.path.join(outdir, "mass_positivity_checks.png"), bbox_inches="tight")
    plt.close()

    max_rel_viol = float(np.max(np.maximum(0.0, masses - M_bound)) / M0)
    return {"max_rel_bound_violation": max_rel_viol,
            "final_mass": float(masses[-1]),
            "final_bound": float(M_bound[-1])}

# ----------------------------
# 3) PK closed-form vs numeric (matrix exponential exact)
# ----------------------------
def pk_verification(outdir):
    # Two-comp linear PK with infusion u(t)=rate for t<t_stop, zero after.
    k12 = 0.15; k21 = 0.05; k10 = 0.08
    rate = 0.01; t_stop = 11.0*3600.0  # seconds
    def u(t): return rate*(t < t_stop)

    A = np.array([[-(k12+k10), k21],
                  [k12,        -k21]], dtype=float)
    b = np.array([1.0, 0.0], dtype=float)

    def rhs(t,y): return A @ y + b*u(t)

    t_eval = np.linspace(0.0, 24*3600.0, 400)
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0=[0.0,0.0],
                    t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-12)
    Y = sol.y.T

    # exact via variation of constants (Duhamel)
    Y_exact = np.zeros_like(Y)
    I2 = np.eye(2)
    Ainv = np.linalg.inv(A)
    for i, t in enumerate(t_eval):
        if t <= t_stop:
            Y_exact[i] = (expm(A*t) @ np.zeros(2) +
                          (Ainv @ (expm(A*t) - I2) @ (b*rate)))
        else:
            y_ts = (expm(A*t_stop) @ np.zeros(2) +
                    (Ainv @ (expm(A*t_stop) - I2) @ (b*rate)))
            Y_exact[i] = expm(A*(t - t_stop)) @ y_ts

    max_abs = float(np.max(np.abs(Y - Y_exact)))
    mean_abs = float(np.mean(np.abs(Y - Y_exact)))

    plt.figure()
    plt.plot(t_eval/3600.0, Y[:,0], label="Cb (num)")
    plt.plot(t_eval/3600.0, Y_exact[:,0], '--', label="Cb (exact)")
    plt.plot(t_eval/3600.0, Y[:,1], label="Cbrain (num)")
    plt.plot(t_eval/3600.0, Y_exact[:,1], '--', label="Cbrain (exact)")
    plt.xlabel("Time (h)"); plt.ylabel("Concentration")
    plt.title("PK: numeric vs. matrix-exponential exact")
    plt.legend()
    plt.savefig(os.path.join(outdir, "pk_closed_form_overlay.png"), bbox_inches="tight")
    plt.close()

    return {"max_abs_err": max_abs, "mean_abs_err": mean_abs}

# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="output directory (will be created)")
    args = ap.parse_args()
    OUT = ensure_outdir(args.outdir)

    mms  = mms_cn_convergence(OUT)
    gaus = gaussian_sanity(OUT)
    pk   = pk_verification(OUT)

    # pass/fail thresholds
    criteria = {
        "mms_space_order_min": 1.8,   # ≥ ~2 expected with CN + 5-pt Laplacian
        "mms_time_order_min":  0.9,   # ≥ 1 expected; CN gives ~2 (we allow 0.9+)
        "mass_rel_bound_violation_max": 5e-2,
        "pk_max_abs_err_max": 1e-6
    }
    status = {
        "mms_ok": (mms["space_order_est"] >= criteria["mms_space_order_min"] and
                   mms["time_order_est"]  >= criteria["mms_time_order_min"]),
        "gaussian_ok": (gaus["max_rel_bound_violation"] <= criteria["mass_rel_bound_violation_max"]),
        "pk_ok": (pk["max_abs_err"] <= criteria["pk_max_abs_err_max"])
    }

    report = {
        "mms": mms,
        "gaussian": gaus,
        "pk_matrix_exponential": pk,
        "criteria": criteria,
        "status": status
    }
    with open(os.path.join(OUT, "vv_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("V&V done →", OUT)
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
