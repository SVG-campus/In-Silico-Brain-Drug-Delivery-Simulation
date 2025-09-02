#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V&V runner: MMS (space & time order), Gaussian sanity (mass bound & positivity),
and PK 2-compartment numeric vs. matrix-exponential "exact".
Outputs a vv_report.json and PNGs into --outdir.
"""

import os, json, math, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# -------------------------
# helpers
# -------------------------
def l2_error(U, V, dx, dy):
    return float(np.sqrt(np.sum((U - V)**2) * dx * dy))

def new_outdir(path):
    os.makedirs(path, exist_ok=True)
    return path

# -------------------------
# 1) MMS: u_t = D Δu - k u + f
# Manufactured solution: u = sin(pi x) sin(pi y) e^{-t}
# Forcing: f = (-1 + 2 D pi^2 + k) u
# -------------------------
def mms_convergence(outdir):
    D = 1e-3
    k = 1e-3
    T = 5e-3    # short final time (parabolic problem)

    def u_exact(X, Y, t):
        return np.sin(np.pi*X)*np.sin(np.pi*Y)*np.exp(-t)

    const = -1.0 + 2.0*D*(np.pi**2) + k
    def forcing(Xi, Yi, t):
        return const * np.sin(np.pi*Xi)*np.sin(np.pi*Yi)*np.exp(-t)

    def run_grid(nx, ny, dt=None):
        Lx = Ly = 1.0
        x = np.linspace(0.0, Lx, nx)
        y = np.linspace(0.0, Ly, ny)
        dx = Lx/(nx-1); dy = Ly/(ny-1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Xi, Yi = X[1:-1,1:-1], Y[1:-1,1:-1]

        # FTCS stability limit (2-D, equal dx=dy reduces to dt <= dx^2/(4D))
        dt_max = (dx*dx*dy*dy) / (2.0*D*((dx*dx)+(dy*dy)))
        if dt is None:
            # For spatial order, use dt << dt_max (make temporal error negligible)
            dt = min(1e-6, 0.25*dt_max)
        else:
            # Always respect stability
            dt = min(dt, 0.9*dt_max)

        steps = int(np.ceil(T/dt))
        dt = T/steps  # land exactly at T

        C = u_exact(X, Y, 0.0)
        for n in range(steps):
            t = n*dt
            lap = ((C[2:,1:-1] - 2*C[1:-1,1:-1] + C[:-2,1:-1])/(dy*dy) +
                   (C[1:-1,2:] - 2*C[1:-1,1:-1] + C[1:-1,:-2])/(dx*dx))
            F = forcing(Xi, Yi, t)
            Cn = C.copy()
            Cn[1:-1,1:-1] = C[1:-1,1:-1] + dt*(D*lap - k*C[1:-1,1:-1] + F)
            # Dirichlet-0 (matches manufactured solution on boundary)
            Cn[0,:]=0; Cn[-1,:]=0; Cn[:,0]=0; Cn[:,-1]=0
            C = Cn

        err = l2_error(C, u_exact(X, Y, T), dx, dy)
        return err, dx, dy

    # ---- spatial order: vary h, keep dt extremely small and independent of h
    errs_s, hs = [], []
    for nx in (33, 49, 65):  # coarse -> fine
        err, dx, dy = run_grid(nx, nx, dt=None)  # dt chosen tiny internally
        errs_s.append(err); hs.append(max(dx,dy))

    # p ≈ log(E(h2)/E(h3)) / log(h2/h3) (using last two points)
    p_space = float(np.log(errs_s[-2]/errs_s[-1]) / np.log(hs[-2]/hs[-1]))

    # ---- temporal order: fix a fine grid, vary dt
    nx = ny = 129
    # ensure a fine grid so spatial error is negligible across dt sweep
    # choose a set of stable dt values
    err1, dx, dy = run_grid(nx, ny, dt=1.0e-4)
    err2, *_      = run_grid(nx, ny, dt=5.0e-5)
    err3, *_      = run_grid(nx, ny, dt=2.5e-5)
    dts = [1.0e-4, 5.0e-5, 2.5e-5]
    errs_t = [err1, err2, err3]
    p_time = float(np.log(errs_t[-2]/errs_t[-1]) / np.log(dts[-2]/dts[-1]))

    # ---- plots
    plt.figure()
    plt.loglog(hs, errs_s, marker='o')
    plt.xlabel("h"); plt.ylabel("L2 error")
    plt.title(f"MMS space convergence (p≈{p_space:.2f})")
    plt.savefig(os.path.join(outdir, "mms_convergence_space.png"), bbox_inches="tight"); plt.close()

    plt.figure()
    plt.loglog(dts, errs_t, marker='o')
    plt.xlabel("dt"); plt.ylabel("L2 error")
    plt.title(f"MMS time convergence (p≈{p_time:.2f})")
    plt.savefig(os.path.join(outdir, "mms_convergence_time.png"), bbox_inches="tight"); plt.close()

    return {
        "space_order_est": float(p_space),
        "time_order_est": float(p_time),
        "space_errs": [float(e) for e in errs_s],
        "space_hs": [float(h) for h in hs],
        "time_errs": [float(e) for e in errs_t],
        "time_dts": dts,
    }

# -------------------------
# 2) Gaussian sanity (positivity + mass upper bound with decay)
# -------------------------
def gaussian_sanity(outdir):
    D = 1e-3
    k = 1e-3
    Lx = Ly = 1.0
    nx = ny = 129
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    dx = Lx/(nx-1); dy = Ly/(ny-1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # centered Gaussian
    sigma2 = 0.0025
    C = np.exp(-((X-0.5)**2 + (Y-0.5)**2)/sigma2)

    # FTCS stability: dt <= dx^2/(4D) (for dx=dy)
    dt_max = (dx*dx*dy*dy) / (2.0*D*((dx*dx)+(dy*dy)))
    dt = 0.25*dt_max
    T = 0.5
    steps = int(np.ceil(T/dt)); dt = T/steps

    def total_mass(A):
        # 2D trapezoid integral
        return float(np.trapz(np.trapz(A, y, axis=1), x, axis=0))

    times = [0.0]
    mass = [total_mass(C)]
    t_marks = [0.125, 0.250, 0.500]
    slices = {}

    for n in range(steps):
        lap = ((C[2:,1:-1] - 2*C[1:-1,1:-1] + C[:-2,1:-1])/(dy*dy) +
               (C[1:-1,2:] - 2*C[1:-1,1:-1] + C[1:-1,:-2])/(dx*dx))
        Cn = C.copy()
        Cn[1:-1,1:-1] = C[1:-1,1:-1] + dt*(D*lap - k*C[1:-1,1:-1])
        # homogeneous Dirichlet
        Cn[0,:]=0; Cn[-1,:]=0; Cn[:,0]=0; Cn[:,-1]=0
        C = Cn

        t = (n+1)*dt
        times.append(t)
        mass.append(total_mass(C))
        for tm in t_marks:
            if (abs(t - tm) <= 0.5*dt) and (tm not in slices):
                plt.figure()
                i_mid = nx//2
                plt.plot(x, C[i_mid,:], label=f"num t={tm:.3f}")
                plt.xlabel("x"); plt.ylabel("C")
                plt.title("Gaussian slice")
                plt.legend()
                fn = f"gaussian_slice_t{int(round(tm*1000)):04d}ms.png"
                plt.savefig(os.path.join(outdir, fn), bbox_inches="tight"); plt.close()
                slices[tm] = fn

    times_arr = np.array(times)
    M0 = mass[0]
    M_bound = M0*np.exp(-k*times_arr)  # with Dirichlet boundaries, mass ≤ this bound
    mass_ok = np.all(np.array(mass) <= M_bound*(1.0+1e-3))  # allow tiny num. slack
    min_val = float(np.min(C))

    plt.figure()
    plt.plot(times_arr, mass, label="numeric mass")
    plt.plot(times_arr, M_bound, "--", label="M0 e^{-k t} (upper bound)")
    plt.xlabel("t (s)"); plt.ylabel("mass"); plt.title("Mass vs analytic decay bound")
    plt.legend()
    plt.savefig(os.path.join(outdir, "mass_positivity_checks.png"), bbox_inches="tight"); plt.close()

    return {
        "min_value": min_val,
        "mass_ok": bool(mass_ok),
        "mass0": float(M0),
        "mass_final": float(mass[-1]),
        "mass_max_over_bound_ratio": float(np.max(np.array(mass)/M_bound)),
    }

# -------------------------
# 3) PK verification (2-compartment; numeric vs closed-form w/ matrix exponential)
# -------------------------
def pk_verification(outdir):
    k12 = 0.15; k21 = 0.05; k10 = 0.08
    rate = 0.012; t_stop = 11.0*3600.0
    def u(t): return rate*(t <= t_stop)

    A = np.array([[-(k12+k10), k21],
                  [k12,        -k21]], dtype=float)
    b = np.array([1.0, 0.0], dtype=float)

    # numeric
    t_eval = np.linspace(0.0, 24.0*3600.0, 800)
    def rhs(t,y): return A@y + b*u(t)
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0=[0.0,0.0], t_eval=t_eval, rtol=1e-9, atol=1e-12)
    Y = sol.y.T

    # exact via variation of constants
    Y_exact = np.zeros_like(Y)
    I = np.eye(2)
    Ainv = np.linalg.inv(A)
    for i, t in enumerate(t_eval):
        if t <= t_stop:
            Y_exact[i] = (expm(A*t) @ np.zeros(2) + Ainv @ ((expm(A*t)-I) @ (b*rate)))
        else:
            Yt = (expm(A*t_stop) @ np.zeros(2) + Ainv @ ((expm(A*t_stop)-I) @ (b*rate)))
            Y_exact[i] = expm(A*(t-t_stop)) @ Yt

    max_abs = float(np.max(np.abs(Y - Y_exact)))
    mean_abs = float(np.mean(np.abs(Y - Y_exact)))

    # plot
    plt.figure()
    plt.plot(t_eval/3600.0, Y[:,0], label="Cb (num)")
    plt.plot(t_eval/3600.0, Y_exact[:,0], "--", label="Cb (exact)")
    plt.plot(t_eval/3600.0, Y[:,1], label="Cbrain (num)")
    plt.plot(t_eval/3600.0, Y_exact[:,1], "--", label="Cbrain (exact)")
    plt.xlabel("Time (h)"); plt.ylabel("Concentration")
    plt.title("PK: numeric vs. matrix-exponential exact")
    plt.legend()
    plt.savefig(os.path.join(outdir, "pk_closed_form_overlay.png"), bbox_inches="tight"); plt.close()

    return {"max_abs_err": max_abs, "mean_abs_err": mean_abs}

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = new_outdir(args.outdir)

    mms   = mms_convergence(outdir)
    gauss = gaussian_sanity(outdir)
    pk    = pk_verification(outdir)

    criteria = {
        "mms_space_order_min": 1.8,  # ~2 expected (5-pt Laplacian)
        "mms_time_order_min":  0.9,  # ~1 expected for explicit Euler/FTCS in time
        "gaussian_min_value": -1e-12,
        "gaussian_mass_bound_ratio_max": 1.01,
        "pk_max_abs_err_max": 1e-9
    }

    status = {
        "mms_ok": (mms["space_order_est"] >= criteria["mms_space_order_min"] and
                   mms["time_order_est"]  >= criteria["mms_time_order_min"]),
        "gaussian_ok": (gauss["min_value"] >= criteria["gaussian_min_value"] and
                        gauss["mass_max_over_bound_ratio"] <= criteria["gaussian_mass_bound_ratio_max"]),
        "pk_ok": (pk["max_abs_err"] <= criteria["pk_max_abs_err_max"])
    }

    report = {"mms": mms, "gaussian": gauss, "pk": pk, "criteria": criteria, "status": status}
    with open(os.path.join(outdir, "vv_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("V&V done →", outdir)
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
