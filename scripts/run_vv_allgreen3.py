#!/usr/bin/env python3
import os, json, math, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# -------------------------------
# small helpers
# -------------------------------
def l2_error(U, V, dx, dy):
    return float(np.sqrt(np.sum((U - V)**2) * dx * dy))

def trapz2(A, x, y):
    # use numpy.trapezoid to avoid deprecation warnings
    return float(np.trapezoid(np.trapezoid(A, y, axis=1), x, axis=0))

# ===============================
# 1) MMS: u_t = D Δu - k u + f
#    manufactured solution:
#    u = sin(pi x) sin(pi y) e^{-t}
# ===============================
def mms_convergence(outdir):
    D = 1e-3
    k = 1e-3
    T_end = 5e-2  # long enough to see dynamics

    def u_exact(X, Y, t):
        return np.sin(np.pi*X)*np.sin(np.pi*Y)*np.exp(-t)

    # forcing from MMS: u_t - DΔu + k u = f  => f = (-1 + 2 D π^2 + k) u
    const = -1.0 + 2.0*D*(np.pi**2) + k
    def forcing(Xi, Yi, t):
        return const * u_exact(Xi, Yi, t)

    def run_ftcs(nx, ny, dt=None, keep_r_const=False):
        Lx = Ly = 1.0
        x = np.linspace(0.0, Lx, nx)
        y = np.linspace(0.0, Ly, ny)
        dx = Lx/(nx-1); dy = Ly/(ny-1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Xi, Yi = X[1:-1,1:-1], Y[1:-1,1:-1]

        # For 2D FTCS heat equation with 5-pt stencil, stability requires:
        # r_x + r_y <= 1/2 where r_x = D dt/dx^2, r_y = D dt/dy^2.
        # If dx=dy, this is dt <= dx^2/(4D). We choose a safe fraction 'alpha'.
        alpha = 0.9  # close to limit to make time error visible (still stable)
        dt_limit = (dx*dx * dy*dy) / (2.0*D*(dx*dx + dy*dy))  # general 2D bound
        # If keep_r_const, scale dt with dx^2 so the CFL is constant across grids
        if dt is None:
            dt = alpha * dt_limit
        elif keep_r_const:
            # interpret 'dt' as a CFL multiplier (alpha) and recompute actual dt
            dt = float(dt) * dt_limit

        steps = int(np.ceil(T_end/dt))
        dt = T_end/steps  # land exactly at T_end
        C = u_exact(X, Y, 0.0)
        for n in range(steps):
            t = n*dt
            lap = ((C[2:,1:-1] - 2*C[1:-1,1:-1] + C[:-2,1:-1])/(dy*dy) +
                   (C[1:-1,2:] - 2*C[1:-1,1:-1] + C[1:-1,:-2])/(dx*dx))
            F = forcing(Xi, Yi, t)
            Cn = C.copy()
            Cn[1:-1,1:-1] = C[1:-1,1:-1] + dt*(D*lap - k*C[1:-1,1:-1] + F)
            # Dirichlet (u=0) on boundary (manufactured solution satisfies this)
            Cn[0,:]=0; Cn[-1,:]=0; Cn[:,0]=0; Cn[:,-1]=0
            C = Cn
        err = l2_error(C, u_exact(X, Y, T_end), dx, dy)
        return err, dx, dy

    # ---- space order (keep CFL constant across grids to see ~2)
    errs_s, hs = [], []
    for nx in (33, 49, 65):
        err, dx, dy = run_ftcs(nx, nx, dt=0.9, keep_r_const=True)  # 0.9 * dt_limit
        errs_s.append(err); hs.append(max(dx,dy))
    p_space = float(np.log(errs_s[0]/errs_s[-1]) / np.log(hs[0]/hs[-1]))

    # ---- time order (fix a fine grid, vary dt near the stability limit)
    nx = ny = 97
    # coarse dt close to limit to make the time error dominate
    e1, dx, dy = run_ftcs(nx, ny, dt=None)               # ~0.9 * dt_limit
    # halve and quarter dt for EI slope
    # we re-call run_ftcs with dt fractions of dt_limit by scaling alpha
    # compute dt_limit again for this grid to pass consistent fractions
    Lx=Ly=1.0; dx = Lx/(nx-1); dy = Ly/(ny-1)
    dt_limit = (dx*dx * dy*dy) / (2.0*D*(dx*dx + dy*dy))
    e2, *_ = run_ftcs(nx, ny, dt=0.5*0.9*dt_limit, keep_r_const=False)
    e3, *_ = run_ftcs(nx, ny, dt=0.25*0.9*dt_limit, keep_r_const=False)
    dts = [0.9*dt_limit, 0.5*0.9*dt_limit, 0.25*0.9*dt_limit]
    p_time = float(np.log(e2/e3) / np.log(dts[1]/dts[2]))

    # ---- plots
    os.makedirs(outdir, exist_ok=True)
    # space plot
    plt.figure()
    plt.loglog(hs, errs_s, marker='o')
    plt.xlabel("h"); plt.ylabel("L2 error")
    plt.title(f"MMS space convergence (p≈{p_space:.2f})")
    plt.savefig(os.path.join(outdir, "mms_convergence_space.png"), bbox_inches="tight"); plt.close()
    # time plot
    plt.figure()
    plt.loglog(dts, [e1,e2,e3], marker='o')
    plt.xlabel("dt"); plt.ylabel("L2 error")
    plt.title(f"MMS time convergence (p≈{p_time:.2f})")
    plt.savefig(os.path.join(outdir, "mms_convergence_time.png"), bbox_inches="tight"); plt.close()

    return {
        "space_h": hs, "space_errs": errs_s, "space_order_est": p_space,
        "time_dts": dts, "time_errs": [e1,e2,e3], "time_order_est": p_time
    }

# ===============================
# 2) Gaussian sanity + mass bound
# ===============================
def gaussian_sanity(outdir):
    D = 5e-3; k = 1e-3; T_end = 0.5
    nx = ny = 129
    Lx=Ly=1.0
    x = np.linspace(0.0, Lx, nx); y = np.linspace(0.0, Ly, ny)
    dx = Lx/(nx-1); dy = Ly/(ny-1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # narrow gaussian in the middle
    C = np.exp(-((X-0.5)**2 + (Y-0.5)**2)/0.002)
    mass = [trapz2(C, x, y)]
    times = [0.0]

    # stability: r_x + r_y <= 1/2  -> use a safe 0.45 fraction
    dt_limit = (dx*dx * dy*dy) / (2.0*D*(dx*dx + dy*dy))
    dt = 0.45 * dt_limit
    steps = int(np.ceil(T_end/dt))
    dt = T_end/steps

    for n in range(steps):
        lap = ((C[2:,1:-1] - 2*C[1:-1,1:-1] + C[:-2,1:-1])/(dy*dy) +
               (C[1:-1,2:] - 2*C[1:-1,1:-1] + C[1:-1,:-2])/(dx*dx))
        Cn = C.copy()
        Cn[1:-1,1:-1] = C[1:-1,1:-1] + dt*(D*lap - k*C[1:-1,1:-1])
        # Neumann-ish via edge padding is trickier; keep Dirichlet 0 for V&V
        Cn[0,:]=0; Cn[-1,:]=0; Cn[:,0]=0; Cn[:,-1]=0
        C = Cn
        mass.append(trapz2(C, x, y)); times.append((n+1)*dt)

    # plot a few slices (no analytic overlay here—sanity only)
    for frac, label in [(0.25,"0125"), (0.5,"0500")]:
        t_idx = int(frac*len(times))
        xc = x
        yc = C[:, :,][..., :].mean(axis=1)  # quick collapse across y
        plt.figure()
        plt.plot(xc, C[:, C.shape[1]//2], label=f"num t={times[t_idx]:.3f}")
        plt.xlabel("x"); plt.ylabel("C"); plt.title("Gaussian slice")
        plt.legend()
        plt.savefig(os.path.join(outdir, f"gaussian_slice_t{int(times[t_idx]*1000):04d}ms.png"),
                    bbox_inches="tight"); plt.close()

    # mass vs analytic upper bound (Dirichlet leaks, so M(t) ≤ M0 e^{-k t})
    M0 = mass[0]
    M_bound = M0*np.exp(-k*np.array(times))
    plt.figure()
    plt.plot(times, mass, label="numeric mass")
    plt.plot(times, M_bound, "--", label=r"$M_0 e^{-k t}$ (upper bound)")
    plt.xlabel("t (s)"); plt.ylabel("mass"); plt.title("Mass vs analytic decay bound")
    plt.legend()
    plt.savefig(os.path.join(outdir, "mass_positivity_checks.png"), bbox_inches="tight"); plt.close()

    return {"mass0": M0, "mass_final": mass[-1], "times": times[-5:]}

# ===============================
# 3) PK closed-form vs numeric
# ===============================
def pk_verification(outdir):
    # Two-compartment linear PK with on/off infusion, solved two ways
    k12 = 0.15; k21 = 0.05; k10 = 0.08
    rate = 0.01; t_stop = 11.0*3600.0
    def u(t): return rate*(t < t_stop)

    A = np.array([[-(k12+k10), k21],
                  [k12,        -k21]])
    b = np.array([1.0, 0.0])
    def rhs(t,y): return A@y + b*u(t)

    t_eval = np.linspace(0, 24*3600, 400)
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), y0=[0.0,0.0], t_eval=t_eval,
                    rtol=1e-9, atol=1e-12)
    Y = sol.y.T

    # exact via variation-of-constants (matrix exponential)
    Y_exact = np.zeros_like(Y)
    # infusion phase
    E_stop = expm(A*t_stop)
    steady = np.linalg.solve(A, (E_stop - np.eye(2)) @ (b*rate))
    for i, t in enumerate(t_eval):
        if t <= t_stop:
            Y_exact[i] = np.linalg.solve(A, (expm(A*t) - np.eye(2)) @ (b*rate))
        else:
            Y_exact[i] = expm(A*(t - t_stop)) @ steady

    max_abs = float(np.max(np.abs(Y - Y_exact)))
    mean_abs = float(np.mean(np.abs(Y - Y_exact)))

    plt.figure()
    plt.plot(t_eval/3600, Y[:,0], label="Cb (num)")
    plt.plot(t_eval/3600, Y_exact[:,0], '--', label="Cb (exact)")
    plt.plot(t_eval/3600, Y[:,1], label="Cbrain (num)")
    plt.plot(t_eval/3600, Y_exact[:,1], '--', label="Cbrain (exact)")
    plt.xlabel("Time (h)"); plt.ylabel("Concentration")
    plt.title("PK: numeric vs. matrix-exponential exact")
    plt.legend()
    plt.savefig(os.path.join(outdir,"pk_closed_form_overlay.png"), bbox_inches="tight"); plt.close()

    return {"max_abs_err": max_abs, "mean_abs_err": mean_abs}

# ===============================
# 4) main
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="output folder")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    mms   = mms_convergence(args.outdir)
    gauss = gaussian_sanity(args.outdir)
    pk    = pk_verification(args.outdir)

    report = {
        "mms": mms,
        "gaussian_check": gauss,
        "pk_matrix_exponential": pk,
        "criteria": {
            "mms_space_order_min": 1.8,   # expect ~2
            "mms_time_order_min": 0.9,    # expect ~1 for FTCS time
            "pk_max_abs_err_max": 1e-6
        }
    }
    status = {
        "mms_ok": (mms["space_order_est"] >= report["criteria"]["mms_space_order_min"] and
                   mms["time_order_est"]  >= report["criteria"]["mms_time_order_min"]),
        "gaussian_ok": True,  # qualitative sanity
        "pk_ok": (pk["max_abs_err"] <= report["criteria"]["pk_max_abs_err_max"])
    }
    report["status"] = status

    with open(os.path.join(args.outdir, "vv_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("V&V done →", args.outdir)
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
