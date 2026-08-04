"""
Pleistocene Ice Layer (PIL) modeling and visualization script for Barnes Ice Cap.
Computes the estimated PIL thickness along MCoRDS flight tracks and solves the
Shallow Ice Approximation (SIA) velocity profile showing basal shear enhancement.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "2015_85564591_v2", "IRMCR2_20150507_07.csv")


def load_data() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing primary data file at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["LAT", "LON"])
    df["ice_thickness"] = df["THICK"].apply(lambda val: float(val) if val > 0 else 0.0)
    
    # Compute surface elevation
    df["surface_elevation"] = df["Actual surface"]
    return df


def estimate_pil_thickness(thick: float) -> float:
    """Estimates Pleistocene Ice Layer (PIL) thickness as a basal layer.
    Typical models limit PIL to deep ice (e.g. thickness > 150m) and cap it at 80m."""
    if thick > 150.0:
        return min(0.12 * thick, 80.0)
    return 0.0


def compute_sia_velocity(H: float, Hp: float, E: float = 3.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes SIA vertical velocity profile relative to the bed.
    H: total ice thickness (meters)
    Hp: Pleistocene Ice Layer thickness (meters)
    E: enhancement factor (softness multiplier, typical 3.1 - 3.5)
    Returns:
    - z: vertical grid from bed (0) to surface (H)
    - u_with_pil: velocity profile with PIL
    - u_without_pil: velocity profile without PIL (E=1)
    """
    n = 3
    A0 = 1e-16  # base ice fluidity (Pa^-3 yr^-1)
    rho = 917.0  # ice density (kg/m^3)
    g = 9.81  # gravity (m/s^2)
    slope = 0.02  # surface slope angle (radians)
    
    factor = 2 * A0 * (rho * g * slope) ** n
    z = np.linspace(0, H, 200)
    
    # 1. Base profile (no PIL, E=1 everywhere)
    u_no = factor * (H**(n+1) - (H - z)**(n+1)) / (n+1)
    
    # 2. Profile with PIL (E enhancement for z <= Hp)
    u_pil = np.zeros_like(z)
    for i, zi in enumerate(z):
        if zi <= Hp:
            # Inside the PIL layer (enhanced fluidity E * A0)
            u_pil[i] = E * factor * (H**(n+1) - (H - zi)**(n+1)) / (n+1)
        else:
            # Above the PIL layer (normal fluidity A0)
            # Velocity at transition height Hp
            u_transition = E * factor * (H**(n+1) - (H - Hp)**(n+1)) / (n+1)
            # Added velocity above Hp
            u_above = factor * ((H - Hp)**(n+1) - (H - zi)**(n+1)) / (n+1)
            u_pil[i] = u_transition + u_above
            
    # Normalize to surface velocity of the base case for visualization
    u_surf_no = u_no[-1]
    return z, u_pil / u_surf_no, u_no / u_surf_no


def run_pil_analysis():
    print("Loading flight track data...")
    df = load_data()
    
    # Compute PIL thickness along tracks
    print("Estimating Pleistocene Ice Layer (PIL) thickness...")
    df["pil_thickness"] = df["ice_thickness"].apply(estimate_pil_thickness)
    
    # Generate Map of PIL distribution
    print("Plotting PIL distribution map...")
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(df["LON"], df["LAT"], c=df["pil_thickness"], cmap="Purples", s=2, alpha=0.8)
    ax.set_title("Estimated Pleistocene Ice Layer (PIL) Thickness\nBarnes Ice Cap 2015", fontsize=10, fontweight="bold")
    ax.set_xlabel("Longitude (deg W)", fontsize=8)
    ax.set_ylabel("Latitude (deg N)", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("PIL Thickness (meters)", fontsize=8)
    
    map_path = os.path.join(BASE_DIR, "pil_distribution_map.png")
    fig.savefig(map_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PIL map to: {map_path}")
    
    # Solve SIA profile for a typical deep track point (e.g. H = 400m, Hp = 48m)
    print("Solving SIA velocity profiles...")
    H_val = 400.0
    Hp_val = estimate_pil_thickness(H_val)
    z, u_pil, u_no = compute_sia_velocity(H_val, Hp_val, E=3.5)
    
    # Plot velocity profile comparison
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(u_no, z / H_val, 'k--', label="Holocene Ice Only (E=1)")
    ax.plot(u_pil, z / H_val, 'purple', linewidth=2, label="With Basal Pleistocene Ice Layer (E=3.5)")
    
    # Draw horizontal line at Hp transition
    ax.axhline(Hp_val / H_val, color="red", linestyle=":", label=f"PIL Boundary (z/H = {Hp_val/H_val:.2f})")
    
    # Shading the PIL layer
    ax.axhspan(0, Hp_val / H_val, color='purple', alpha=0.1, label="Pleistocene Ice Layer")
    
    ax.set_title(f"SIA Vertical Velocity Profile Comparison\n(Ice Thickness H = {H_val:.0f}m, PIL Hp = {Hp_val:.0f}m)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Normalized Velocity u(z)/u_surf(base)", fontsize=9)
    ax.set_ylabel("Normalized Height Above Bed (z/H)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)
    
    profile_path = os.path.join(BASE_DIR, "pil_velocity_profile.png")
    fig.savefig(profile_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved velocity profile to: {profile_path}")
    print("=== PIL Analysis Complete ===")


if __name__ == "__main__":
    run_pil_analysis()
