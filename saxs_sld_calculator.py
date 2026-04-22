"""
SAXS Scattering Length Density (SLD) Calculator for Lipid-Polymer Nanoparticles (LPNs)
========================================================================================

SLD formula (X-ray, electron density based):

    SLD [10⁻⁶ Å⁻²] = ρ × (Z / M) × 16.97

Derivation:
    SLD = r_e × N_A × ρ × (Z / M)

    where:
        r_e  = 2.818 × 10⁻¹³ cm   (classical electron radius)
        N_A  = 6.022 × 10²³ mol⁻¹  (Avogadro's number)
        ρ    = density in g/cm³
        Z    = electrons per molecule (dimensionless)
        M    = molecular weight in g/mol

    r_e × N_A = 2.818e-13 × 6.022e23 = 16.97 × 10¹⁰ cm² mol⁻¹

    SLD [cm⁻²] = 16.97 × 10¹⁰ × ρ × Z / M

    Convert cm⁻² → Å⁻²: multiply by (1 cm / 10⁸ Å)² = 10⁻¹⁶
    Then express in 10⁻⁶ Å⁻² units: multiply by 10⁶

    → SLD [10⁻⁶ Å⁻²] = 16.97 × ρ × (Z / M)

Why volume-weighted fractions:
    SLD is an intrinsic property of electron density (electrons per unit volume).
    When mixing materials, each contributes electrons in proportion to the VOLUME
    it occupies — not its mass or mole fraction. Using mass fractions would
    incorrectly up-weight denser components; using mole fractions would ignore
    molecular size. Volume fraction is the only physically correct weight.

Why hydration pulls SLD toward 9.42:
    Water has SLD = 9.42 × 10⁻⁶ Å⁻². When water fills a fraction f_water of the
    volume, those voxels contribute 9.42 instead of the dry material SLD.
    The resulting SLD is a linear mix:
        SLD_hydrated = (1 - f_water) × SLD_dry + f_water × 9.42
    Lipid/polymer SLDs (8–11) bracket water, so hydration nudges them all toward 9.42.
"""
# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  QUICK REFERENCE - individual component SAXS SLDs (×10⁻⁶ Å⁻²)               ║
# ╠═════════════════════════════════════════════════════════════════════════════╣
# ║  Component                           Z (e⁻)    MW      ρ        SLD         ║
# ║  ─────────────────────────────────────────────────────────────────────      ║
# ║  SM-102 (ionizable lipid, F3)         398    710.17  1.00     9.511         ║
# ║  C12-200 (ionizable lipid, F3C)       616   1136.96  1.00     9.194         ║
# ║  DSPC (saturated phospholipid)        438    790.15  1.02     9.595         ║
# ║  DOPE (unsaturated phospholipid)      410    744.03  0.97     9.071         ║
# ║  PLGA (repeat unit, 50:50)             68    130.10  1.34    11.886         ║
# ║  DMG-PEG2000 (non-phospho)           1350   2510.00  1.12    10.223         ║
# ║  DMPE-PEG2000 (phospho)              1332   2509.00  1.12    10.090         ║
# ║  mRNA (avg nucleotide residue)        172    340.00  1.65    14.165         ║
# ║  Water / PBS (reference)               10     18.02  1.00     9.417         ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import re
import sys

# Force UTF-8 output on Windows so Unicode superscripts render correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────
# Element electron counts (atomic number = number of electrons)
# ─────────────────────────────────────────────────────────────────
ELECTRONS: Dict[str, float] = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "P": 15,
    "S": 16,
}

SLD_PREFACTOR = 16.97  # r_e × N_A converted to Å⁻² units (see module docstring)


def parse_formula(formula: str) -> float:
    """
    Parse a chemical formula string and return total electron count.

    Handles fractional subscripts (e.g. C9.5H11.5N3.5O8P) for average residues.
    Supports simple formulas with no nested parentheses.

    Examples: "H2O" → 10.0,  "C9.5H11.5N3.5O8P" → 172.0
    """
    # Tokenise: each token is (element_symbol, count)
    # Pattern: capital letter + optional lowercase, then optional float subscript
    pattern = r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)"
    total = 0.0
    for element, count_str in re.findall(pattern, formula):
        if element not in ELECTRONS:
            raise ValueError(f"Unknown element '{element}' in formula '{formula}'")
        count = float(count_str) if count_str else 1.0
        total += ELECTRONS[element] * count
    return total


# ─────────────────────────────────────────────────────────────────
# Component dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class Component:
    """
    A molecular component with its physical properties.

    Fields
    ------
    name           : human-readable label
    formula        : chemical formula string (used to auto-compute n_electrons)
    molecular_weight: g/mol
    density        : g/cm³ (bulk or estimated)
    n_electrons    : total electrons per molecule (computed from formula if not given)
    """
    name: str
    formula: str
    molecular_weight: float   # g/mol
    density: float            # g/cm³
    n_electrons: float = field(default=0.0)

    def __post_init__(self) -> None:
        # Auto-compute n_electrons from formula if not manually provided
        if self.n_electrons == 0.0:
            self.n_electrons = parse_formula(self.formula)

    def sld(self) -> float:
        """
        Return SAXS SLD in 10⁻⁶ Å⁻² units.

        SLD = ρ × (Z / M) × 16.97
        """
        return self.density * (self.n_electrons / self.molecular_weight) * SLD_PREFACTOR


# ─────────────────────────────────────────────────────────────────
# Component library
# ─────────────────────────────────────────────────────────────────
# n_electrons comments show the per-element breakdown so the sum is auditable:
#   C44: 44×6=264, H87: 87×1=87, N1: 7, O5: 40  → 398
# etc.

COMPONENTS: Dict[str, Component] = {

    # ── Ionizable lipids ──────────────────────────────────────────
    "SM-102": Component(
        name="SM-102 (ionizable lipid)",
        formula="C44H87NO5",
        # electrons: C44=264, H87=87, N=7, O5=40  → 398
        molecular_weight=710.17,
        density=1.00,
    ),
    "C12-200": Component(
        name="C12-200 (ionizable lipid)",
        formula="C68H140N4O5",
        # electrons: C68=408, H140=140, N4=28, O5=40  → 616
        molecular_weight=1136.96,
        density=1.00,
    ),

    # ── Helper / structural lipids ────────────────────────────────
    "DSPC": Component(
        name="DSPC (saturated phospholipid)",
        formula="C44H88NO8P",
        # electrons: C44=264, H88=88, N=7, O8=64, P=15  → 438
        molecular_weight=790.15,
        density=1.02,
    ),
    "DOPE": Component(
        name="DOPE (unsaturated phospholipid)",
        formula="C41H78NO8P",
        # electrons: C41=246, H78=78, N=7, O8=64, P=15  → 410
        molecular_weight=744.03,
        density=0.97,
    ),

    # ── Polymer core ──────────────────────────────────────────────
    "PLGA": Component(
        name="PLGA (average repeat unit, 50:50)",
        formula="C5H6O4",
        # electrons: C5=30, H6=6, O4=32  → 68
        # This is an AVERAGE monomer (lactic + glycolic acid repeat ÷ 2).
        # For a polymer, SLD depends only on repeat-unit composition, not chain length,
        # so using the monomer gives the correct bulk SLD.
        molecular_weight=130.1,
        density=1.34,
    ),

    # ── PEG-lipids (stealth / corona) ────────────────────────────
    "DMG-PEG2000": Component(
        name="DMG-PEG2000 (non-phospho PEG-lipid)",
        formula="C117H232O52",
        # electrons: C117=702, H232=232, O52=416  → 1350
        # Approximate formula for a ~2 kDa PEG chain + dimyristoylglycerol anchor.
        molecular_weight=2510.0,
        density=1.12,
    ),
    "DMPE-PEG2000": Component(
        name="DMPE-PEG2000 (phospho PEG-lipid)",
        formula="C111H220NO53P",
        # electrons: C111=666, H220=220, N=7, O53=424, P=15  → 1332
        # Approximate formula; phosphate head replaces the glycerol ester of DMG-PEG.
        molecular_weight=2509.0,
        density=1.12,
    ),

    # ── Cargo ─────────────────────────────────────────────────────
    "mRNA": Component(
        name="mRNA (average nucleotide residue)",
        formula="C9.5H11.5N3.5O8P",
        # electrons: C9.5=57, H11.5=11.5, N3.5=24.5, O8=64, P=15  → 172
        # This is an AVERAGE over A/U/G/C residues weighted by typical mRNA composition.
        # Using the monomer SLD is valid for the same reason as PLGA above.
        molecular_weight=340.0,
        density=1.65,
    ),

    # ── Solvent reference ─────────────────────────────────────────
    "Water": Component(
        name="Water (buffer reference)",
        formula="H2O",
        # electrons: H2=2, O=8  → 10
        molecular_weight=18.02,
        density=1.00,
    ),
}


# ─────────────────────────────────────────────────────────────────
# Volume-weighted SLD mixer
# ─────────────────────────────────────────────────────────────────

def combine_sld(
    components_with_masses: Dict[str, float],
    water_fraction: float = 0.0,
) -> tuple[float, float]:
    """
    Compute the volume-weighted average SLD of a mixture.

    Parameters
    ----------
    components_with_masses : dict mapping component key → mass (in any consistent unit,
                             e.g. mg, relative weight fractions, or molar ratios × MW).
                             The absolute scale cancels; only the ratios matter.
    water_fraction         : volume fraction of water (0 = dry, 1 = pure water).

    Returns
    -------
    (sld_dry, sld_hydrated) both in 10⁻⁶ Å⁻²

    Volume fraction calculation
    ---------------------------
    For component i with mass m_i and density ρ_i:
        volume_i  = m_i / ρ_i
        v_frac_i  = volume_i / Σ(volume_j)    ← over DRY components only

    We use volume fractions (not mass fractions) because SLD is an electron
    density — electrons per unit volume — so each component contributes in
    proportion to the volume it occupies, not its mass.
    """
    if not components_with_masses:
        raise ValueError("components_with_masses must not be empty")
    if not (0.0 <= water_fraction < 1.0):
        raise ValueError("water_fraction must be in [0, 1)")

    # Compute partial volumes (volume ∝ mass / density)
    volumes: Dict[str, float] = {}
    for key, mass in components_with_masses.items():
        if key not in COMPONENTS:
            raise KeyError(f"Unknown component '{key}'. Available: {list(COMPONENTS)}")
        if mass < 0:
            raise ValueError(f"Mass for '{key}' must be ≥ 0, got {mass}")
        volumes[key] = mass / COMPONENTS[key].density

    total_volume = sum(volumes.values())
    if total_volume == 0:
        raise ValueError("All masses are zero — cannot compute SLD.")

    # Volume-weighted average of dry SLDs
    sld_dry = sum(
        (vol / total_volume) * COMPONENTS[key].sld()
        for key, vol in volumes.items()
    )

    # Apply hydration: water occupies water_fraction of the total wet volume.
    # The dry material fills the remaining (1 - water_fraction) fraction.
    # SLD_hydrated = (1 - f_w) × SLD_dry + f_w × SLD_water
    sld_water = COMPONENTS["Water"].sld()
    sld_hydrated = (1.0 - water_fraction) * sld_dry + water_fraction * sld_water

    return sld_dry, sld_hydrated


# ─────────────────────────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────────────────────────

def print_component_table() -> None:
    """Print a formatted table of individual component SLDs."""
    col_name  = 38
    col_z     = 8
    col_mw    = 10
    col_rho   = 8
    col_sld   = 12

    header = (
        f"{'Component':<{col_name}}"
        f"{'Z (e⁻)':>{col_z}}"
        f"{'MW (g/mol)':>{col_mw}}"
        f"{'ρ (g/cm³)':>{col_rho}}"
        f"{'SLD (10⁻⁶ Å⁻²)':>{col_sld}}"
    )
    sep = "─" * len(header)
    print("\n" + sep)
    print("  Available components and their SLDs")
    print(sep)
    print(header)
    print(sep)
    for key, comp in COMPONENTS.items():
        if key == "Water":
            continue
        sld_val = comp.sld()
        print(
            f"{comp.name:<{col_name}}"
            f"{comp.n_electrons:>{col_z}.1f}"
            f"{comp.molecular_weight:>{col_mw}.2f}"
            f"{comp.density:>{col_rho}.2f}"
            f"{sld_val:>{col_sld}.4f}"
        )
    sld_w = COMPONENTS["Water"].sld()
    print(sep)
    print(f"  Water (reference, SLD = {sld_w:.4f} × 10⁻⁶ Å⁻²) — set via water content below")
    print(sep + "\n")


if __name__ == "__main__":
    print_component_table()

    keys = [k for k in COMPONENTS if k != "Water"]
    print("  Enter components by number (comma-separated), e.g.  1,3,5")
    for i, k in enumerate(keys, 1):
        print(f"    {i:2d}. {k}")
    print()

    while True:
        raw = input("  Components: ").strip()
        try:
            indices = [int(x.strip()) for x in raw.split(",")]
            selected = [keys[i - 1] for i in indices]
            break
        except (ValueError, IndexError):
            print("  Invalid input — enter numbers from the list above.")

    masses: Dict[str, float] = {}
    print()
    for key in selected:
        while True:
            raw = input(f"  Mass / amount for {key} (any consistent unit): ").strip()
            try:
                masses[key] = float(raw)
                break
            except ValueError:
                print("  Please enter a number.")

    while True:
        raw = input("\n  Water content (% by volume, 0–100): ").strip()
        try:
            water_pct = float(raw)
            if 0.0 <= water_pct <= 100.0:
                break
            print("  Must be between 0 and 100.")
        except ValueError:
            print("  Please enter a number.")

    water_frac = water_pct / 100.0
    sld_dry, sld_hydrated = combine_sld(masses, water_fraction=water_frac)

    print("\n" + "═" * 50)
    print("  Result")
    print("═" * 50)
    print(f"  Components : {', '.join(masses.keys())}")
    print(f"  Water      : {water_pct:.1f} %")
    print(f"  SLD (dry)  : {sld_dry:.4f}  × 10⁻⁶ Å⁻²")
    print(f"  SLD (wet)  : {sld_hydrated:.4f}  × 10⁻⁶ Å⁻²")
    print("═" * 50 + "\n")
