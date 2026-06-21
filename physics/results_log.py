# Copyright (c) 2026 Mariusz "Vidi" Rossa
# Licensed under the MIT License - see LICENSE file for details.

"""
results_log.py - saving and reading Wheel analysis results

Results are GENERATED through actual analysis of equations from equations_db,
not manually hardcoded.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path

LOG_PATH          = Path(__file__).parent.parent / "results" / "wheel_results.json"
CALCULUS_LOG_PATH = Path(__file__).parent.parent / "results" / "calculus_results.json"


@dataclass
class CalculusEntry:
    """
    Entry from wheel_calculus analysis - four-fold division of singularities.
    Saved to calculus_results.json.
    """
    timestamp:      str
    name:           str
    expression_str: str
    variables:      str          # e.g., "x→0" or "p→m, m→0"
    wheel_result:   str          # always "⊥" (calculus runs when Wheel yields ⊥)
    calculus_type:  str          # "REMOVABLE" | "POLE" | "BOTTOM" | "FINITE"
    limit_value:    str          # limit value or "⊥"
    taylor_order:   int | None   # expansion order (REMOVABLE) or pole order (POLE)
    series_hint:    str          # fragment of Taylor expansion or Laurent hint
    pole_order:     int | None   # pole order (only POLE), otherwise None
    residue:        str          # residue as string (only POLE order=1), otherwise ""
    note:           str = ""


@dataclass
class AnalysisEntry:
    """
    One equation's Wheel analysis, across all of its known singular points.

    wheel_verdict reflects ONLY the raw Wheel result on the real axis
    ("BOTTOM" | "MIXED" | "FINITE") - it is never overridden by complex-pole
    findings. A complex pole near a real point where Wheel is finite is
    architecturally a separate fact (see wheel_calculus.ComplexPoleSingularity):
    Wheel is correct to be finite there; the complex-pole note is additional
    context, not a correction. Each dict in singular_points may carry an
    optional "complex_pole_note" key (str, only present when Wheel was finite
    AND a nearby off-axis pole was found) - see generate_log().
    """
    timestamp:       str
    equation_name:   str
    domain:          str
    expression_str:  str
    singular_points: list[dict]
    wheel_verdict:   str        # "BOTTOM" | "MIXED" | "FINITE"
    hypothesis:      str
    notes:           str = ""

    @staticmethod
    def _compute_verdict(singular_points: list[dict]) -> str:
        """
        BOTTOM  - all known singularities yield ⊥
        FINITE  - none yield ⊥ (Wheel didn't detect)
        MIXED   - some ⊥, some finite AND the finite ones are not artifacts
        
        If finite results have the word 'artifact'/'regular'/'finite'/'regular horizon'
        in their note - we treat them as correct (we do not lower the verdict).
        """
        if not singular_points:
            return "FINITE"

        results  = [sp["wheel_result"] for sp in singular_points]
        bottoms  = [r == "⊥" for r in results]
        finites  = [r != "⊥" for r in results]

        if all(bottoms):
            return "BOTTOM"
        if not any(bottoms):
            return "FINITE"

        # We have a mix - check if the finite ones are artifacts
        artifact_keywords = ("artifact", "regular", "finite", "regular horizon")
        finite_pts = [sp for sp in singular_points if sp["wheel_result"] != "⊥"]
        all_artifacts = all(
            any(kw in sp.get("note", "").lower() for kw in artifact_keywords)
            for sp in finite_pts
        )
        return "BOTTOM" if all_artifacts else "MIXED"


class ResultsLog:
    def __init__(self, path: Path = LOG_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[AnalysisEntry] = self._load()

    def _load(self) -> list[AnalysisEntry]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text())
            return [AnalysisEntry(**e) for e in data]
        except Exception:
            return []

    def save(self) -> None:
        self.path.write_text(
            json.dumps([asdict(e) for e in self._entries],
                       indent=2, ensure_ascii=False)
        )

    def add(self, entry: AnalysisEntry) -> None:
        self._entries.append(entry)
        self.save()

    def clear(self) -> None:
        self._entries = []
        self.save()

    def all(self) -> list[AnalysisEntry]:
        return self._entries

    def by_domain(self, domain: str) -> list[AnalysisEntry]:
        return [e for e in self._entries if e.domain.upper() == domain.upper()]

    def by_verdict(self, verdict: str) -> list[AnalysisEntry]:
        return [e for e in self._entries if e.wheel_verdict == verdict.upper()]

    def print_summary(self) -> None:
        if not self._entries:
            print("  Log is empty. Run: python main.py --regen-log")
            return

        print("═" * 64)
        print(f"  WHEELPHYSICS - Results log  [{len(self._entries)} entries]")
        print("═" * 64)

        by_domain: dict[str, list] = {}
        for e in self._entries:
            by_domain.setdefault(e.domain, []).append(e)

        icons = {"BOTTOM": "⊥", "FINITE": "✓", "MIXED": "±"}
        for domain, entries in sorted(by_domain.items()):
            print(f"\n  [{domain}]")
            for e in entries:
                icon = icons.get(e.wheel_verdict, "?")
                print(f"    {icon}  {e.equation_name:<42} {e.timestamp[:10]}")
                if e.hypothesis:
                    short = e.hypothesis[:70] + "..." if len(e.hypothesis) > 70 else e.hypothesis
                    print(f"       → {short}")

        b = len(self.by_verdict("BOTTOM"))
        f = len(self.by_verdict("FINITE"))
        m = len(self.by_verdict("MIXED"))
        print(f"\n  Verdicts: ⊥={b}  ✓={f}  ±={m}")
        print("═" * 64)


class CalculusLog:
    """
    Log of wheel_calculus results - three-fold division of singularities.
    Saves to calculus_results.json next to wheel_results.json.
    """

    def __init__(self, path: Path = CALCULUS_LOG_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[CalculusEntry] = self._load()

    def _load(self) -> list[CalculusEntry]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text())
            return [CalculusEntry(**e) for e in data]
        except Exception:
            return []

    def save(self) -> None:
        self.path.write_text(
            json.dumps([asdict(e) for e in self._entries],
                       indent=2, ensure_ascii=False)
        )

    def add(self, entry: CalculusEntry) -> None:
        self._entries.append(entry)
        self.save()

    def clear(self) -> None:
        self._entries = []
        self.save()

    def all(self) -> list[CalculusEntry]:
        return self._entries

    def by_type(self, calculus_type: str) -> list[CalculusEntry]:
        return [e for e in self._entries if e.calculus_type == calculus_type.upper()]

    def print_summary(self) -> None:
        if not self._entries:
            print("  Calculus log is empty. Run: python main.py --module calculus")
            return

        rem  = self.by_type("REMOVABLE")
        pol  = self.by_type("POLE")
        bot  = self.by_type("BOTTOM")
        fin  = self.by_type("FINITE")
        total = len(self._entries)

        print("═" * 64)
        print(f"  WHEEL CALCULUS - Results log  [{total} entries]")
        print(f"  {self.path}")
        print("═" * 64)

        if rem:
            print(f"\n  [REMOVABLE - Wheel=⊥ but lim exists]  ({len(rem)})")
            for e in rem:
                print(f"    ⊥→{e.limit_value:<8}  {e.name:<40}  order={e.taylor_order}")
                if e.series_hint:
                    hint = e.series_hint[:60] + "…" if len(e.series_hint) > 60 else e.series_hint
                    print(f"             Taylor: {hint}")

        if pol:
            print(f"\n  [POLES - order + residue (Cauchy)]  ({len(pol)})")
            for e in pol:
                res_str = f"res={e.residue}" if e.residue else "res=N/A"
                print(f"    ⊥  order={e.pole_order}  {res_str:<20}  {e.name}")
                if e.series_hint:
                    hint = e.series_hint[:60] + "…" if len(e.series_hint) > 60 else e.series_hint
                    print(f"             Laurent: {hint}")

        if bot:
            print(f"\n  [BOTTOM - ⊥ without structure (fallback)]  ({len(bot)})")
            for e in bot:
                print(f"    ⊥         {e.name:<40}  {e.variables}")

        if fin:
            print(f"\n  [REGULAR - Wheel OK]  ({len(fin)})")
            for e in fin:
                print(f"    ✓={e.limit_value:<8}  {e.name:<40}")

        score = (len(rem) + len(pol) + len(fin)) / total if total else 0
        print(f"\n  Four-fold division: REMOVABLE={len(rem)}  POLES={len(pol)}  BOTTOM={len(bot)}  REGULAR={len(fin)}")
        ts = self._entries[-1].timestamp[:10] if self._entries else "—"
        print(f"  Last analysis: {ts}")
        print("═" * 64)


# ─── Generator - produces log from actual analysis ────────────────────────────

# Research hypotheses per equation - the only place where they are hardcoded,
# because this is interpretation, not a computational result.
_HYPOTHESES = {
    "Schwarzschild g_rr": (
        "g_rr(r_s)=⊥ and g_rr(0)=⊥, but K(r_s) is finite. "
        "Wheel itself does not distinguish artifact from physical singularity - "
        "only combining g_rr with K shows it.",
        "Key result: need for invariant K to distinguish singularity types"
    ),
    "Schwarzschild g_tt": (
        "g_tt(r=0)=⊥. At r=r_s: g_tt=0 (not ⊥) - horizon is regular for g_tt.",
        ""
    ),
    "Kretschmann invariant": (
        "K(r=0)=⊥ - true singularity. K(r=r_s)=12/r_s⁴ - finite. "
        "Wheel via K correctly distinguishes physical singularity from coordinate artifact. "
        "This is the strongest result of the project in GR domain.",
        "Result 100% consistent with GR - K is a tensor invariant"
    ),
    "Christoffel symbol Γ^t_tr": (
        "Γ=⊥ both at r=r_s and r=0. Christoffel symbols depend on coordinate "
        "system - their ⊥ at r=r_s is an artifact, not physics.",
        "Confirms: Wheel on frame-dependent objects requires caution"
    ),
    "Friedmann - curvature term": (
        "H²(a=0)=⊥. Wheel allows a<0 mathematically - symmetry through a=0. "
        "Hypothesis: a<0 = pre-big-bang. Formal equations are continuous through singularity.",
        "Pre-big-bang hypothesis requires physical interpretation"
    ),
    "Matter density ρ~1/a³": (
        "ρ_matter(a=0)=⊥. Consistent with Friedmann - Big Bang is a singularity "
        "in all terms of the equation simultaneously.",
        ""
    ),
    "Radiation density ρ~1/a⁴": (
        "ρ_rad(a=0)=⊥. Stronger divergence than matter (a⁻⁴ vs a⁻³), "
        "Wheel treats both identically - ⊥.",
        ""
    ),
    "Feynman scalar propagator": (
        "On-shell pole (p=±m) → ⊥. Off-shell: finite and computable. "
        "Feynman's iε prescription is a classical trick bypassing the pole in the complex plane. "
        "Wheel passes through the pole directly. "
        "Hypothesis: ⊥ on-shell = algebraic definition of observability.",
        "On-shell=⊥ could be a deeper structure of QFT"
    ),
    "Photon propagator": (
        "k=0 → ⊥. Massless photon on-shell at k=0. "
        "IR singularity - physically: zero-momentum photon does not exist.",
        ""
    ),
    "Fermion propagator (simplified)": (
        "p=±m_e → ⊥. Electron and positron on-shell. "
        "Numerator (p+m_e) at p=-m_e gives 0, denominator also 0 → 0/0 form → ⊥.",
        "Both particle and antiparticle marked by the same ⊥"
    ),
    "Massless fermion propagator": (
        "p=0 → ⊥. Massless fermion (neutrino) cannot have p=0 - "
        "it always moves at speed c. "
        "Wheel algebraically forbids this state, does not postulate the ban.",
        "Cleanest example: ⊥ = physically impossible state"
    ),
    "Boltzmann distribution 1/T": (
        "T=0 → ⊥. Absolute zero is a thermodynamic singularity. "
        "Wheel: at T=0 occupation probability → ⊥ instead of 0 or ∞.",
        "Third law of thermodynamics - T=0 unattainable"
    ),
    "Planck distribution": (
        "T=0 → ⊥ (denominator exp(∞)-1 → ∞). "
        "0/0 form when T→∞ (classical Rayleigh-Jeans limit = 2k_BT) - "
        "Wheel gives ⊥ instead of finite limit. "
        "Same limitation as sinc: Wheel ≠ limit theory.",
        "Edge case: limit T→∞ exists classically, Wheel yields ⊥"
    ),
    "sinc(x) = sin(x)/x": (
        "KEY COUNTEREXAMPLE: sin(0)/0 = 0/0 → ⊥ in Wheel. "
        "Classically: lim(x→0) sin(x)/x = 1 (limit exists and is finite). "
        "Wheel DOES NOT compute limits - it substitutes the value literally. "
        "Fundamental method boundary: Wheel is an algebra (pointwise substitution), "
        "not mathematical analysis (limits). "
        "0/0 forms where limit is finite lie OUTSIDE the scope of Wheel.",
        "IMPORTANT COUNTEREXAMPLE - Wheel ≠ limit theory"
    ),
}


def generate_log(log: ResultsLog, verbose: bool = True) -> None:
    """
    Generates log via actual analysis of all equations from equations_db.
    Clears previous log before generating.

    For singular points where Wheel returns a FINITE result, additionally
    attempts a complex-pole scan (wheel_calculus.analyse_complex_poles).
    This does NOT change wheel_verdict - it only attaches an optional
    "complex_pole_note" to the relevant singular_points entry when a nearby
    off-real-axis pole is found. See AnalysisEntry docstring.
    """
    from physics.equations_db import EquationsDB
    from core.sympy_extension import wheel_subs
    from core.wheel_calculus import analyse_complex_poles
    import sympy as sp

    log.clear()
    db = EquationsDB()

    if verbose:
        print(f"  Analyzing {len(db.all())} equations...\n")

    for eq in db.all():
        singular_points = []

        for sing in eq.known_singular:
            var = sing["var"]
            val = sing["value"]

            # Run actual Wheel analysis
            w = wheel_subs(eq.expression, {var: val})
            w_str = "⊥" if w.is_bottom else str(w.value)[:40]

            # Classical result
            try:
                classical = sp.simplify(eq.expression.subs({var: val}))
                cl_str = str(classical)[:40]
            except Exception:
                cl_str = "error"

            point_entry = {
                "var":          str(var),
                "value":        str(val),
                "wheel_result": w_str,
                "classical":    cl_str,
                "note":         sing["description"],
            }

            # Complex-pole scan: only attempted when Wheel is finite (no ⊥ to
            # diagnose) and the substitution point is purely numeric/real, so
            # a complex root nearby would genuinely be invisible to Wheel.
            # Best-effort: sp.solve can fail or be slow on complicated
            # denominators, so any failure silently skips the note rather
            # than affecting the verdict or aborting the run.
            if not w.is_bottom:
                try:
                    cp = analyse_complex_poles(eq.expression, var, val, verbose=False)
                    if cp is not None:
                        point_entry["complex_pole_note"] = (
                            f"Wheel finite, but nearest complex pole at "
                            f"{var}={cp.nearest_pole} (|Im|={cp.distance_to_real_axis})"
                            + (f", res={cp.residue}" if cp.residue is not None else "")
                        )
                except Exception:
                    pass

            singular_points.append(point_entry)

        verdict = AnalysisEntry._compute_verdict(singular_points)

        hypothesis, notes = _HYPOTHESES.get(
            eq.name,
            (eq.physical_meaning, eq.notes)
        )

        entry = AnalysisEntry(
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
            equation_name=eq.name,
            domain=eq.domain,
            expression_str=str(eq.expression),
            singular_points=singular_points,
            wheel_verdict=verdict,
            hypothesis=hypothesis,
            notes=notes,
        )
        log.add(entry)

        if verbose:
            icon = {"BOTTOM": "⊥", "FINITE": "✓", "MIXED": "±"}.get(verdict, "?")
            has_cp = any("complex_pole_note" in sp_ for sp_ in singular_points)
            cp_flag = "  [complex pole nearby]" if has_cp else ""
            print(f"  [{icon}] {eq.name}{cp_flag}")

    if verbose:
        print(f"\n  Saved {len(log.all())} entries → {log.path}")


def generate_calculus_log(log: "CalculusLog", verbose: bool = True) -> None:
    """
    Runs wheel_calculus on test suite and saves results.

    Tests: counterexamples from db + genuine poles + regular points.
    Results land in calculus_results.json.
    """
    from core.wheel_calculus import wheel_limit, RemovableSingularity, PoleSingularity
    import sympy as sp

    log.clear()

    x, m, p, r, r_s = sp.symbols("x m p r r_s", real=True)
    omega, omega0    = sp.symbols("omega omega0", real=True)

    cases = [
        # ── Removable (counterexamples from db) ───────────────────────────
        dict(name="sinc(x) = sin(x)/x",
             expr=sp.sin(x)/x, variables=[(x, sp.S.Zero)],
             note="Key counterexample - Wheel=⊥, lim=1"),
        dict(name="sinc²(x) = sin²(x)/x²",
             expr=sp.sin(x)**2/x**2, variables=[(x, sp.S.Zero)],
             note="Square of sinc - same limit, higher Taylor order"),
        dict(name="(1 - cos(x))/x²",
             expr=(1 - sp.cos(x))/x**2, variables=[(x, sp.S.Zero)],
             note="Limit = 1/2 (not 1!) - different expansion type"),
        dict(name="Rayleigh-Jeans: x/(e^x - 1) as x→0",
             expr=x/(sp.exp(x) - 1), variables=[(x, sp.S.Zero)],
             note="x = ħω/kT → 0 as T→∞: classical Rayleigh-Jeans limit"),
        dict(name="(e^x - 1)/x",
             expr=(sp.exp(x) - 1)/x, variables=[(x, sp.S.Zero)],
             note="Derivative of exp at x=0 via difference quotient"),
        dict(name="tan(x)/x",
             expr=sp.tan(x)/x, variables=[(x, sp.S.Zero)],
             note="Removable - lim=1"),
        dict(name="(sin(3x))/(sin(5x)) as x→0",
             expr=sp.sin(3*x)/sp.sin(5*x), variables=[(x, sp.S.Zero)],
             note="Limit = 3/5 - L'Hôpital's rule or Taylor"),
        dict(name="(x² - 1)/(x - 1) as x→1",
             expr=(x**2 - 1)/(x - 1), variables=[(x, sp.S.One)],
             note="Classical example of removable singularity - lim=2"),
        # ── Poles (residue analysis) ──────────────────────────────────────
        dict(name="Scalar propagator 1/(p²-m²) [on-shell]",
             expr=1/(p**2 - m**2), variables=[(p, m)],
             note="Simple on-shell pole - res=1/(2m), isomorphism with resonance"),
        dict(name="Schwarzschild g_rr at r=r_s",
             expr=1/(1 - r_s/r), variables=[(r, r_s)],
             note="Event horizon - coordinate system pole, res=r_s"),
        dict(name="Oscillator resonance 1/(ω²-ω₀²) at ω=ω₀",
             expr=1/(omega**2 - omega0**2), variables=[(omega, omega0)],
             note="Classical resonance - res=1/(2ω₀), isomorphism with propagator"),
        dict(name="Photon propagator 1/k² at k=0",
             expr=1/x**2, variables=[(x, sp.S.Zero)],
             note="IR pole of order 2 - residue undefined (Cauchy only order=1)"),
        # ── Regular ───────────────────────────────────────────────────────
        dict(name="Euclidean KG 1/(p²+m²) at p=0, m=1",
             expr=1/(p**2 + m**2), variables=[(p, sp.S.Zero), (m, sp.S.One)],
             note="No pole on real axis - Wick rotation"),
        dict(name="Photon potential V_ph at r=3r_s/2",
             expr=(1 - r_s/r)/r**2, variables=[(r, sp.Rational(3,2)*r_s)],
             note="Schwarzschild photosphere - regular point"),
    ]

    if verbose:
        print(f"  Analyzing {len(cases)} wheel_calculus cases...\n")

    for case in cases:
        result = wheel_limit(
            case["expr"], case["variables"], max_order=8, verbose=False
        )

        vars_str = ", ".join(f"{v}→{p}" for v, p in case["variables"])

        if isinstance(result, RemovableSingularity):
            c_type    = "REMOVABLE"
            lim_str   = str(result.limit_value)
            order     = result.taylor_order
            hint      = result.series_hint
            pole_ord  = None
            residue   = ""
        elif isinstance(result, PoleSingularity):
            c_type    = "POLE"
            lim_str   = "⊥"
            order     = result.pole_order   # pole order
            hint      = result.laurent_hint
            pole_ord  = result.pole_order
            residue   = str(result.residue) if result.residue is not None else ""
        elif result.is_bottom:
            c_type    = "BOTTOM"
            lim_str   = "⊥"
            order     = None
            hint      = ""
            pole_ord  = None
            residue   = ""
        else:
            c_type    = "FINITE"
            lim_str   = str(result.value)
            order     = None
            hint      = ""
            pole_ord  = None
            residue   = ""

        entry = CalculusEntry(
            timestamp      = datetime.datetime.now().isoformat(timespec="seconds"),
            name           = case["name"],
            expression_str = str(case["expr"]),
            variables      = vars_str,
            wheel_result   = "⊥" if c_type in ("REMOVABLE", "POLE", "BOTTOM") else lim_str,
            calculus_type  = c_type,
            limit_value    = lim_str,
            taylor_order   = order,
            series_hint    = hint,
            pole_order     = pole_ord,
            residue        = residue,
            note           = case.get("note", ""),
        )
        log.add(entry)

        if verbose:
            if c_type == "REMOVABLE":
                label = f"⊥→{lim_str}"
            elif c_type == "POLE":
                res_part = f" res={residue}" if residue else ""
                label = f"⊥ POLE[{pole_ord}]{res_part}"
            elif c_type == "FINITE":
                label = f"✓={lim_str}"
            else:
                label = "⊥"
            print(f"  [{label:<20}]  {case['name']}")

    if verbose:
        rem = sum(1 for e in log.all() if e.calculus_type == "REMOVABLE")
        pol = sum(1 for e in log.all() if e.calculus_type == "POLE")
        bot = sum(1 for e in log.all() if e.calculus_type == "BOTTOM")
        fin = sum(1 for e in log.all() if e.calculus_type == "FINITE")
        print(f"\n  Saved {len(log.all())} entries → {log.path}")
        print(f"  Four-fold division: REMOVABLE={rem}  POLES={pol}  BOTTOM={bot}  REGULAR={fin}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regen", action="store_true",
                        help="Generate log from scratch via actual analysis")
    args = parser.parse_args()

    log = ResultsLog()

    if args.regen or not log.all():
        print("  Generating log from analysis...\n")
        generate_log(log)

    log.print_summary()