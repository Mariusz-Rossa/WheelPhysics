# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
wheel_calculus.py — Wheel Algebra with analytical extension

Wheel Algebra (wheel_algebra.py) is a point algebra:
  every 0/0 form → ⊥, without exception.

This module adds an analytical layer: when Wheel returns ⊥, it diagnoses
the type of singularity and — for poles — calculates the residue and pole order.

Five-fold division of results (wheel_limit):
  ┌──────────────────────────────────────────────────────────────┐
  │ WheelFinite(value)        — regular point (Wheel OK)         │
  │ WheelBottom()             — irremovable ⊥ (undefined type)   │
  │ RemovableSingularity      — removable: Wheel=⊥, lim=val      │
  │ PoleSingularity           — alg. pole: order + res + Laurent │
  │ LogarithmicSingularity    — log. pole: log zeroes denominator│
  └──────────────────────────────────────────────────────────────┘

Formal type system — SingularityType (enum):
  REGULAR      → regular point (Wheel is finite)
  REMOVABLE    → removable singularity (Taylor)
  POLE_SIMPLE  → simple pole order=1 (Cauchy res defined)
  POLE_HIGHER  → higher-order pole (res N/A)
  ESSENTIAL    → essential singularity (Picard) — requires external analysis
  LOGARITHMIC  → ⊥ logarithmic pole: log factor zeroes the denominator (QCD)
  BRANCH_POINT → branch point — requires external analysis
  COORDINATE   → coordinate system artifact — requires invariants
  PHYSICAL     → confirmed physical singularity — requires invariants
  COMPLEX_POLE → complex pole (outside R) — open research question
  UNKNOWN      → fallback when classification fails

Architecture (intentional):
  wheel_algebra.py    — axiomatic, pure, unchanged
  wheel_calculus.py   — separate module, analytical extension
  consistency_checker — tests both, compares results

This distinction is important for the preprint:
  Wheel Algebra ≠ limit theory
  Wheel + Calculus = full apparatus for singularity analysis

Reference: Carlström (2004) "Wheels — On Division by Zero"
"""

from __future__ import annotations

import sys, os
# Support for both structure variants (flat repo and core/ subpackage)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# If repo uses core/ structure, create symlink-like namespace
try:
    from wheel_number import WheelNumber, BOTTOM, W, _coerce
except ImportError:
    from core.wheel_number import WheelNumber, BOTTOM, W, _coerce  # type: ignore

# We create a temporary 'core' module pointing to the current directory
# so that wheel_algebra.py and sympy_extension.py can do 'from core.xxx'
import types as _types
if 'core' not in sys.modules:
    _core_mod = _types.ModuleType('core')
    _core_mod.__path__ = [_HERE]
    _core_mod.__package__ = 'core'
    sys.modules['core'] = _core_mod

from dataclasses import dataclass, field
from typing import Optional, Union
import sympy as sp

try:
    from wheel_algebra import WheelAlgebra, _has_division_by_zero_at
except ImportError:
    from core.wheel_algebra import WheelAlgebra, _has_division_by_zero_at  # type: ignore

try:
    from sympy_extension import wheel_subs
except ImportError:
    from core.sympy_extension import wheel_subs  # type: ignore

_wa = WheelAlgebra()


# ─── SingularityType — formal singularity type ───────────────────────────────

from enum import Enum, auto

class SingularityType(Enum):
    """
    Formal singularity type — classification result of wheel_calculus.

    Hierarchy (from "most regular" to "most singular"):

      REGULAR          — regular point: Wheel yields a finite value,
                         no singularity. Calculus does not interfere.

      REMOVABLE        — removable singularity: Wheel yields ⊥ (0/0 form),
                         but an analytical limit exists and is finite.
                         Taylor expansion removes the singularity.
                         Example: sin(x)/x at x=0, lim=1

      POLE             — algebraic pole: Wheel yields ⊥, denominator → 0,
                         numerator ≠ 0. Limit = ±∞. Possesses order and residue.
                         Example: 1/(p²-m²) at p=m, res=1/(2m)

      POLE_SIMPLE      — simple pole (order=1): special case of POLE.
                         Residue is defined in Cauchy's sense.
                         Most important physically — QFT propagators.

      POLE_HIGHER      — higher-order pole (order≥2): residue N/A.
                         Example: 1/r² at r=0 (order=2).

      ESSENTIAL        — essential singularity:
                         not a pole, not removable — chaotic behavior
                         in the vicinity of the point (Picard-Weierstrass theorem).
                         Example: exp(1/z) at z=0.
                         Wheel yields ⊥; Taylor and Laurent do not help.

      LOGARITHMIC      — logarithmic singularity: log-type divergence.
                         Example: QCD gluon propagator with a loop correction.
                         Taylor does not work; requires asymptotic series.

      BRANCH_POINT     — branch point: multi-valued function
                         (e.g., √z, log z at z=0). Not a pole.

      COORDINATE       — coordinate system artifact, not a physical singularity.
                         Detected via invariants (e.g., Kretschmann K).
                         Example: g_rr at r=r_s — a pole in Schwarzschild,
                         but K(r_s) is finite.

      PHYSICAL         — confirmed physical singularity (not an artifact).
                         Wheel=⊥ AND scalar invariant is also ⊥.
                         Example: K at r=0 (curvature singularity).

      COMPLEX_POLE     — complex pole (lying outside the real axis, Im(z₀) ≠ 0).
                         Wheel (operating on R) misses it through
                         real substitution. Open research question.
                         Example: Green function of a damped oscillator γ>0.

      UNKNOWN          — unrecognized type — fallback when classification
                         fails or expression is too complex.
    """

    REGULAR       = auto()   # ✓  regular point
    REMOVABLE     = auto()   # ⊥→v  removable singularity
    POLE          = auto()   # ⊥  pole (general)
    POLE_SIMPLE   = auto()   # ⊥  simple pole (order=1, res defined)
    POLE_HIGHER   = auto()   # ⊥  higher-order pole (order≥2)
    ESSENTIAL     = auto()   # ⊥  essential singularity (Picard)
    LOGARITHMIC   = auto()   # ⊥  logarithmic divergence
    BRANCH_POINT  = auto()   # ⊥  branch point
    COORDINATE    = auto()   # ⊥* coordinate system artifact
    PHYSICAL      = auto()   # ⊥  confirmed physical singularity
    COMPLEX_POLE  = auto()   # ⊥? complex pole (outside R)
    UNKNOWN       = auto()   # ?  unrecognized

    @property
    def is_genuine_singularity(self) -> bool:
        """Is this a genuine (irremovable) singularity?"""
        return self not in (
            SingularityType.REGULAR,
            SingularityType.REMOVABLE,
            SingularityType.COORDINATE,
        )

    @property
    def has_residue(self) -> bool:
        """Is the Cauchy residue defined?"""
        return self == SingularityType.POLE_SIMPLE

    @property
    def short(self) -> str:
        """Short label for tables and logs."""
        _labels = {
            SingularityType.REGULAR:      "REGULAR",
            SingularityType.REMOVABLE:    "REMOVABLE",
            SingularityType.POLE:         "POLE",
            SingularityType.POLE_SIMPLE:  "POLE[1]",
            SingularityType.POLE_HIGHER:  "POLE[n]",
            SingularityType.ESSENTIAL:    "ESSENTIAL",
            SingularityType.LOGARITHMIC:  "LOG",
            SingularityType.BRANCH_POINT: "BRANCH",
            SingularityType.COORDINATE:   "COORD",
            SingularityType.PHYSICAL:     "PHYSICAL",
            SingularityType.COMPLEX_POLE: "COMPLEX",
            SingularityType.UNKNOWN:      "UNKNOWN",
        }
        return _labels.get(self, self.name)

    def __str__(self) -> str:
        return self.short


def singularity_type_of(result: "WheelCalcResult") -> SingularityType:
    """
    Returns the SingularityType for any wheel_calculus result.

    Mapping:
      WheelNumber (finite)       → REGULAR
      WheelNumber (⊥, BOTTOM)    → UNKNOWN  (no further information)
      RemovableSingularity       → REMOVABLE
      PoleSingularity order=1    → POLE_SIMPLE
      PoleSingularity order≥2    → POLE_HIGHER

    Note: COORDINATE, PHYSICAL, ESSENTIAL, LOGARITHMIC, COMPLEX_POLE
    require additional context (e.g., scalar invariants, asymptotic
    analysis) and cannot be deduced automatically from the wheel_limit
    result alone. They are assigned manually or by dedicated modules.
    """
    if isinstance(result, RemovableSingularity):
        return SingularityType.REMOVABLE
    if isinstance(result, LogarithmicSingularity):
        return SingularityType.LOGARITHMIC
    if isinstance(result, PoleSingularity):
        if result.pole_order == 1:
            return SingularityType.POLE_SIMPLE
        return SingularityType.POLE_HIGHER
    # WheelNumber
    if hasattr(result, 'is_bottom') and result.is_bottom:
        return SingularityType.UNKNOWN
    return SingularityType.REGULAR


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class RemovableSingularity:
    """
    Removable singularity — Wheel yields ⊥, but an analytical limit exists.

    Example:  sin(x)/x at x=0
      wheel_result  = ⊥
      limit_value   = 1
      taylor_order  = 1   (sin(x) ≈ x — x³/6, so sin(x)/x ≈ 1 - x²/6 → 1)
      variables     = [(x, 0)]
    """
    wheel_result:  WheelNumber          # always ⊥
    limit_value:   sp.Basic             # limit value
    taylor_order:  int                  # expansion order that gave the result
    variables:     list[tuple]          # [(var, point), ...] — can be multiple
    expression:    sp.Basic             # original expression
    series_hint:   str = ""             # readable Taylor expansion

    @property
    def is_removable(self) -> bool:
        return True

    @property
    def is_bottom(self) -> bool:
        return False

    @property
    def singularity_type(self) -> SingularityType:
        return SingularityType.REMOVABLE

    def __str__(self) -> str:
        vars_str = ", ".join(f"{v}→{p}" for v, p in self.variables)
        return (
            f"RemovableSingularity("
            f"lim[{vars_str}] = {self.limit_value}, "
            f"Taylor order = {self.taylor_order})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def report(self) -> str:
        """Readable report for the user / log."""
        vars_str = ", ".join(f"{v}→{p}" for v, p in self.variables)
        lines = [
            f"  Expression  : {self.expression}",
            f"  Point       : {vars_str}",
            f"  Wheel       : ⊥  (0/0 form — point substitution)",
            f"  Limit       : {self.limit_value}  ← wheel_calculus result",
            f"  Type        : REMOVABLE SINGULARITY",
            f"  Order       : Taylor to order {self.taylor_order}",
        ]
        if self.series_hint:
            lines.append(f"  Taylor      : {self.series_hint}")
        return "\n".join(lines)


@dataclass
class PoleSingularity:
    """
    Algebraic pole — Wheel yields ⊥, denominator zeroes out, numerator does not.

    Contains the full local pole structure:
      - pole order (1=simple, 2=double, ...)
      - residue (for order 1: coefficient of 1/(x-x0))
      - Laurent expansion around the pole

    Example: 1/(p²-m²) at p=m
      pole_order   = 1
      residue      = 1/(2m)
      laurent_hint = "1/(2m) · 1/(p-m) + O(1)"

    Example: 1/r² at r=0
      pole_order   = 2
      residue      = None  (residue only for order 1)
      principal_part = "1/r²"

    Connection with QFT:
      Propagator residue at pole = transition amplitude on the mass shell.
      Cauchy's residue theorem: ∮ f(z) dz = 2πi · Σ res(f, zₖ)
    """
    expression:    sp.Basic          # original expression
    variables:     list[tuple]       # [(var, point), ...]
    pole_order:    int               # pole order
    residue:       Optional[sp.Basic]  # residue (only order=1)
    laurent_coeff: sp.Basic          # lim (x-x0)^n · f(x) — principal coefficient
    laurent_hint:  str = ""          # readable description of the Laurent expansion

    @property
    def is_bottom(self) -> bool:
        return True   # ⊥ in the Wheel sense — a pole is an irremovable singularity

    @property
    def is_pole(self) -> bool:
        return True

    @property
    def is_simple_pole(self) -> bool:
        return self.pole_order == 1

    @property
    def singularity_type(self) -> SingularityType:
        if self.pole_order == 1:
            return SingularityType.POLE_SIMPLE
        return SingularityType.POLE_HIGHER

    def __str__(self) -> str:
        vars_str = ", ".join(f"{v}→{p}" for v, p in self.variables)
        res_str = f", res={self.residue}" if self.residue is not None else ""
        return (
            f"PoleSingularity("
            f"order={self.pole_order}{res_str}, "
            f"at [{vars_str}])"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def report(self) -> str:
        """Readable report for the user / log."""
        vars_str = ", ".join(f"{v}→{p}" for v, p in self.variables)
        lines = [
            f"  Expression   : {self.expression}",
            f"  Point        : {vars_str}",
            f"  Wheel        : ⊥  (pole — denominator→0, numerator≠0)",
            f"  Type         : ALGEBRAIC POLE (order {self.pole_order})",
            f"  Pole order   : {self.pole_order}",
        ]
        if self.residue is not None:
            lines.append(f"  Residue      : {self.residue}")
        else:
            lines.append(f"  Residue      : N/A (only for order 1 pole)")
        lines.append(f"  Prin. coeff. : {self.laurent_coeff}  [= lim (x-x₀)ⁿ·f(x)]")
        if self.laurent_hint:
            lines.append(f"  Laurent      : {self.laurent_hint}")
        lines.append(
            f"  QFT          : res = on-shell amplitude (Cauchy's theorem)"
            if self.pole_order == 1 else
            f"  QFT          : higher-order pole — anomalous divergence"
        )
        return "\n".join(lines)


@dataclass
class LogarithmicSingularity:
    """
    Logarithmic pole — Wheel yields ⊥, denominator zeroes out through a log factor.

    Difference compared to PoleSingularity:
      PoleSingularity  — denominator is a polynomial: (x - x₀)ⁿ
      LogarithmicSingularity — denominator contains log(x/μ²) which zeroes out

    Example: QCD gluon propagator 1/(k²·(1 + αs·log(k²/μ²)))
      singular_point     = μ²·exp(-1/αs)   ← Landau pole
      pole_order         = 1               ← simple pole (log zeroes linearly)
      residue            = 1/αs            ← computed via sp.residue
      log_factor         = "1 + αs·log(k²/μ²)"  ← factor that zeroes out
      laurent_hint       = "(1/αs) · 1/(k²-k²_L) + O(1)"

    Physical significance:
      QCD Landau pole appears at the confinement scale (~ΛQCD).
      Its counterpart in QED is unphysical (10^280 GeV).
      Residue 1/αs reflects the coupling strength at the singular point.

    Connection with Wheel:
      Wheel(expr, k2=k2_L) = ⊥  (denominator → 0, numerator = 1)
      Type: LOGARITHMIC — identified by the presence of log(var) in the denominator.
    """
    expression:    sp.Basic           # original expression
    variables:     list[tuple]        # [(var, point), ...]
    singular_point: sp.Basic          # variable value at the Landau pole
    pole_order:    int                # pole order (usually 1)
    residue:       Optional[sp.Basic] # residue (sp.residue)
    log_factor:    sp.Basic           # logarithmic factor that zeroes out
    laurent_hint:  str = ""           # readable description of the expansion

    @property
    def is_bottom(self) -> bool:
        return True  # ⊥ in the Wheel sense — a logarithmic pole is irremovable

    @property
    def is_logarithmic(self) -> bool:
        return True

    @property
    def singularity_type(self) -> SingularityType:
        return SingularityType.LOGARITHMIC

    def __str__(self) -> str:
        vars_str = ", ".join(f"{v}→{p}" for v, p in self.variables)
        res_str = f", res={self.residue}" if self.residue is not None else ""
        return (
            f"LogarithmicSingularity("
            f"order={self.pole_order}{res_str}, "
            f"at [{vars_str}], log_factor={self.log_factor})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def report(self) -> str:
        """Readable report for the user / log."""
        vars_str = ", ".join(f"{v}→{p}" for v, p in self.variables)
        lines = [
            f"  Expression   : {self.expression}",
            f"  Point        : {vars_str}",
            f"  Sing. point  : {self.singular_point}",
            f"  Wheel        : ⊥  (log pole — log factor zeroes the denominator)",
            f"  Type         : LOGARITHMIC POLE (order {self.pole_order})",
            f"  Log factor   : {self.log_factor}",
            f"  Pole order   : {self.pole_order}",
        ]
        if self.residue is not None:
            lines.append(f"  Residue      : {self.residue}")
        else:
            lines.append(f"  Residue      : N/A")
        if self.laurent_hint:
            lines.append(f"  Laurent      : {self.laurent_hint}")
        lines.append(
            f"  QCD          : residue = 1/αs — coupling strength at the Landau point"
            if self.pole_order == 1 else
            f"  QCD          : higher-order logarithmic pole — anomaly"
        )
        return "\n".join(lines)


# Union of types returned by wheel_calculus
WheelCalcResult = Union[WheelNumber, RemovableSingularity, PoleSingularity, LogarithmicSingularity]


# ─── Main function ────────────────────────────────────────────────────────────

def wheel_limit(
    expr:      sp.Basic,
    variables: list[tuple[sp.Symbol, sp.Basic]],
    max_order: int = 8,
    verbose:   bool = False,
) -> WheelCalcResult:
    """
    Main function of wheel_calculus — unified interface.

    Algorithm:
      1. Check Wheel (wheel_subs). If finite → return WheelNumber.
      2. If ⊥ → investigate the cause:
         a. Is it a 0/0 form? (numerator AND denominator → 0)
         b. Is it a genuine pole? (numerator ≠ 0, denominator → 0)
      3. For 0/0 form → attempt Taylor expansion (sequentially orders 1..max_order).
      4. If Taylor yields a finite value → RemovableSingularity.
      5. If Taylor does not help → WheelNumber(BOTTOM) (irremovable).

    Args:
        expr:      SymPy expression
        variables: list of pairs (symbol, limit_value)
                   e.g. [(x, 0)] or [(m, 0), (p, 0)]
        max_order: maximum Taylor expansion order (default 8)
        verbose:   whether to print diagnostic steps

    Returns:
        WheelNumber           — when regular point or irremovable ⊥
        RemovableSingularity  — when removable singularity (Wheel=⊥, lim=val)
    """
    subs_dict = {var: point for var, point in variables}

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  wheel_limit: {expr}")
        print(f"  at: {', '.join(f'{v}→{p}' for v,p in variables)}")

    # ── Step 1: Wheel Result ────────────────────────────────────────────────
    wheel_result = wheel_subs(expr, subs_dict)

    if not wheel_result.is_bottom:
        if verbose:
            print(f"  Wheel: {wheel_result}  (regular point, no action)")
        return wheel_result

    if verbose:
        print(f"  Wheel: ⊥  — checking singularity type...")

    # ── Step 2: Diagnosis — 0/0 or genuine pole? ──────────────────────────
    singularity_type = _classify_singularity(expr, subs_dict, verbose)

    if singularity_type == "logarithmic_pole":
        if verbose:
            print(f"  Type: LOGARITHMIC POLE (log factor zeroes the denominator) → log analysis")
        return _compute_logarithmic_pole(expr, variables, verbose)

    if singularity_type == "pole":
        if verbose:
            print(f"  Type: GENUINE POLE (numerator≠0, denominator→0) → residue analysis")
        return _compute_pole(expr, variables, verbose)

    if singularity_type == "essential":
        if verbose:
            print(f"  Type: ESSENTIAL SINGULARITY (e.g. exp(1/x)) → irremovable ⊥")
        return W(BOTTOM)

    if singularity_type == "unknown":
        if verbose:
            print(f"  Type: unknown — attempting Taylor as a fallback")

    # singularity_type == "removable_candidate" or "unknown"
    if verbose:
        print(f"  Type: CANDIDATE for removable singularity (0/0 form) → attempting Taylor")

    # ── Step 3: Taylor Expansion ───────────────────────────────────────────
    result = _try_taylor(expr, variables, max_order, verbose)

    if result is not None:
        limit_val, order, series_hint = result
        if verbose:
            print(f"  Taylor order {order}: lim = {limit_val}  → REMOVABLE ✓")
        return RemovableSingularity(
            wheel_result=wheel_result,
            limit_value=limit_val,
            taylor_order=order,
            variables=variables,
            expression=expr,
            series_hint=series_hint,
        )

    # ── Step 5: Taylor didn't help ────────────────────────────────────────
    if verbose:
        print(f"  Taylor to order {max_order}: no finite limit → irremovable ⊥")
    return W(BOTTOM)


# ─── Singularity classification ───────────────────────────────────────────────

def _classify_singularity(
    expr: sp.Basic,
    subs_dict: dict,
    verbose: bool = False,
) -> str:
    """
    Classifies the singularity type at a given substitution.

    Returns:
        "removable_candidate" — numerator AND denominator → 0 (0/0 form)
        "pole"                — numerator ≠ 0, denominator → 0
        "essential"           — essential singularity (exp(1/x))
        "unknown"             — cannot be classified
    """
    try:
        # Separate into numerator and denominator
        numer, denom = sp.fraction(sp.cancel(expr))

        numer_sub = sp.simplify(numer.subs(subs_dict))
        denom_sub = sp.simplify(denom.subs(subs_dict))

        numer_is_zero = (numer_sub == sp.S.Zero) or (
            hasattr(numer_sub, 'is_zero') and numer_sub.is_zero
        )
        denom_is_zero = (denom_sub == sp.S.Zero) or (
            hasattr(denom_sub, 'is_zero') and denom_sub.is_zero
        ) or denom_sub in (sp.zoo, sp.nan, sp.oo, -sp.oo)

        # When subs yields nan (e.g. 0*log(0)), use sp.limit as a fallback
        if denom_sub is sp.nan or denom_sub == sp.nan:
            try:
                if len(subs_dict) == 1:
                    v, pt = list(subs_dict.items())[0]
                    denom_lim = sp.limit(denom, v, pt)
                    denom_is_zero = (denom_lim == sp.S.Zero)
                    if verbose:
                        print(f"    denominator@subs=nan → sp.limit={denom_lim} {'(→0)' if denom_is_zero else ''}")
            except Exception:
                pass

        if verbose:
            print(f"    numerator after substitution  : {numer_sub} {'(=0)' if numer_is_zero else ''}")
            print(f"    denominator after substitution: {denom_sub} {'(=0)' if denom_is_zero else ''}")

        if numer_is_zero and denom_is_zero:
            return "removable_candidate"
        elif not numer_is_zero and denom_is_zero:
            # Check if the pole comes from a logarithmic factor.
            # We use sp.denom(expr) instead of sp.fraction(sp.cancel(expr)) —
            # sp.cancel expands the denominator and destroys the factor structure.
            # sp.denom preserves factors: k2*(1 + αs·log(k2/μ²)) → [k2, 1+αs·log(...)].
            #
            # Two cases of logarithmic pole:
            #   (A) Log factor zeroes out: 1+αs·log(k2/μ²)=0 → Landau pole
            #   (B) Log factor diverges and dominates: log(k2/μ²)→-∞ as k2→0
            #       Then k2^n * f → 0 for any n (no algebraic order)
            try:
                raw_denom = sp.denom(expr)
                denom_factors = sp.Mul.make_args(raw_denom)
                for fac in denom_factors:
                    if fac.has(sp.log):
                        # Case A: log factor zeroes out at the point
                        fac_val = sp.simplify(fac.subs(subs_dict))
                        if fac_val == sp.S.Zero:
                            if verbose:
                                print(f"    log factor {fac} → 0 → LOG POLE (Landau)")
                            return "logarithmic_pole"
                        # Case B: log factor diverges → no alg. order
                        # Test: lim (var-point)^1 * expr = 0 (not finite non-zero)
                        if len(subs_dict) == 1:
                            v, pt = list(subs_dict.items())[0]
                            try:
                                test_lim = sp.limit((v - pt) * expr, v, pt)
                                if test_lim == sp.S.Zero:
                                    if verbose:
                                        print(f"    lim (x-x0)·f=0 with log in denom → LOG POLE (IR)")
                                    return "logarithmic_pole"
                            except Exception:
                                pass
            except Exception:
                pass
            return "pole"

        # Check if it is an essential singularity (exp(1/x) etc.)
        if _has_essential_singularity(expr, subs_dict):
            return "essential"

        return "unknown"

    except Exception:
        return "unknown"


def _compute_pole(
    expr:      sp.Basic,
    variables: list[tuple[sp.Symbol, sp.Basic]],
    verbose:   bool = False,
    max_order: int = 8,
) -> WheelCalcResult:
    """
    Computes pole order and residue for an irremovable singularity.

    Algorithm:
      For n = 1, 2, ..., max_order:
        candidate = lim_{x→x0} (x - x0)^n · f(x)
        If candidate is finite and non-zero → order = n
        If candidate = 0 → order too low, try n+1
        If candidate = ∞ → computation error, try n+1

    Residue:
      For order 1: res = candidate  (because res = lim (x-x0)¹ · f(x))
      For order n: res = candidate / (n-1)!  — generalized formula
      But physically meaningful residue (in Cauchy's sense) only for n=1.

    Multivariate handling:
      For multiple variables, analysis based on the first variable (main singularity).
      Other variables treated as parameters.

    Returns:
        PoleSingularity — if successfully calculated order and residue
        WheelNumber(⊥) — fallback when computation is impossible
    """
    # For multiple variables: analysis based on the first one
    if len(variables) == 1:
        var, point = variables[0]
    else:
        # Find the variable with respect to which the denominator zeroes out
        var, point = variables[0]
        for v, p in variables:
            try:
                _, denom = sp.fraction(sp.cancel(expr))
                if sp.simplify(denom.subs(v, p)) == sp.S.Zero:
                    var, point = v, p
                    break
            except Exception:
                pass

    for n in range(1, max_order + 1):
        try:
            factor = (var - point) ** n
            candidate = sp.limit(factor * expr, var, point)

            if candidate in (sp.oo, -sp.oo, sp.zoo, sp.nan):
                continue  # order too low

            if candidate == sp.S.Zero:
                continue  # also too low — expression vanishes faster

            # We have a finite, non-zero result — this is the pole order
            candidate = sp.simplify(candidate)

            # Residue: only for simple pole (n=1)
            if n == 1:
                residue = candidate
            else:
                # Generalized Laurent coefficient: a_{-n} = candidate/(n-1)!
                residue = None   # Cauchy residue defined only for n=1

            # Build Laurent expansion hint
            if n == 1:
                hint = f"({candidate}) · 1/({var}-{point}) + O(1)"
            else:
                hint = f"({candidate}) · 1/({var}-{point})^{n} + O(1/({var}-{point})^{n-1})"

            if verbose:
                print(f"  Pole order   : {n}")
                print(f"  Prin. coeff. : {candidate}")
                if n == 1:
                    print(f"  Residue      : {residue}")
                print(f"  Laurent      : {hint}")

            return PoleSingularity(
                expression=expr,
                variables=variables,
                pole_order=n,
                residue=residue,
                laurent_coeff=candidate,
                laurent_hint=hint,
            )

        except Exception as e:
            if verbose:
                print(f"  Order {n}: error ({e}), attempting higher")
            continue

    # Fallback — failed to calculate
    if verbose:
        print(f"  Residue analysis failed → ⊥ (fallback)")
    return W(BOTTOM)


def _compute_logarithmic_pole(
    expr:      sp.Basic,
    variables: list[tuple[sp.Symbol, sp.Basic]],
    verbose:   bool = False,
) -> WheelCalcResult:
    """
    Logarithmic pole analysis — log factor log(var/μ²) zeroes out in the denominator.

    Strategy:
      1. Identify the variable and singular point (denominator → 0).
      2. Extract the logarithmic factor from the denominator.
      3. Compute residue via sp.residue (works for simple log poles).
      4. Check the order via lim (x-x0)^n · f(x).
      5. Build the Laurent expansion hint.

    Why sp.residue instead of sp.limit * (x-x0)?
      sp.residue uses SymPy's internal Laurent expansion,
      which correctly handles log factors in the denominator.

    Returns:
        LogarithmicSingularity  — when successfully calculated order and residue
        PoleSingularity         — fallback when log factor does not affect the order
        WheelNumber(⊥)          — fallback when computation is impossible
    """
    if len(variables) == 1:
        var, point = variables[0]
    else:
        var, point = variables[0]
        for v, p in variables:
            try:
                _, denom = sp.fraction(sp.cancel(expr))
                if sp.simplify(denom.subs(v, p)) == sp.S.Zero:
                    var, point = v, p
                    break
            except Exception:
                pass

    try:
        # Extract logarithmic factor and establish pole type.
        raw_denom = sp.denom(expr)
        denom_factors = sp.Mul.make_args(raw_denom)
        log_factor = None
        log_type = "landau"   # "landau" = log zeroes out | "ir" = log diverges

        for fac in denom_factors:
            if fac.has(sp.log):
                fac_val = sp.simplify(fac.subs({var: point}))
                if fac_val == sp.S.Zero:
                    log_factor = fac
                    log_type = "landau"
                    break
                else:
                    # IR case: log diverges, but algebraic factor also → 0
                    log_factor = fac
                    log_type = "ir"

        # Fallback for log_factor
        if log_factor is None:
            _, denom_fb = sp.fraction(sp.cancel(expr))
            log_atoms = [a for a in denom_fb.atoms(sp.log) if a.has(var)]
            log_factor = log_atoms[0] if log_atoms else sp.log(var)
            log_type = "ir"

        if verbose:
            print(f"  Log factor   : {log_factor}  [{log_type}]")

        # IR case: lim (x-x0)^n * f = 0 for every n
        # No algebraic order — logarithmically enhanced pole
        if log_type == "ir":
            try:
                # Calculate residue via sp.residue (might work)
                try:
                    residue = sp.residue(expr, var, point)
                    residue = sp.simplify(residue)
                except Exception:
                    residue = None

                hint = f"IR log pole: 1/({var}·log({var}/μ²)) — no alg. order"
                if verbose:
                    print(f"  IR Type      : logarithmically enhanced (lim x^n·f=0 ∀n)")
                    print(f"  Residue      : {residue}")
                    print(f"  Laurent      : {hint}")

                return LogarithmicSingularity(
                    expression=expr,
                    variables=variables,
                    singular_point=point,
                    pole_order=1,    # convention: report "effective" order
                    residue=residue,
                    log_factor=log_factor,
                    laurent_hint=hint,
                )
            except Exception as e:
                if verbose:
                    print(f"  IR fallback error: {e}")
                return W(BOTTOM)

        # Compute residue via SymPy (handles log in denominator)
        try:
            residue = sp.residue(expr, var, point)
            residue = sp.simplify(residue)
            if verbose:
                print(f"  sp.residue   : {residue}")
        except Exception as e:
            if verbose:
                print(f"  sp.residue error: {e}")
            residue = None

        # Check pole order via lim (x-x0)^n · f(x)
        pole_order = 1  # default — log poles are usually order 1
        for n in range(1, 6):
            try:
                factor = (var - point) ** n
                cand = sp.limit(factor * expr, var, point)
                if cand in (sp.oo, -sp.oo, sp.zoo, sp.nan):
                    continue
                if cand == sp.S.Zero:
                    continue
                pole_order = n
                if verbose:
                    print(f"  Pole order   : {n} (lim (x-x0)^{n}·f = {sp.simplify(cand)})")
                break
            except Exception:
                continue

        # Build Laurent hint
        if residue is not None and pole_order == 1:
            hint = f"({residue}) · 1/({var}-{point}) + O(1)"
        elif residue is not None:
            hint = f"({residue}) · 1/({var}-{point})^{pole_order} + O(1/({var}-{point})^{pole_order-1})"
        else:
            hint = f"log pole of order {pole_order} at {var}={point}"

        if verbose:
            print(f"  Laurent      : {hint}")

        return LogarithmicSingularity(
            expression=expr,
            variables=variables,
            singular_point=point,
            pole_order=pole_order,
            residue=residue,
            log_factor=log_factor,
            laurent_hint=hint,
        )

    except Exception as e:
        if verbose:
            print(f"  _compute_logarithmic_pole error: {e} → fallback ⊥")
        return W(BOTTOM)


def _has_essential_singularity(expr: sp.Basic, subs_dict: dict) -> bool:
    """Heuristic detection of essential singularities (exp(1/x), sin(1/x))."""
    try:
        expr_str = str(expr)
        # Simple heuristics — exp(1/x) at x=0
        for var, point in subs_dict.items():
            if point == sp.S.Zero:
                # Looking for 1/var patterns inside transcendental functions
                if expr.has(sp.exp) or expr.has(sp.sin) or expr.has(sp.cos):
                    inner_check = expr.subs(var, sp.Symbol('_test_eps'))
                    if f"1/_test_eps" in str(inner_check) or f"/_test_eps" in str(inner_check):
                        return True
        return False
    except Exception:
        return False


# ─── Taylor Expansion ─────────────────────────────────────────────────────────

def _try_taylor(
    expr:      sp.Basic,
    variables: list[tuple[sp.Symbol, sp.Basic]],
    max_order: int,
    verbose:   bool,
) -> Optional[tuple[sp.Basic, int, str]]:
    """
    Attempts to compute the limit value via Taylor expansion.

    Single variable strategy:
      Expand expr into a series around the point:
        expr = a_n*(x-x0)^n + a_{n+1}*(x-x0)^{n+1} + ...
      If lowest order n = 0 → limit = a_0 (finite).
      If n > 0 → limit = 0 (expression → 0).
      If n < 0 → genuine pole (confirmation of ⊥).

    Multi-variable strategy:
      Iterative substitution: first Taylor over x1, then over x2...
      If any yields a pole → ⊥.

    Returns:
        (limit_value, order, series_hint) if limit found
        None if a pole or cannot be calculated
    """
    if len(variables) == 1:
        return _taylor_single(expr, variables[0][0], variables[0][1], max_order, verbose)
    else:
        return _taylor_multivar(expr, variables, max_order, verbose)


def _taylor_single(
    expr:      sp.Basic,
    var:       sp.Symbol,
    point:     sp.Basic,
    max_order: int,
    verbose:   bool,
) -> Optional[tuple[sp.Basic, int, str]]:
    """Taylor for a single variable."""
    for order in range(1, max_order + 1):
        try:
            # Laurent/Taylor expansion up to `order`
            series = sp.series(expr, var, point, n=order + 2)

            if verbose and order == 1:
                print(f"    Laurent series: {series}")

            # Remove O(...) term
            series_no_O = series.removeO()

            # Compute limit — substitute var=point into the expansion
            limit_candidate = sp.simplify(series_no_O.subs(var, point))

            # Check if the result is finite
            if _is_finite_value(limit_candidate):
                # Build readable hint
                series_str = str(series).replace("O(", "O(").replace("\n", "")
                if len(series_str) > 80:
                    series_str = series_str[:77] + "..."
                return limit_candidate, order, series_str

            # If limit_candidate contains infinity → pole
            if limit_candidate in (sp.oo, sp.zoo, sp.nan, -sp.oo):
                if verbose:
                    print(f"    Order {order}: limit → {limit_candidate} (pole)")
                return None

        except (sp.core.power.PoleError, ZeroDivisionError):
            if verbose:
                print(f"    Order {order}: PoleError — genuine pole")
            return None
        except Exception as e:
            if verbose:
                print(f"    Order {order}: error ({e}), trying higher order")
            continue

    return None


def _taylor_multivar(
    expr:      sp.Basic,
    variables: list[tuple[sp.Symbol, sp.Basic]],
    max_order: int,
    verbose:   bool,
) -> Optional[tuple[sp.Basic, int, str]]:
    """
    Multi-variable Taylor — iterative.

    Strategy: substitute variables one by one, calculating
    Taylor series each time. If any yields a pole → None.

    Limitation: substitution order may matter.
    We try both orders and take the non-contradictory result.
    """
    from itertools import permutations

    best_result = None

    for perm in permutations(variables):
        result = _taylor_sequential(expr, list(perm), max_order, verbose)
        if result is not None:
            val, order, hint = result
            if _is_finite_value(val):
                # Check if other orders give the same result
                if best_result is None:
                    best_result = result
                else:
                    prev_val = best_result[0]
                    try:
                        if sp.simplify(val - prev_val) != sp.S.Zero:
                            if verbose:
                                print(f"    ⚠ Different substitution orders yield different limits!")
                                print(f"      {perm} → {val}")
                                print(f"      previous → {prev_val}")
                            # Choose the simpler value
                            best_result = result if sp.count_ops(val) < sp.count_ops(prev_val) else best_result
                    except Exception:
                        pass

    if best_result is not None:
        val, order, hint = best_result
        max_ord = max(order, len(variables))
        return val, max_ord, hint

    # Fallback: use sp.limit directly (for simple cases)
    return _sympy_limit_fallback(expr, variables, verbose)


def _taylor_sequential(
    expr:      sp.Basic,
    variables: list[tuple[sp.Symbol, sp.Basic]],
    max_order: int,
    verbose:   bool,
) -> Optional[tuple[sp.Basic, int, str]]:
    """Sequential Taylor substitution for a list of variables."""
    current_expr = expr
    max_used_order = 1

    for var, point in variables:
        result = _taylor_single(current_expr, var, point, max_order, verbose=False)
        if result is None:
            return None
        limit_val, order, hint = result
        max_used_order = max(max_used_order, order)
        # Substitute the limit value for the next step
        current_expr = sp.sympify(limit_val)

    if _is_finite_value(current_expr):
        return current_expr, max_used_order, f"multivariate ({len(variables)} variables)"
    return None


def _sympy_limit_fallback(
    expr:      sp.Basic,
    variables: list[tuple[sp.Symbol, sp.Basic]],
    verbose:   bool,
) -> Optional[tuple[sp.Basic, int, str]]:
    """
    Fallback: use sp.limit directly when Taylor fails.
    Works for simpler multivariable cases.
    """
    try:
        current = expr
        for var, point in variables:
            lim = sp.limit(current, var, point)
            if not _is_finite_value(lim):
                return None
            current = lim
        if _is_finite_value(current):
            if verbose:
                print(f"    Fallback sp.limit: {current}")
            return current, 0, "sp.limit (fallback)"
    except Exception as e:
        if verbose:
            print(f"    Fallback sp.limit error: {e}")
    return None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _is_finite_value(val) -> bool:
    """Is the value finite and well-defined."""
    if val is None:
        return False
    try:
        if val in (sp.oo, sp.zoo, sp.nan, -sp.oo):
            return False
        if hasattr(val, 'has'):
            if val.has(sp.oo) or val.has(sp.zoo) or val.has(sp.nan):
                return False
            # Check if it does not contain O(...)
            if val.has(sp.Order):
                return False
        return True
    except Exception:
        return False


# ─── Expression classification — full analysis ────────────────────────────────

def analyse_singularity(
    expr:      sp.Basic,
    variables: list[tuple[sp.Symbol, sp.Basic]],
    name:      str = "",
    max_order: int = 8,
    verbose:   bool = True,
) -> WheelCalcResult:
    """
    Full singularity analysis with report.
    Wrapper around wheel_limit with more detailed output.
    """
    if verbose:
        print(f"\n{'═'*62}")
        print(f"  ANALYSIS: {name or expr}")
        vars_str = ", ".join(f"{v}→{p}" for v,p in variables)
        print(f"  Point   : {vars_str}")
        print(f"{'─'*62}")

    result = wheel_limit(expr, variables, max_order=max_order, verbose=verbose)

    if verbose:
        print()
        if isinstance(result, RemovableSingularity):
            print(result.report())
            print(f"\n  ✓ RESULT: REMOVABLE singularity → lim = {result.limit_value}")
        elif isinstance(result, LogarithmicSingularity):
            print(result.report())
            res_str = f", res={result.residue}" if result.residue is not None else ""
            print(f"\n  ✗ RESULT: LOGARITHMIC POLE of order {result.pole_order}{res_str} → ⊥")
        elif isinstance(result, PoleSingularity):
            print(result.report())
            res_str = f", res={result.residue}" if result.residue is not None else ""
            print(f"\n  ✗ RESULT: POLE of order {result.pole_order}{res_str} → ⊥")
        elif result.is_bottom:
            print(f"  ✗ RESULT: IRREMOVABLE singularity → ⊥")
        else:
            print(f"  ✓ RESULT: regular point → {result}")
        print(f"{'═'*62}")

    return result


def classify_batch(
    cases: list[dict],
    max_order: int = 8,
    verbose: bool = True,
) -> list[dict]:
    """
    Analyzes a batch of expressions and returns a results table.

    cases: list of dictionaries with keys:
        name: str
        expr: sp.Basic
        variables: [(var, point), ...]
        expected_limit: sp.Basic or None (optional for verification)

    Returns:
        list of results with fields:
            name, result_type, wheel, limit, order, correct
    """
    results = []
    for case in cases:
        name = case.get('name', str(case['expr']))
        expr = case['expr']
        variables = case['variables']
        expected = case.get('expected_limit', None)

        result = wheel_limit(expr, variables, max_order=max_order, verbose=False)

        if isinstance(result, RemovableSingularity):
            r_type = "REMOVABLE"
            w_val = "⊥"
            lim = result.limit_value
            order = result.taylor_order
            correct = (expected is None) or _values_match(lim, expected)
        elif isinstance(result, LogarithmicSingularity):
            r_type = "LOG_POLE"
            w_val = "⊥"
            lim = sp.zoo
            order = result.pole_order
            correct = (expected is None) or (expected in (sp.oo, sp.zoo, -sp.oo))
        elif isinstance(result, PoleSingularity):
            r_type = "POLE"
            w_val = "⊥"
            lim = sp.zoo
            order = result.pole_order
            correct = (expected is None) or (expected in (sp.oo, sp.zoo, -sp.oo))
        elif result.is_bottom:
            r_type = "BOTTOM"
            w_val = "⊥"
            lim = sp.zoo
            order = None
            correct = (expected is None) or (expected in (sp.oo, sp.zoo, -sp.oo))
        else:
            r_type = "FINITE"
            w_val = result.value
            lim = result.value
            order = None
            correct = (expected is None) or _values_match(lim, expected)

        results.append({
            'name': name, 'result_type': r_type,
            'wheel': w_val, 'limit': lim,
            'order': order, 'correct': correct,
            'residue': result.residue if isinstance(result, PoleSingularity) else None,
        })

        if verbose:
            mark = "✓" if correct else "✗"
            if r_type == "REMOVABLE":
                type_str = f"{'REMOVABLE':12}"
            elif r_type == "LOG_POLE":
                res_str = f" res={result.residue}" if result.residue is not None else ""
                type_str = f"{'LOG_POLE['+str(order)+']':14}"
            elif r_type == "POLE":
                res_str = f" res={result.residue}" if result.residue is not None else ""
                type_str = f"{'POLE['+str(order)+']':12}"
            elif r_type == "BOTTOM":
                type_str = f"{'BOTTOM':12}"
            else:
                type_str = f"{'REGULAR':12}"
            lim_str = str(lim)[:20]
            exp_str = f" (expected: {expected})" if expected is not None and not correct else ""
            extra = ""
            if r_type == "POLE" and result.residue is not None:
                extra = f" | res={result.residue}"
            print(f"  {mark} {name:<40} {type_str} lim={lim_str}{extra}{exp_str}")

    return results


def _values_match(a, b) -> bool:
    try:
        return bool(sp.simplify(a - b) == sp.S.Zero)
    except Exception:
        return str(a) == str(b)


# ─── Tests and demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 62)
    print("  wheel_calculus.py — Wheel + analytical extension")
    print("  Tripartite division: finite | irremovable ⊥ | removable ⊥→val")
    print("═" * 62)

    x, m, p, r, r_s, omega, omega0 = sp.symbols(
        "x m p r r_s omega omega0", real=True
    )

    # ════════════════════════════════════════════════════════════
    # SECTION 1: Known counterexamples from DB (should → REMOVABLE)
    # ════════════════════════════════════════════════════════════
    print("\n▶  COUNTEREXAMPLES — removable singularities")
    print("   (Wheel yields ⊥, but limit exists)\n")

    cases_removable = [
        {
            "name": "sinc(x) = sin(x)/x",
            "expr": sp.sin(x) / x,
            "variables": [(x, sp.S.Zero)],
            "expected_limit": sp.S.One,
        },
        {
            "name": "sinc²(x) = sin²(x)/x²",
            "expr": sp.sin(x)**2 / x**2,
            "variables": [(x, sp.S.Zero)],
            "expected_limit": sp.S.One,
        },
        {
            "name": "(1 - cos(x))/x²",
            "expr": (1 - sp.cos(x)) / x**2,
            "variables": [(x, sp.S.Zero)],
            "expected_limit": sp.Rational(1, 2),
        },
        {
            "name": "(e^x - 1)/x",
            "expr": (sp.exp(x) - 1) / x,
            "variables": [(x, sp.S.Zero)],
            "expected_limit": sp.S.One,
        },
        {
            "name": "(x² - 1)/(x - 1)  at x=1",
            "expr": (x**2 - 1) / (x - 1),
            "variables": [(x, sp.S.One)],
            "expected_limit": sp.Integer(2),
        },
        {
            "name": "tan(x)/x",
            "expr": sp.tan(x) / x,
            "variables": [(x, sp.S.Zero)],
            "expected_limit": sp.S.One,
        },
        {
            "name": "(sin(3x))/(sin(5x))  at x=0",
            "expr": sp.sin(3*x) / sp.sin(5*x),
            "variables": [(x, sp.S.Zero)],
            "expected_limit": sp.Rational(3, 5),
        },
    ]

    print(f"  {'Equation':<42} {'Type':<14} {'lim':<10} {'OK?'}")
    print(f"  {'─'*42} {'─'*14} {'─'*10} {'─'*4}")
    classify_batch(cases_removable, max_order=8, verbose=True)

    # ════════════════════════════════════════════════════════════
    # SECTION 2: Genuine poles with residue analysis
    # ════════════════════════════════════════════════════════════
    print("\n▶  ALGEBRAIC POLES — order + residue (Cauchy)\n")

    cases_poles = [
        {
            "name": "1/x at x=0",
            "expr": 1/x,
            "variables": [(x, sp.S.Zero)],
            "expected_limit": sp.zoo,
        },
        {
            "name": "Scalar propagator 1/(p²-m²) at p=m",
            "expr": 1/(p**2 - m**2),
            "variables": [(p, m)],
            "expected_limit": sp.zoo,
        },
        {
            "name": "Schwarzschild g_rr at r=r_s",
            "expr": 1/(1 - r_s/r),
            "variables": [(r, r_s)],
            "expected_limit": sp.zoo,
        },
        {
            "name": "Resonance 1/(ω²-ω₀²) at ω=ω₀",
            "expr": 1/(omega**2 - omega0**2),
            "variables": [(omega, omega0)],
            "expected_limit": sp.zoo,
        },
        {
            "name": "Double pole 1/(p-m)²",
            "expr": 1/(p - m)**2,
            "variables": [(p, m)],
            "expected_limit": sp.zoo,
        },
        {
            "name": "Double pole 1/x²",
            "expr": 1/x**2,
            "variables": [(x, sp.S.Zero)],
            "expected_limit": sp.zoo,
        },
    ]

    print(f"  {'Equation':<42} {'Type':<14} {'res':<20} {'OK?'}")
    print(f"  {'─'*42} {'─'*14} {'─'*20} {'─'*4}")
    classify_batch(cases_poles, max_order=8, verbose=True)

    # Detailed propagator analysis
    print()
    analyse_singularity(
        1/(p**2 - m**2),
        [(p, m)],
        name="Scalar propagator 1/(p²-m²) — full analysis",
        max_order=6,
        verbose=True,
    )

    # ════════════════════════════════════════════════════════════
    # SECTION 3: Regular points (Wheel yields finite value)
    # ════════════════════════════════════════════════════════════
    print("\n▶  REGULAR POINTS — Wheel OK, calculus does not interfere\n")

    x_sym = sp.Symbol("x", positive=True)
    cases_regular = [
        {
            "name": "1/(p²+m²) at p=0, m=1",
            "expr": 1/(p**2 + m**2),
            "variables": [(p, sp.S.Zero), (m, sp.S.One)],
            "expected_limit": sp.S.One,
        },
        {
            "name": "sin(x)/x² at x=1",
            "expr": sp.sin(x)/x**2,
            "variables": [(x, sp.S.One)],
            "expected_limit": sp.sin(sp.S.One),
        },
    ]

    print(f"  {'Equation':<42} {'Type':<14} {'lim':<10} {'OK?'}")
    print(f"  {'─'*42} {'─'*14} {'─'*10} {'─'*4}")
    classify_batch(cases_regular, max_order=8, verbose=True)

    # ════════════════════════════════════════════════════════════
    # SECTION 4: Multivariable (m=0 and p=0 simultaneously)
    # ════════════════════════════════════════════════════════════
    print("\n▶  MULTIVARIABLE — two variables simultaneously\n")

    cases_multi = [
        {
            "name": "m·p/(m²+p²) at m=0, p=0",
            "expr": m*p / (m**2 + p**2),
            "variables": [(m, sp.S.Zero), (p, sp.S.Zero)],
            "expected_limit": None,  # limit depends on direction!
        },
        {
            "name": "(sin(m)+sin(p))/(m+p) at m=0, p=0",
            "expr": (sp.sin(m) + sp.sin(p)) / (m + p),
            "variables": [(m, sp.S.Zero), (p, sp.S.Zero)],
            "expected_limit": sp.S.One,
        },
    ]

    print(f"  {'Equation':<42} {'Type':<14} {'lim':<10} {'OK?'}")
    print(f"  {'─'*42} {'─'*14} {'─'*10} {'─'*4}")
    classify_batch(cases_multi, max_order=8, verbose=True)

    # ════════════════════════════════════════════════════════════
    # SECTION 5: Detailed report for sinc (verbose demo)
    # ════════════════════════════════════════════════════════════
    print("\n▶  Detailed analysis of sinc(x) at x=0 (verbose)\n")
    analyse_singularity(
        sp.sin(x)/x,
        [(x, sp.S.Zero)],
        name="sinc(x) = sin(x)/x",
        max_order=6,
        verbose=True,
    )

    # ════════════════════════════════════════════════════════════
    # SECTION 6: QCD Logarithmic Poles — Direction 4
    # ════════════════════════════════════════════════════════════
    print("\n▶  QCD LOGARITHMIC POLES — gluon propagator\n")
    print("   (log factor zeroes denominator — new type in wheel_calculus)\n")

    k2      = sp.Symbol("k2",      positive=True)
    alpha_s = sp.Symbol("alpha_s", positive=True)
    mu_r    = sp.Symbol("mu_r",    positive=True)

    gluon_prop = sp.Integer(1) / (k2 * (1 + alpha_s * sp.log(k2 / mu_r**2)))
    k2_ir      = sp.S.Zero
    k2_landau  = mu_r**2 * sp.exp(-sp.Integer(1) / alpha_s)

    cases_qcd = [
        {
            "name": "Gluon prop. — IR pole (k²=0)",
            "expr": gluon_prop,
            "variables": [(k2, k2_ir)],
            "expected_limit": sp.zoo,
        },
        {
            "name": "Gluon prop. — Landau pole",
            "expr": gluon_prop,
            "variables": [(k2, k2_landau)],
            "expected_limit": sp.zoo,
        },
    ]

    print(f"  {'Equation':<42} {'Type':<16} {'res':<20} {'OK?'}")
    print(f"  {'─'*42} {'─'*16} {'─'*20} {'─'*4}")
    classify_batch(cases_qcd, max_order=8, verbose=True)

    print()
    analyse_singularity(
        gluon_prop,
        [(k2, k2_landau)],
        name="QCD Gluon propagator — Landau pole (full analysis)",
        max_order=6,
        verbose=True,
    )

    # ════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════
    print("\n" + "═"*62)
    print("  SUMMARY — wheel_calculus.py")
    print("═"*62)
    print("""
  Five-fold division:
    WheelFinite(v)            → regular point (Wheel OK)
    WheelNumber(⊥)            → ⊥ without structure (fallback)
    RemovableSingularity      → Wheel=⊥ but lim=v (removable singularity)
    PoleSingularity           → algebraic pole: order + residue + Laurent
    LogarithmicSingularity    → log pole: log factor zeroes denominator

  Residue analysis:
    1/(p²-m²) at p=m    → POLE[1], res=1/(2m)
    1/x² at x=0         → POLE[2], res=N/A (Cauchy only for order 1)
    g_rr at r=r_s       → POLE[1], res=r_s
    Gluon @ Landau      → LOG_POLE[1], res=1/αs  ← NEW

  Connection with QFT:
    propagator residue = on-shell transition amplitude
    Landau pole residue = 1/αs (QCD coupling strength)
    Cauchy's theorem: ∮ f(z) dz = 2πi · Σ res(f, zₖ)

  Architecture:
    wheel_algebra.py  — axiomatic Wheel Algebra, unchanged
    wheel_calculus.py — analytical extension (this module)

  Key difference (important for the preprint!):
    Wheel Algebra:   sin(0)/0 = 0/0 = ⊥   (point algebra)
    wheel_calculus:  sin(x)/x at x→0  = 1  (Taylor expansion)
    Both are CORRECT — they answer different questions:
      ⊥ → "what happens at this point algebraically?"
      1 → "what is the limit value of mathematical analysis?"
""")