"""
singularity_finder.py — singularity scanner in SymPy expressions

Takes any SymPy expression and:
  1. Detects all potential singularity locations (division by zero)
  2. Classifies their type
  3. Suggests transformation into Wheel Algebra

Singularity types:
  - POLE        : expression → ∞ as parameter → critical value
  - ZERO_OVER_ZERO : 0/0 indeterminate form
  - ESSENTIAL   : essential singularity (e.g. e^(1/z) as z→0)
  - WHEEL_BOTTOM: in Wheel Algebra → ⊥
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import sympy as sp


# ─── Singularity types ────────────────────────────────────────────────────────

class SingularityType(Enum):
    POLE             = auto()   # 1/x^n as x→0
    ZERO_OVER_ZERO   = auto()   # 0/0 indeterminate form
    ESSENTIAL        = auto()   # oscillating, no limit exists
    LOGARITHMIC      = auto()   # log(0) → -∞
    SQUARE_ROOT      = auto()   # sqrt(x) for x<0 or x→0 in denominator
    WHEEL_BOTTOM     = auto()   # explicit /0 — immediately maps to ⊥
    UNKNOWN          = auto()   # unclassified


@dataclass
class Singularity:
    """Description of a single singularity within an expression."""
    expression:    sp.Basic           # subexpression containing the singularity
    variable:      sp.Symbol          # variable
    critical_value: sp.Basic          # critical value (e.g. r=0)
    sing_type:     SingularityType
    denominator:   Optional[sp.Basic] = None   # what is in the denominator
    wheel_result:  Optional[str]      = None   # what Wheel Algebra will yield
    limit_classical: Optional[str]   = None   # classical limit
    note:          str                = ""

    def __str__(self) -> str:
        lines = [
            f"  Expression : {self.expression}",
            f"  Variable   : {self.variable} → {self.critical_value}",
            f"  Type       : {self.sing_type.name}",
        ]
        if self.denominator is not None:
            lines.append(f"  Denominator: {self.denominator}")
        if self.wheel_result:
            lines.append(f"  Wheel      : {self.wheel_result}")
        if self.limit_classical:
            lines.append(f"  Class. lim : {self.limit_classical}")
        if self.note:
            lines.append(f"  Note       : {self.note}")
        return "\n".join(lines)


@dataclass
class ScanResult:
    """Result of scanning the entire expression."""
    original:     sp.Basic
    variables:    list[sp.Symbol]
    singularities: list[Singularity] = field(default_factory=list)

    @property
    def has_singularities(self) -> bool:
        return len(self.singularities) > 0

    def summary(self) -> str:
        lines = [
            "─" * 60,
            f"EXPRESSION   : {self.original}",
            f"VARIABLES    : {self.variables}",
            f"SINGULARITIES: {len(self.singularities)}",
            "─" * 60,
        ]
        if not self.singularities:
            lines.append("  No singularities detected.")
        for i, s in enumerate(self.singularities, 1):
            lines.append(f"\n[{i}] {s.sing_type.name}")
            lines.append(str(s))
        lines.append("─" * 60)
        return "\n".join(lines)


# ─── Scanner ──────────────────────────────────────────────────────────────────

class SingularityFinder:
    """
    Main singularity scanner.

    Usage:
        finder = SingularityFinder()
        result = finder.scan(expr, variables=[r])
        print(result.summary())
    """

    def scan(
        self,
        expr: sp.Basic,
        variables: Optional[list[sp.Symbol]] = None,
        critical_values: Optional[list[sp.Basic]] = None,
    ) -> ScanResult:
        """
        Scans an expression for singularities.

        Args:
            expr:            SymPy expression
            variables:       List of variables to check (auto-detect if None)
            critical_values: Critical values to check (default: [0])
        """
        expr = sp.sympify(expr)

        if variables is None:
            variables = sorted(expr.free_symbols, key=str)

        if critical_values is None:
            critical_values = [sp.S.Zero]

        result = ScanResult(original=expr, variables=list(variables))

        for var in variables:
            for crit in critical_values:
                sings = self._analyze(expr, var, crit)
                result.singularities.extend(sings)

        return result

    # ── Analysis of a single variable at a single critical value ─────────────

    def _analyze(
        self,
        expr: sp.Basic,
        var: sp.Symbol,
        crit: sp.Basic,
    ) -> list[Singularity]:
        results = []

        # 1. Collect all subexpressions
        subexprs = self._collect_subexpressions(expr)

        for sub in subexprs:
            # 2. Look for denominators
            numer, denom = sp.fraction(sub)

            if denom != sp.S.One and denom != sp.S.NegativeOne:
                sing = self._classify_fraction(sub, numer, denom, var, crit)
                if sing is not None:
                    results.append(sing)

            # 3. Look for log(0)
            log_sing = self._check_log(sub, var, crit)
            if log_sing:
                results.append(log_sing)

        # 4. Deduplication
        seen = set()
        unique = []
        for s in results:
            key = (str(s.expression), str(s.variable), str(s.critical_value))
            if key not in seen:
                seen.add(key)
                unique.append(s)

        return unique

    def _collect_subexpressions(self, expr: sp.Basic) -> list[sp.Basic]:
        """Recursively collects all subexpressions."""
        subexprs = set()
        subexprs.add(expr)

        def walk(e):
            for arg in e.args:
                subexprs.add(arg)
                walk(arg)

        walk(expr)
        return list(subexprs)

    def _classify_fraction(
        self,
        sub: sp.Basic,
        numer: sp.Basic,
        denom: sp.Basic,
        var: sp.Symbol,
        crit: sp.Basic,
    ) -> Optional[Singularity]:
        """Classifies a fraction for singularities as var → crit."""

        # Substitute critical value into the denominator
        denom_at_crit = denom.subs(var, crit)

        try:
            denom_zero = bool(sp.simplify(denom_at_crit) == 0)
        except Exception:
            denom_zero = False

        if not denom_zero:
            return None  # Denominator ≠ 0, no singularity

        # Denominator zeroes out — determine the type
        numer_at_crit = sp.simplify(numer.subs(var, crit))

        try:
            numer_zero = bool(numer_at_crit == 0)
        except Exception:
            numer_zero = False

        # Compute the classical limit
        limit_str = self._compute_limit(sub, var, crit)

        if numer_zero:
            # 0/0 — indeterminate form
            sing_type = SingularityType.ZERO_OVER_ZERO
            wheel_result = "⊥  (0·/0 in wheel)"
            note = "Indeterminate form — L'Hôpital or Taylor expansion"
        else:
            # x/0 where x ≠ 0 — pole
            sing_type = SingularityType.POLE

            # Check the pole order
            order = self._pole_order(denom, var, crit)
            wheel_result = f"⊥  ({numer_at_crit}·/0 = ⊥ in wheel)"
            note = f"Pole of order {order}"

        return Singularity(
            expression=sub,
            variable=var,
            critical_value=crit,
            sing_type=sing_type,
            denominator=denom,
            wheel_result=wheel_result,
            limit_classical=limit_str,
            note=note,
        )

    def _check_log(
        self,
        sub: sp.Basic,
        var: sp.Symbol,
        crit: sp.Basic,
    ) -> Optional[Singularity]:
        """Checks if the subexpression is log(f) where f→0."""
        if not isinstance(sub, sp.log):
            return None

        arg = sub.args[0]
        arg_at_crit = sp.simplify(arg.subs(var, crit))

        try:
            is_zero = bool(arg_at_crit == 0)
        except Exception:
            is_zero = False

        if not is_zero:
            return None

        return Singularity(
            expression=sub,
            variable=var,
            critical_value=crit,
            sing_type=SingularityType.LOGARITHMIC,
            denominator=None,
            wheel_result="⊥  (log(0) = ⊥ in wheel)",
            limit_classical="-∞",
            note="log(0) — logarithmic singularity",
        )

    def _compute_limit(self, expr: sp.Basic, var: sp.Symbol, crit: sp.Basic) -> str:
        """Computes classical limit (as string)."""
        try:
            lim = sp.limit(expr, var, crit)
            return str(lim)
        except Exception:
            try:
                lim_plus  = sp.limit(expr, var, crit, "+")
                lim_minus = sp.limit(expr, var, crit, "-")
                if lim_plus == lim_minus:
                    return str(lim_plus)
                return f"{lim_minus} (left) / {lim_plus} (right)"
            except Exception:
                return "cannot be calculated"

    def _pole_order(self, denom: sp.Basic, var: sp.Symbol, crit: sp.Basic) -> int:
        """Estimates the pole order based on the denominator's expansion."""
        try:
            series = sp.series(denom, var, crit, n=6)
            for n in range(1, 6):
                coeff = series.coeff(var - crit, n)
                if coeff != 0:
                    return n
        except Exception:
            pass
        return 1


# ─── Built-in predefined physical equations ───────────────────────────────────

def scan_schwarzschild() -> ScanResult:
    """Schwarzschild Metric — singularity at r=0 and r=r_s."""
    r, r_s, c, t = sp.symbols("r r_s c t", positive=True)
    M, G = sp.symbols("M G", positive=True)

    # ds^2 = -(1 - r_s/r)c^2 dt^2 + (1 - r_s/r)^{-1} dr^2 + ...
    # Key metric factor:
    f = 1 - r_s / r          # zeroes out at r = r_s
    g_tt  = -f * c**2        # g_tt component
    g_rr  = 1 / f            # g_rr component — singularity at r = r_s

    finder = SingularityFinder()
    result_rrs = finder.scan(g_rr, variables=[r], critical_values=[r_s, sp.S.Zero])
    result_rrs.original = sp.Symbol("g_rr (Schwarzschild)")
    return result_rrs


def scan_friedmann() -> ScanResult:
    """Friedmann equations — singularity at a=0 (Big Bang)."""
    a, H, k, Lambda, rho, G, c = sp.symbols("a H k Lambda rho G c")

    # H^2 = (8πGρ/3) - kc^2/a^2 + Λc^2/3
    # Component with singularity:
    term = k * c**2 / a**2

    finder = SingularityFinder()
    result = finder.scan(term, variables=[a], critical_values=[sp.S.Zero])
    result.original = sp.Symbol("Friedmann: k·c²/a²")
    return result


def scan_feynman_propagator() -> ScanResult:
    """Feynman propagator — pole on the real axis."""
    p, m, epsilon = sp.symbols("p m epsilon", real=True, positive=True)

    # D_F(p) = i / (p^2 - m^2 + iε)
    # In limit ε→0: pole at p^2 = m^2
    denom = p**2 - m**2
    propagator = 1 / denom

    finder = SingularityFinder()
    result = finder.scan(propagator, variables=[p], critical_values=[m, -m])
    result.original = sp.Symbol("Feynman propagator: 1/(p²-m²)")
    return result


# ─── CLI / demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  WHEELPHYSICS — Singularity Scanner")
    print("═" * 60)

    # ── Test 1: simple expression
    print("\n▶  Test 1: expression 1/x + 1/x^2")
    x = sp.Symbol("x")
    expr = 1/x + 1/x**2
    finder = SingularityFinder()
    print(finder.scan(expr, [x]).summary())

    # ── Test 2: Schwarzschild metric
    print("\n▶  Test 2: Schwarzschild g_rr")
    print(scan_schwarzschild().summary())

    # ── Test 3: Friedmann
    print("\n▶  Test 3: Friedmann Equation")
    print(scan_friedmann().summary())

    # ── Test 4: Feynman Propagator
    print("\n▶  Test 4: Feynman Propagator")
    print(scan_feynman_propagator().summary())