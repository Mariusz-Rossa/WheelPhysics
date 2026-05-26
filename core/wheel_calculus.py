# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
wheel_calculus.py — Wheel Algebra z rozszerzeniem analitycznym

Wheel Algebra (wheel_algebra.py) jest algebrą punktową:
  każda forma 0/0 → ⊥, bez wyjątku.

Ten moduł dodaje warstwę analityczną: gdy Wheel zwróci ⊥, diagnozuje
typ osobliwości i — dla biegunów — oblicza residuum i rząd bieguna.

Pięciopodział wyników (wheel_limit):
  ┌──────────────────────────────────────────────────────────────┐
  │ WheelFinite(value)        — regularny punkt (Wheel OK)       │
  │ WheelBottom()             — nieusuwalna ⊥ (nieokreślony typ) │
  │ RemovableSingularity      — usuwalna: Wheel=⊥, lim=val      │
  │ PoleSingularity           — biegun alg.: rząd + res + Laurent│
  │ LogarithmicSingularity    — biegun log.: log zeruje mianownik│
  └──────────────────────────────────────────────────────────────┘

Formalny system typów — SingularityType (enum):
  REGULAR      → punkt regularny (Wheel skończony)
  REMOVABLE    → osobliwość usuwalna (Taylor)
  POLE_SIMPLE  → biegun prosty rząd=1 (res Cauchy zdefiniowane)
  POLE_HIGHER  → biegun wyższego rzędu (res N/A)
  ESSENTIAL    → osobliwość istotna (Picard) — wymaga zewnętrznej analizy
  LOGARITHMIC  → ⊥  biegun logarytmiczny: czynnik log zeruje mianownik (QCD)
  BRANCH_POINT → punkt rozgałęzienia — wymaga zewnętrznej analizy
  COORDINATE   → artefakt układu współrzędnych — wymaga niezmienników
  PHYSICAL     → potwierdzona osobliwość fizyczna — wymaga niezmienników
  COMPLEX_POLE → biegun zespolony (poza R) — otwarte pytanie badawcze
  UNKNOWN      → fallback gdy klasyfikacja nie powiodła się

Architektura (celowa):
  wheel_algebra.py    — aksjomatyczna, czysta, bez zmian
  wheel_calculus.py   — osobny moduł, rozszerzenie analityczne
  consistency_checker — testuje oba, porównuje wyniki

To rozróżnienie jest ważne dla preprintu:
  Wheel Algebra ≠ teoria granic
  Wheel + Calculus = pełny aparat do analizy osobliwości

Referencja: Carlström (2004) "Wheels — On Division by Zero"
"""

from __future__ import annotations

import sys, os
# Obsługa obu wariantów struktury (flat repo i core/ subpackage)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# Jeśli repo używa struktury core/, stwórz symlink-like namespace
try:
    from wheel_number import WheelNumber, BOTTOM, W, _coerce
except ImportError:
    from core.wheel_number import WheelNumber, BOTTOM, W, _coerce  # type: ignore

# Tworzymy tymczasowy moduł 'core' wskazujący na bieżący katalog
# żeby wheel_algebra.py i sympy_extension.py mogły robić 'from core.xxx'
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


# ─── SingularityType — formalny typ osobliwości ────────────────────────────────

from enum import Enum, auto

class SingularityType(Enum):
    """
    Formalny typ osobliwości — wynik klasyfikacji wheel_calculus.

    Hierarchia (od "najbardziej regularna" do "najbardziej osobliwa"):

      REGULAR          — punkt regularny: Wheel daje skończoną wartość,
                         brak osobliwości. Calculus nie ingeruje.

      REMOVABLE        — osobliwość usuwalna: Wheel daje ⊥ (forma 0/0),
                         ale granica analityczna istnieje i jest skończona.
                         Rozwinięcie Taylora usuwa osobliwość.
                         Przykład: sin(x)/x przy x=0, lim=1

      POLE             — biegun algebraiczny: Wheel daje ⊥, mianownik → 0,
                         licznik ≠ 0. Granica = ±∞. Posiada rząd i residuum.
                         Przykład: 1/(p²-m²) przy p=m, res=1/(2m)

      POLE_SIMPLE      — biegun prosty (rząd=1): szczególny przypadek POLE.
                         Residuum zdefiniowane w sensie Cauchy'ego.
                         Najważniejszy fizycznie — propagatory QFT.

      POLE_HIGHER      — biegun wyższego rzędu (rząd≥2): residuum N/A.
                         Przykład: 1/r² przy r=0 (rząd=2).

      ESSENTIAL        — osobliwość istotna (essential singularity):
                         nie biegun, nie usuwalna — zachowanie chaotyczne
                         w otoczeniu punktu (twierdzenie Picarda-Weierstrassa).
                         Przykład: exp(1/z) przy z=0.
                         Wheel daje ⊥; Taylor i Laurent nie pomagają.

      LOGARITHMIC      — osobliwość logarytmiczna: dywergencja log-typu.
                         Przykład: propagator gluonu QCD z poprawką pętlową.
                         Taylor nie działa; wymaga szeregu asymptotycznego.

      BRANCH_POINT     — punkt rozgałęzienia: funkcja wielowartościowa
                         (np. √z, log z przy z=0). Nie jest biegunem.

      COORDINATE       — artefakt układu współrzędnych, nie fizyczna osobliwość.
                         Wykrywana przez niezmienniki (np. K Kretschmanna).
                         Przykład: g_rr przy r=r_s — biegun w Schwarzschildzie,
                         ale K(r_s) skończone.

      PHYSICAL         — potwierdzona fizyczna osobliwość (nie artefakt).
                         Wheel=⊥ I niezmiennik skalarny też ⊥.
                         Przykład: K przy r=0 (osobliwość krzywizny).

      COMPLEX_POLE     — biegun leżący poza osią rzeczywistą (Im(z₀) ≠ 0).
                         Wheel (operując na R) nie trafia w niego przez
                         podstawienie rzeczywiste. Otwarte pytanie badawcze.
                         Przykład: Green oscylatora z tłumieniem γ>0.

      UNKNOWN          — typ nierozpoznany — fallback gdy klasyfikacja
                         nie powiodła się lub wyrażenie zbyt złożone.
    """

    REGULAR       = auto()   # ✓  punkt regularny
    REMOVABLE     = auto()   # ⊥→v  osobliwość usuwalna
    POLE          = auto()   # ⊥  biegun (ogólny)
    POLE_SIMPLE   = auto()   # ⊥  biegun prosty (rząd=1, res zdefiniowane)
    POLE_HIGHER   = auto()   # ⊥  biegun wyższego rzędu (rząd≥2)
    ESSENTIAL     = auto()   # ⊥  osobliwość istotna (Picard)
    LOGARITHMIC   = auto()   # ⊥  dywergencja logarytmiczna
    BRANCH_POINT  = auto()   # ⊥  punkt rozgałęzienia
    COORDINATE    = auto()   # ⊥* artefakt układu współrzędnych
    PHYSICAL      = auto()   # ⊥  potwierdzona osobliwość fizyczna
    COMPLEX_POLE  = auto()   # ⊥? biegun zespolony (poza R)
    UNKNOWN       = auto()   # ?  nierozpoznany

    @property
    def is_genuine_singularity(self) -> bool:
        """Czy to prawdziwa (nieusuwalna) osobliwość?"""
        return self not in (
            SingularityType.REGULAR,
            SingularityType.REMOVABLE,
            SingularityType.COORDINATE,
        )

    @property
    def has_residue(self) -> bool:
        """Czy residuum Cauchy'ego jest zdefiniowane?"""
        return self == SingularityType.POLE_SIMPLE

    @property
    def short(self) -> str:
        """Krótki label do tabel i logów."""
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
    Zwraca SingularityType dla dowolnego wyniku wheel_calculus.

    Mapowanie:
      WheelNumber (skończony)    → REGULAR
      WheelNumber (⊥, BOTTOM)   → UNKNOWN  (brak dalszej informacji)
      RemovableSingularity       → REMOVABLE
      PoleSingularity rząd=1     → POLE_SIMPLE
      PoleSingularity rząd≥2     → POLE_HIGHER

    Uwaga: COORDINATE, PHYSICAL, ESSENTIAL, LOGARITHMIC, COMPLEX_POLE
    wymagają dodatkowego kontekstu (np. niezmienniki skalarne, analiza
    asymptotyczna) i nie mogą być wyprowadzone automatycznie z samego
    wyniku wheel_limit. Przypisywane ręcznie lub przez dedykowane moduły.
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


# ─── Typy wyników ─────────────────────────────────────────────────────────────

@dataclass
class RemovableSingularity:
    """
    Osobliwość usuwalna — Wheel daje ⊥, ale granica analityczna istnieje.

    Przykład:  sin(x)/x przy x=0
      wheel_result  = ⊥
      limit_value   = 1
      taylor_order  = 1   (sin(x) ≈ x — x³/6, więc sin(x)/x ≈ 1 - x²/6 → 1)
      variables     = [(x, 0)]
    """
    wheel_result:  WheelNumber          # zawsze ⊥
    limit_value:   sp.Basic             # wartość granicy
    taylor_order:  int                  # rząd rozwinięcia który dał wynik
    variables:     list[tuple]          # [(var, point), ...] — może być wiele
    expression:    sp.Basic             # oryginalne wyrażenie
    series_hint:   str = ""             # czytelne rozwinięcie Taylora

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
            f"rząd Taylora = {self.taylor_order})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def report(self) -> str:
        """Czytelny raport dla użytkownika / logu."""
        vars_str = ", ".join(f"{v}→{p}" for v, p in self.variables)
        lines = [
            f"  Wyrażenie : {self.expression}",
            f"  Punkt     : {vars_str}",
            f"  Wheel     : ⊥  (forma 0/0 — podstawienie punktowe)",
            f"  Granica   : {self.limit_value}  ← wynik wheel_calculus",
            f"  Typ       : OSOBLIWOŚĆ USUWALNA",
            f"  Rząd      : Taylor do rzędu {self.taylor_order}",
        ]
        if self.series_hint:
            lines.append(f"  Taylor    : {self.series_hint}")
        return "\n".join(lines)


@dataclass
class PoleSingularity:
    """
    Biegun algebraiczny — Wheel daje ⊥, mianownik zeruje się, licznik nie.

    Zawiera pełną lokalną strukturę bieguna:
      - rząd bieguna (1=prosty, 2=podwójny, ...)
      - residuum (dla rzędu 1: współczynnik przy 1/(x-x0))
      - rozwinięcie Laurenta wokół bieguna

    Przykład: 1/(p²-m²) przy p=m
      pole_order   = 1
      residue      = 1/(2m)
      laurent_hint = "1/(2m) · 1/(p-m) + O(1)"

    Przykład: 1/r² przy r=0
      pole_order   = 2
      residue      = None  (residuum tylko dla rzędu 1)
      principal_part = "1/r²"

    Związek z QFT:
      Residuum przy bieguniepropagatu = amplituda przejścia na powłoce masowej.
      Twierdzenie o residuach Cauchy'ego: ∮ f(z) dz = 2πi · Σ res(f, zₖ)
    """
    expression:    sp.Basic          # oryginalne wyrażenie
    variables:     list[tuple]       # [(var, point), ...]
    pole_order:    int               # rząd bieguna
    residue:       Optional[sp.Basic]  # residuum (tylko rząd=1)
    laurent_coeff: sp.Basic          # lim (x-x0)^n · f(x) — współczynnik główny
    laurent_hint:  str = ""          # czytelny opis rozwinięcia Laurenta

    @property
    def is_bottom(self) -> bool:
        return True   # ⊥ w sensie Wheel — biegun to nieusuwalna osobliwość

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
            f"rząd={self.pole_order}{res_str}, "
            f"przy [{vars_str}])"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def report(self) -> str:
        """Czytelny raport dla użytkownika / logu."""
        vars_str = ", ".join(f"{v}→{p}" for v, p in self.variables)
        lines = [
            f"  Wyrażenie    : {self.expression}",
            f"  Punkt        : {vars_str}",
            f"  Wheel        : ⊥  (biegun — mianownik→0, licznik≠0)",
            f"  Typ          : BIEGUN ALGEBRAICZNY (rząd {self.pole_order})",
            f"  Rząd bieguna : {self.pole_order}",
        ]
        if self.residue is not None:
            lines.append(f"  Residuum     : {self.residue}")
        else:
            lines.append(f"  Residuum     : N/A (tylko dla bieguna rzędu 1)")
        lines.append(f"  Wsp. główny  : {self.laurent_coeff}  [= lim (x-x₀)ⁿ·f(x)]")
        if self.laurent_hint:
            lines.append(f"  Laurent      : {self.laurent_hint}")
        lines.append(
            f"  QFT          : res = amplituda on-shell (twierdzenie Cauchy'ego)"
            if self.pole_order == 1 else
            f"  QFT          : biegun wyższego rzędu — anomalna dywergencja"
        )
        return "\n".join(lines)


@dataclass
class LogarithmicSingularity:
    """
    Biegun logarytmiczny — Wheel daje ⊥, mianownik zeruje się przez czynnik log.

    Różnica względem PoleSingularity:
      PoleSingularity  — mianownik to wielomian: (x - x₀)ⁿ
      LogarithmicSingularity — mianownik zawiera log(x/μ²) który zeruje się

    Przykład: propagator gluonu QCD 1/(k²·(1 + αs·log(k²/μ²)))
      singular_point     = μ²·exp(-1/αs)   ← biegun Landaua
      pole_order         = 1               ← biegun prosty (log zeruje liniowo)
      residue            = 1/αs            ← obliczone przez sp.residue
      log_factor         = "1 + αs·log(k²/μ²)"  ← czynnik który się zeruje
      laurent_hint       = "(1/αs) · 1/(k²-k²_L) + O(1)"

    Fizyczne znaczenie:
      Biegun Landaua QCD pojawia się w skali konfinementu (~ΛQCD).
      Odpowiednik w QED jest niefizyczny (10^280 GeV).
      Residuum 1/αs odzwierciedla siłę sprzężenia QCD w punkcie osobliwym.

    Związek z Wheel:
      Wheel(expr, k2=k2_L) = ⊥  (mianownik → 0, licznik = 1)
      Typ: LOGARITHMIC — identyfikowany przez obecność log(var) w mianowniku.
    """
    expression:    sp.Basic           # oryginalne wyrażenie
    variables:     list[tuple]        # [(var, point), ...]
    singular_point: sp.Basic          # wartość zmiennej w biegunieLandaua
    pole_order:    int                # rząd bieguna (zazwyczaj 1)
    residue:       Optional[sp.Basic] # residuum (sp.residue)
    log_factor:    sp.Basic           # czynnik logarytmiczny który się zeruje
    laurent_hint:  str = ""           # czytelny opis rozwinięcia

    @property
    def is_bottom(self) -> bool:
        return True  # ⊥ w sensie Wheel — biegun logarytmiczny jest niesuwalny

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
            f"rząd={self.pole_order}{res_str}, "
            f"przy [{vars_str}], log_factor={self.log_factor})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def report(self) -> str:
        """Czytelny raport dla użytkownika / logu."""
        vars_str = ", ".join(f"{v}→{p}" for v, p in self.variables)
        lines = [
            f"  Wyrażenie    : {self.expression}",
            f"  Punkt        : {vars_str}",
            f"  Punkt osobl. : {self.singular_point}",
            f"  Wheel        : ⊥  (biegun log — czynnik log zeruje mianownik)",
            f"  Typ          : BIEGUN LOGARYTMICZNY (rząd {self.pole_order})",
            f"  Czynnik log  : {self.log_factor}",
            f"  Rząd bieguna : {self.pole_order}",
        ]
        if self.residue is not None:
            lines.append(f"  Residuum     : {self.residue}")
        else:
            lines.append(f"  Residuum     : N/A")
        if self.laurent_hint:
            lines.append(f"  Laurent      : {self.laurent_hint}")
        lines.append(
            f"  QCD          : residuum = 1/αs — siła sprzężenia w punkcie Landaua"
            if self.pole_order == 1 else
            f"  QCD          : biegun logarytmiczny wyższego rzędu — anomalia"
        )
        return "\n".join(lines)


# Unia typów zwracanych przez wheel_calculus
WheelCalcResult = Union[WheelNumber, RemovableSingularity, PoleSingularity, LogarithmicSingularity]


# ─── Główna funkcja ────────────────────────────────────────────────────────────

def wheel_limit(
    expr:      sp.Basic,
    variables: list[tuple[sp.Symbol, sp.Basic]],
    max_order: int = 8,
    verbose:   bool = False,
) -> WheelCalcResult:
    """
    Główna funkcja wheel_calculus — ujednolicony interfejs.

    Algorytm:
      1. Sprawdź Wheel (wheel_subs). Jeśli skończony → zwróć WheelNumber.
      2. Jeśli ⊥ → zbadaj przyczynę:
         a. Czy to forma 0/0? (licznik I mianownik → 0)
         b. Czy to prawdziwy biegun? (licznik ≠ 0, mianownik → 0)
      3. Dla formy 0/0 → próbuj rozwinięcia Taylora (kolejno rzędy 1..max_order).
      4. Jeśli Taylor da skończoną wartość → RemovableSingularity.
      5. Jeśli Taylor nie pomaga → WheelNumber(BOTTOM) (nieusuwalna).

    Args:
        expr:      wyrażenie SymPy
        variables: lista par (symbol, wartość_graniczna)
                   np. [(x, 0)] lub [(m, 0), (p, 0)]
        max_order: maksymalny rząd rozwinięcia Taylora (default 8)
        verbose:   czy drukować kroki diagnostyczne

    Returns:
        WheelNumber           — gdy punkt regularny lub nieusuwalna ⊥
        RemovableSingularity  — gdy osobliwość usuwalna (Wheel=⊥, lim=val)
    """
    subs_dict = {var: point for var, point in variables}

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  wheel_limit: {expr}")
        print(f"  przy: {', '.join(f'{v}→{p}' for v,p in variables)}")

    # ── Krok 1: Wynik Wheel ────────────────────────────────────────────────
    wheel_result = wheel_subs(expr, subs_dict)

    if not wheel_result.is_bottom:
        if verbose:
            print(f"  Wheel: {wheel_result}  (regularny punkt, brak działania)")
        return wheel_result

    if verbose:
        print(f"  Wheel: ⊥  — sprawdzam typ osobliwości...")

    # ── Krok 2: Diagnoza — 0/0 czy prawdziwy biegun? ──────────────────────
    singularity_type = _classify_singularity(expr, subs_dict, verbose)

    if singularity_type == "logarithmic_pole":
        if verbose:
            print(f"  Typ: BIEGUN LOGARYTMICZNY (czynnik log zeruje mianownik) → analiza log")
        return _compute_logarithmic_pole(expr, variables, verbose)

    if singularity_type == "pole":
        if verbose:
            print(f"  Typ: PRAWDZIWY BIEGUN (licznik≠0, mianownik→0) → analiza residuum")
        return _compute_pole(expr, variables, verbose)

    if singularity_type == "essential":
        if verbose:
            print(f"  Typ: OSOBLIWOŚĆ ISTOTNA (np. exp(1/x)) → ⊥ nieusuwalna")
        return W(BOTTOM)

    if singularity_type == "unknown":
        if verbose:
            print(f"  Typ: nieznany — próbuję Taylora jako fallback")

    # singularity_type == "removable_candidate" lub "unknown"
    if verbose:
        print(f"  Typ: KANDYDAT na osobliwość usuwalną (forma 0/0) → próbuję Taylor")

    # ── Krok 3: Rozwinięcie Taylora ────────────────────────────────────────
    result = _try_taylor(expr, variables, max_order, verbose)

    if result is not None:
        limit_val, order, series_hint = result
        if verbose:
            print(f"  Taylor rząd {order}: lim = {limit_val}  → USUWALNA ✓")
        return RemovableSingularity(
            wheel_result=wheel_result,
            limit_value=limit_val,
            taylor_order=order,
            variables=variables,
            expression=expr,
            series_hint=series_hint,
        )

    # ── Krok 5: Taylor nie pomógł ─────────────────────────────────────────
    if verbose:
        print(f"  Taylor do rzędu {max_order}: brak skończonej granicy → ⊥ nieusuwalna")
    return W(BOTTOM)


# ─── Klasyfikacja osobliwości ──────────────────────────────────────────────────

def _classify_singularity(
    expr: sp.Basic,
    subs_dict: dict,
    verbose: bool = False,
) -> str:
    """
    Klasyfikuje typ osobliwości przy danym podstawieniu.

    Zwraca:
        "removable_candidate" — licznik I mianownik → 0 (forma 0/0)
        "pole"                — licznik ≠ 0, mianownik → 0
        "essential"           — osobliwość istotna (exp(1/x))
        "unknown"             — nie da się sklasyfikować
    """
    try:
        # Rozdziel na licznik i mianownik
        numer, denom = sp.fraction(sp.cancel(expr))

        numer_sub = sp.simplify(numer.subs(subs_dict))
        denom_sub = sp.simplify(denom.subs(subs_dict))

        numer_is_zero = (numer_sub == sp.S.Zero) or (
            hasattr(numer_sub, 'is_zero') and numer_sub.is_zero
        )
        denom_is_zero = (denom_sub == sp.S.Zero) or (
            hasattr(denom_sub, 'is_zero') and denom_sub.is_zero
        ) or denom_sub in (sp.zoo, sp.nan, sp.oo, -sp.oo)

        # Gdy subs daje nan (np. 0*log(0)), użyj sp.limit jako fallback
        if denom_sub is sp.nan or denom_sub == sp.nan:
            try:
                if len(subs_dict) == 1:
                    v, pt = list(subs_dict.items())[0]
                    denom_lim = sp.limit(denom, v, pt)
                    denom_is_zero = (denom_lim == sp.S.Zero)
                    if verbose:
                        print(f"    mianownik@subs=nan → sp.limit={denom_lim} {'(→0)' if denom_is_zero else ''}")
            except Exception:
                pass

        if verbose:
            print(f"    licznik po podstawieniu : {numer_sub} {'(=0)' if numer_is_zero else ''}")
            print(f"    mianownik po podstawieniu: {denom_sub} {'(=0)' if denom_is_zero else ''}")

        if numer_is_zero and denom_is_zero:
            return "removable_candidate"
        elif not numer_is_zero and denom_is_zero:
            # Sprawdź czy biegun pochodzi z czynnika logarytmicznego.
            # Używamy sp.denom(expr) zamiast sp.fraction(sp.cancel(expr)) —
            # sp.cancel rozszerza mianownik i niszczy strukturę czynnikową.
            # sp.denom zachowuje czynniki: k2*(1 + αs·log(k2/μ²)) → [k2, 1+αs·log(...)].
            #
            # Dwa przypadki bieguna logarytmicznego:
            #   (A) Czynnik log zeruje się: 1+αs·log(k2/μ²)=0 → biegun Landaua
            #   (B) Czynnik log dywerguje i dominuje: log(k2/μ²)→-∞ przy k2→0
            #       Wtedy k2^n * f → 0 dla każdego n (brak rzędu algebraicznego)
            try:
                raw_denom = sp.denom(expr)
                denom_factors = sp.Mul.make_args(raw_denom)
                for fac in denom_factors:
                    if fac.has(sp.log):
                        # Przypadek A: czynnik log zeruje się w punkcie
                        fac_val = sp.simplify(fac.subs(subs_dict))
                        if fac_val == sp.S.Zero:
                            if verbose:
                                print(f"    czynnik log {fac} → 0 → BIEGUN LOG (Landau)")
                            return "logarithmic_pole"
                        # Przypadek B: czynnik log dywerguje → brak rzędu alg.
                        # Test: lim (var-point)^1 * expr = 0 (nie skończone niezerowe)
                        if len(subs_dict) == 1:
                            v, pt = list(subs_dict.items())[0]
                            try:
                                test_lim = sp.limit((v - pt) * expr, v, pt)
                                if test_lim == sp.S.Zero:
                                    if verbose:
                                        print(f"    lim (x-x0)·f=0 przy log w denom → BIEGUN LOG (IR)")
                                    return "logarithmic_pole"
                            except Exception:
                                pass
            except Exception:
                pass
            return "pole"

        # Sprawdź czy to osobliwość istotna (exp(1/x) itp.)
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
    Oblicza rząd bieguna i residuum dla niesuwalnej osobliwości.

    Algorytm:
      Dla n = 1, 2, ..., max_order:
        candidate = lim_{x→x0} (x - x0)^n · f(x)
        Jeśli candidate skończony i niezerowy → rząd = n
        Jeśli candidate = 0 → za niski rząd, próbuj n+1
        Jeśli candidate = ∞ → błąd obliczeń, próbuj n+1

    Residuum:
      Dla rzędu 1: res = candidate  (bo res = lim (x-x0)¹ · f(x))
      Dla rzędu n: res = candidate / (n-1)!  — uogólniony wzór
      Ale fizycznie sensowne residuum (w sensie Cauchy'ego) tylko dla n=1.

    Obsługa wielozmiennowa:
      Dla wielu zmiennych analiza po pierwszej zmiennej (główna osobliwość).
      Pozostałe zmienne traktowane jako parametry.

    Returns:
        PoleSingularity — jeśli udało się obliczyć rząd i residuum
        WheelNumber(⊥) — fallback gdy obliczenie niemożliwe
    """
    # Dla wielu zmiennych: analiza po pierwszej
    if len(variables) == 1:
        var, point = variables[0]
    else:
        # Znajdź zmienną względem której mianownik zeruje się
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
                continue  # za niski rząd

            if candidate == sp.S.Zero:
                continue  # też za niski — wyrażenie zanika szybciej

            # Mamy skończony, niezerowy wynik — to jest rząd bieguna
            candidate = sp.simplify(candidate)

            # Residuum: tylko dla prostego bieguna (n=1)
            if n == 1:
                residue = candidate
            else:
                # Uogólniony współczynnik Laurenta: a_{-n} = candidate/(n-1)!
                residue = None   # Cauchy residuum zdefiniowane tylko dla n=1

            # Zbuduj hint rozwinięcia Laurenta
            if n == 1:
                hint = f"({candidate}) · 1/({var}-{point}) + O(1)"
            else:
                hint = f"({candidate}) · 1/({var}-{point})^{n} + O(1/({var}-{point})^{n-1})"

            if verbose:
                print(f"  Rząd bieguna : {n}")
                print(f"  Wsp. główny  : {candidate}")
                if n == 1:
                    print(f"  Residuum     : {residue}")
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
                print(f"  Rząd {n}: błąd ({e}), próbuję wyższy")
            continue

    # Fallback — nie udało się obliczyć
    if verbose:
        print(f"  Residue analysis nie powiodła się → ⊥ (fallback)")
    return W(BOTTOM)


def _compute_logarithmic_pole(
    expr:      sp.Basic,
    variables: list[tuple[sp.Symbol, sp.Basic]],
    verbose:   bool = False,
) -> WheelCalcResult:
    """
    Analiza bieguna logarytmicznego — czynnik log(var/μ²) zeruje się w mianowniku.

    Strategia:
      1. Zidentyfikuj zmienną i punkt osobliwy (mianownik → 0).
      2. Wyodrębnij czynnik logarytmiczny z mianownika.
      3. Oblicz residuum przez sp.residue (działa dla biegunów prostych log).
      4. Sprawdź rząd przez lim (x-x0)^n · f(x).
      5. Zbuduj hint rozwinięcia Laurenta.

    Dlaczego sp.residue zamiast sp.limit * (x-x0)?
      sp.residue używa wewnętrznego rozwinięcia Laurenta SymPy,
      które poprawnie obsługuje czynniki log w mianowniku.

    Returns:
        LogarithmicSingularity  — gdy udało się obliczyć rząd i residuum
        PoleSingularity         — fallback gdy czynnik log nie wpływa na rząd
        WheelNumber(⊥)          — fallback gdy obliczenie niemożliwe
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
        # Wyodrębnij czynnik logarytmiczny i ustal typ bieguna.
        raw_denom = sp.denom(expr)
        denom_factors = sp.Mul.make_args(raw_denom)
        log_factor = None
        log_type = "landau"   # "landau" = log zeruje się | "ir" = log dywerguje

        for fac in denom_factors:
            if fac.has(sp.log):
                fac_val = sp.simplify(fac.subs({var: point}))
                if fac_val == sp.S.Zero:
                    log_factor = fac
                    log_type = "landau"
                    break
                else:
                    # Przypadek IR: log dywerguje, ale czynnik algebraiczny też → 0
                    log_factor = fac
                    log_type = "ir"

        # Fallback dla log_factor
        if log_factor is None:
            _, denom_fb = sp.fraction(sp.cancel(expr))
            log_atoms = [a for a in denom_fb.atoms(sp.log) if a.has(var)]
            log_factor = log_atoms[0] if log_atoms else sp.log(var)
            log_type = "ir"

        if verbose:
            print(f"  Czynnik log  : {log_factor}  [{log_type}]")

        # Przypadek IR: lim (x-x0)^n * f = 0 dla każdego n
        # Brak rzędu algebraicznego — biegun logarytmicznie wzmocniony
        if log_type == "ir":
            try:
                # Oblicz residuum przez sp.residue (może zadziałać)
                try:
                    residue = sp.residue(expr, var, point)
                    residue = sp.simplify(residue)
                except Exception:
                    residue = None

                hint = f"biegun IR log: 1/({var}·log({var}/μ²)) — brak rzędu alg."
                if verbose:
                    print(f"  Typ IR       : logarytmicznie wzmocniony (lim x^n·f=0 ∀n)")
                    print(f"  Residuum     : {residue}")
                    print(f"  Laurent      : {hint}")

                return LogarithmicSingularity(
                    expression=expr,
                    variables=variables,
                    singular_point=point,
                    pole_order=1,    # konwencja: raportujemy rząd "efektywny"
                    residue=residue,
                    log_factor=log_factor,
                    laurent_hint=hint,
                )
            except Exception as e:
                if verbose:
                    print(f"  IR fallback błąd: {e}")
                return W(BOTTOM)

        # Oblicz residuum przez SymPy (obsługuje log w mianowniku)
        try:
            residue = sp.residue(expr, var, point)
            residue = sp.simplify(residue)
            if verbose:
                print(f"  sp.residue   : {residue}")
        except Exception as e:
            if verbose:
                print(f"  sp.residue błąd: {e}")
            residue = None

        # Sprawdź rząd bieguna przez lim (x-x0)^n · f(x)
        pole_order = 1  # domyślnie — bieguny log są zazwyczaj rzędu 1
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
                    print(f"  Rząd bieguna : {n} (lim (x-x0)^{n}·f = {sp.simplify(cand)})")
                break
            except Exception:
                continue

        # Zbuduj hint Laurenta
        if residue is not None and pole_order == 1:
            hint = f"({residue}) · 1/({var}-{point}) + O(1)"
        elif residue is not None:
            hint = f"({residue}) · 1/({var}-{point})^{pole_order} + O(1/({var}-{point})^{pole_order-1})"
        else:
            hint = f"biegun log rzędu {pole_order} przy {var}={point}"

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
            print(f"  _compute_logarithmic_pole błąd: {e} → fallback ⊥")
        return W(BOTTOM)


def _has_essential_singularity(expr: sp.Basic, subs_dict: dict) -> bool:
    """Heurystyczna detekcja osobliwości istotnych (exp(1/x), sin(1/x))."""
    try:
        expr_str = str(expr)
        # Proste heurystyki — exp(1/x) przy x=0
        for var, point in subs_dict.items():
            if point == sp.S.Zero:
                # Szukamy wzorców 1/var wewnątrz funkcji transcendentnych
                if expr.has(sp.exp) or expr.has(sp.sin) or expr.has(sp.cos):
                    inner_check = expr.subs(var, sp.Symbol('_test_eps'))
                    if f"1/_test_eps" in str(inner_check) or f"/_test_eps" in str(inner_check):
                        return True
        return False
    except Exception:
        return False


# ─── Rozwinięcie Taylora ───────────────────────────────────────────────────────

def _try_taylor(
    expr:      sp.Basic,
    variables: list[tuple[sp.Symbol, sp.Basic]],
    max_order: int,
    verbose:   bool,
) -> Optional[tuple[sp.Basic, int, str]]:
    """
    Próbuje obliczyć wartość graniczną przez rozwinięcie Taylora.

    Strategia dla jednej zmiennej:
      Rozwiń expr w szereg wokół punktu:
        expr = a_n*(x-x0)^n + a_{n+1}*(x-x0)^{n+1} + ...
      Jeśli najniższy rząd n = 0 → granica = a_0 (skończona).
      Jeśli n > 0 → granica = 0 (wyrażenie → 0).
      Jeśli n < 0 → prawdziwy biegun (potwierdzenie ⊥).

    Strategia dla wielu zmiennych:
      Iteracyjne podstawianie: najpierw Taylor po x1, potem po x2...
      Jeśli którakolwiek daje biegun → ⊥.

    Returns:
        (limit_value, order, series_hint) jeśli znaleziono granicę
        None jeśli biegun lub nie można obliczyć
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
    """Taylor dla jednej zmiennej."""
    for order in range(1, max_order + 1):
        try:
            # Rozwinięcie Laurenta/Taylora do rzędu `order`
            series = sp.series(expr, var, point, n=order + 2)

            if verbose and order == 1:
                print(f"    Szereg Laurenta: {series}")

            # Usuń człon O(...)
            series_no_O = series.removeO()

            # Oblicz granicę — podstaw var=point do rozwinięcia
            limit_candidate = sp.simplify(series_no_O.subs(var, point))

            # Sprawdź czy wynik jest skończony
            if _is_finite_value(limit_candidate):
                # Zbuduj czytelny hint
                series_str = str(series).replace("O(", "O(").replace("\n", "")
                if len(series_str) > 80:
                    series_str = series_str[:77] + "..."
                return limit_candidate, order, series_str

            # Jeśli limit_candidate zawiera nieskończoność → biegun
            if limit_candidate in (sp.oo, sp.zoo, sp.nan, -sp.oo):
                if verbose:
                    print(f"    Rząd {order}: granica → {limit_candidate} (biegun)")
                return None

        except (sp.core.power.PoleError, ZeroDivisionError):
            if verbose:
                print(f"    Rząd {order}: PoleError — prawdziwy biegun")
            return None
        except Exception as e:
            if verbose:
                print(f"    Rząd {order}: błąd ({e}), próbuję wyższy rząd")
            continue

    return None


def _taylor_multivar(
    expr:      sp.Basic,
    variables: list[tuple[sp.Symbol, sp.Basic]],
    max_order: int,
    verbose:   bool,
) -> Optional[tuple[sp.Basic, int, str]]:
    """
    Taylor dla wielu zmiennych — iteracyjny.

    Strategia: podstawiaj zmienne jedna po drugiej, każdorazowo
    obliczając szereg Taylora. Jeśli któraś daje biegun → None.

    Ograniczenie: kolejność podstawień może mieć znaczenie.
    Próbujemy obie kolejności i bierzemy wynik niesprzeczny.
    """
    from itertools import permutations

    best_result = None

    for perm in permutations(variables):
        result = _taylor_sequential(expr, list(perm), max_order, verbose)
        if result is not None:
            val, order, hint = result
            if _is_finite_value(val):
                # Sprawdź czy inne kolejności dają ten sam wynik
                if best_result is None:
                    best_result = result
                else:
                    prev_val = best_result[0]
                    try:
                        if sp.simplify(val - prev_val) != sp.S.Zero:
                            if verbose:
                                print(f"    ⚠ Różne kolejności dają różne granice!")
                                print(f"      {perm} → {val}")
                                print(f"      poprzednia → {prev_val}")
                            # Wybierz prostszą wartość
                            best_result = result if sp.count_ops(val) < sp.count_ops(prev_val) else best_result
                    except Exception:
                        pass

    if best_result is not None:
        val, order, hint = best_result
        max_ord = max(order, len(variables))
        return val, max_ord, hint

    # Fallback: użyj sp.limit bezpośrednio (dla prostych przypadków)
    return _sympy_limit_fallback(expr, variables, verbose)


def _taylor_sequential(
    expr:      sp.Basic,
    variables: list[tuple[sp.Symbol, sp.Basic]],
    max_order: int,
    verbose:   bool,
) -> Optional[tuple[sp.Basic, int, str]]:
    """Sekwencyjne podstawianie Taylor dla listy zmiennych."""
    current_expr = expr
    max_used_order = 1

    for var, point in variables:
        result = _taylor_single(current_expr, var, point, max_order, verbose=False)
        if result is None:
            return None
        limit_val, order, hint = result
        max_used_order = max(max_used_order, order)
        # Podstaw wartość graniczną do kolejnego kroku
        current_expr = sp.sympify(limit_val)

    if _is_finite_value(current_expr):
        return current_expr, max_used_order, f"wielozmiennowe ({len(variables)} zmiennych)"
    return None


def _sympy_limit_fallback(
    expr:      sp.Basic,
    variables: list[tuple[sp.Symbol, sp.Basic]],
    verbose:   bool,
) -> Optional[tuple[sp.Basic, int, str]]:
    """
    Fallback: użyj sp.limit bezpośrednio gdy Taylor zawodzi.
    Działa dla prostszych wielozmiennowych przypadków.
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
            print(f"    Fallback sp.limit błąd: {e}")
    return None


# ─── Pomocnicze ────────────────────────────────────────────────────────────────

def _is_finite_value(val) -> bool:
    """Czy wartość jest skończona i dobrze określona."""
    if val is None:
        return False
    try:
        if val in (sp.oo, sp.zoo, sp.nan, -sp.oo):
            return False
        if hasattr(val, 'has'):
            if val.has(sp.oo) or val.has(sp.zoo) or val.has(sp.nan):
                return False
            # Sprawdź czy nie zawiera O(...)
            if val.has(sp.Order):
                return False
        return True
    except Exception:
        return False


# ─── Klasyfikacja wyrażenia — pełna analiza ────────────────────────────────────

def analyse_singularity(
    expr:      sp.Basic,
    variables: list[tuple[sp.Symbol, sp.Basic]],
    name:      str = "",
    max_order: int = 8,
    verbose:   bool = True,
) -> WheelCalcResult:
    """
    Pełna analiza osobliwości z raportem.
    Wrapper wokół wheel_limit z bardziej szczegółowym wyjściem.
    """
    if verbose:
        print(f"\n{'═'*62}")
        print(f"  ANALIZA: {name or expr}")
        vars_str = ", ".join(f"{v}→{p}" for v,p in variables)
        print(f"  Punkt  : {vars_str}")
        print(f"{'─'*62}")

    result = wheel_limit(expr, variables, max_order=max_order, verbose=verbose)

    if verbose:
        print()
        if isinstance(result, RemovableSingularity):
            print(result.report())
            print(f"\n  ✓ WYNIK: osobliwość USUWALNA → lim = {result.limit_value}")
        elif isinstance(result, LogarithmicSingularity):
            print(result.report())
            res_str = f", res={result.residue}" if result.residue is not None else ""
            print(f"\n  ✗ WYNIK: BIEGUN LOGARYTMICZNY rzędu {result.pole_order}{res_str} → ⊥")
        elif isinstance(result, PoleSingularity):
            print(result.report())
            res_str = f", res={result.residue}" if result.residue is not None else ""
            print(f"\n  ✗ WYNIK: BIEGUN rzędu {result.pole_order}{res_str} → ⊥")
        elif result.is_bottom:
            print(f"  ✗ WYNIK: osobliwość NIEUSUWALNA → ⊥")
        else:
            print(f"  ✓ WYNIK: punkt regularny → {result}")
        print(f"{'═'*62}")

    return result


def classify_batch(
    cases: list[dict],
    max_order: int = 8,
    verbose: bool = True,
) -> list[dict]:
    """
    Analizuje batch wyrażeń i zwraca tabelę wyników.

    cases: lista słowników z kluczami:
        name: str
        expr: sp.Basic
        variables: [(var, point), ...]
        expected_limit: sp.Basic lub None (opcjonalnie do weryfikacji)

    Returns:
        lista wyników z polami:
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
                type_str = f"{'USUWALNA':12}"
            elif r_type == "LOG_POLE":
                res_str = f" res={result.residue}" if result.residue is not None else ""
                type_str = f"{'BIEGUN_LOG['+str(order)+']':14}"
            elif r_type == "POLE":
                res_str = f" res={result.residue}" if result.residue is not None else ""
                type_str = f"{'BIEGUN['+str(order)+']':12}"
            elif r_type == "BOTTOM":
                type_str = f"{'BOTTOM':12}"
            else:
                type_str = f"{'REGULARNY':12}"
            lim_str = str(lim)[:20]
            exp_str = f" (oczekiwano: {expected})" if expected is not None and not correct else ""
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


# ─── Testy i demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 62)
    print("  wheel_calculus.py — Wheel + rozszerzenie analityczne")
    print("  Trójpodział: skończone | ⊥ nieusuwalna | ⊥→val usuwalna")
    print("═" * 62)

    x, m, p, r, r_s, omega, omega0 = sp.symbols(
        "x m p r r_s omega omega0", real=True
    )

    # ════════════════════════════════════════════════════════════
    # SEKCJA 1: Znane kontrprzykłady z bazy (powinny → USUWALNE)
    # ════════════════════════════════════════════════════════════
    print("\n▶  KONTRPRZYKŁADY — osobliwości usuwalne")
    print("   (Wheel daje ⊥, ale granica istnieje)\n")

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
            "name": "(x² - 1)/(x - 1)  przy x=1",
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
            "name": "(sin(3x))/(sin(5x))  przy x=0",
            "expr": sp.sin(3*x) / sp.sin(5*x),
            "variables": [(x, sp.S.Zero)],
            "expected_limit": sp.Rational(3, 5),
        },
    ]

    print(f"  {'Równanie':<42} {'Typ':<14} {'lim':<10} {'OK?'}")
    print(f"  {'─'*42} {'─'*14} {'─'*10} {'─'*4}")
    classify_batch(cases_removable, max_order=8, verbose=True)

    # ════════════════════════════════════════════════════════════
    # SEKCJA 2: Prawdziwe bieguny z residue analysis
    # ════════════════════════════════════════════════════════════
    print("\n▶  BIEGUNY ALGEBRAICZNE — rząd + residuum (Cauchy)\n")

    cases_poles = [
        {
            "name": "1/x przy x=0",
            "expr": 1/x,
            "variables": [(x, sp.S.Zero)],
            "expected_limit": sp.zoo,
        },
        {
            "name": "Propagator skalarny 1/(p²-m²) przy p=m",
            "expr": 1/(p**2 - m**2),
            "variables": [(p, m)],
            "expected_limit": sp.zoo,
        },
        {
            "name": "g_rr Schwarzschilda przy r=r_s",
            "expr": 1/(1 - r_s/r),
            "variables": [(r, r_s)],
            "expected_limit": sp.zoo,
        },
        {
            "name": "Rezonans 1/(ω²-ω₀²) przy ω=ω₀",
            "expr": 1/(omega**2 - omega0**2),
            "variables": [(omega, omega0)],
            "expected_limit": sp.zoo,
        },
        {
            "name": "Biegun podwójny 1/(p-m)²",
            "expr": 1/(p - m)**2,
            "variables": [(p, m)],
            "expected_limit": sp.zoo,
        },
        {
            "name": "Biegun podwójny 1/x²",
            "expr": 1/x**2,
            "variables": [(x, sp.S.Zero)],
            "expected_limit": sp.zoo,
        },
    ]

    print(f"  {'Równanie':<42} {'Typ':<14} {'res':<20} {'OK?'}")
    print(f"  {'─'*42} {'─'*14} {'─'*20} {'─'*4}")
    classify_batch(cases_poles, max_order=8, verbose=True)

    # Szczegółowa analiza propagatora
    print()
    analyse_singularity(
        1/(p**2 - m**2),
        [(p, m)],
        name="Propagator skalarny 1/(p²-m²) — pełna analiza",
        max_order=6,
        verbose=True,
    )

    # ════════════════════════════════════════════════════════════
    # SEKCJA 3: Punkty regularne (Wheel daje skończoną wartość)
    # ════════════════════════════════════════════════════════════
    print("\n▶  PUNKTY REGULARNE — Wheel OK, calculus nie ingeruje\n")

    x_sym = sp.Symbol("x", positive=True)
    cases_regular = [
        {
            "name": "1/(p²+m²) przy p=0, m=1",
            "expr": 1/(p**2 + m**2),
            "variables": [(p, sp.S.Zero), (m, sp.S.One)],
            "expected_limit": sp.S.One,
        },
        {
            "name": "sin(x)/x² przy x=1",
            "expr": sp.sin(x)/x**2,
            "variables": [(x, sp.S.One)],
            "expected_limit": sp.sin(sp.S.One),
        },
    ]

    print(f"  {'Równanie':<42} {'Typ':<14} {'lim':<10} {'OK?'}")
    print(f"  {'─'*42} {'─'*14} {'─'*10} {'─'*4}")
    classify_batch(cases_regular, max_order=8, verbose=True)

    # ════════════════════════════════════════════════════════════
    # SEKCJA 4: Wielozmiennowe (m=0 i p=0 jednocześnie)
    # ════════════════════════════════════════════════════════════
    print("\n▶  WIELOZMIENNOWE — dwie zmienne jednocześnie\n")

    cases_multi = [
        {
            "name": "m·p/(m²+p²) przy m=0, p=0",
            "expr": m*p / (m**2 + p**2),
            "variables": [(m, sp.S.Zero), (p, sp.S.Zero)],
            "expected_limit": None,  # granica zależy od kierunku!
        },
        {
            "name": "(sin(m)+sin(p))/(m+p) przy m=0, p=0",
            "expr": (sp.sin(m) + sp.sin(p)) / (m + p),
            "variables": [(m, sp.S.Zero), (p, sp.S.Zero)],
            "expected_limit": sp.S.One,
        },
    ]

    print(f"  {'Równanie':<42} {'Typ':<14} {'lim':<10} {'OK?'}")
    print(f"  {'─'*42} {'─'*14} {'─'*10} {'─'*4}")
    classify_batch(cases_multi, max_order=8, verbose=True)

    # ════════════════════════════════════════════════════════════
    # SEKCJA 5: Szczegółowy raport dla sinc (demo verbose)
    # ════════════════════════════════════════════════════════════
    print("\n▶  Szczegółowa analiza sinc(x) przy x=0 (verbose)\n")
    analyse_singularity(
        sp.sin(x)/x,
        [(x, sp.S.Zero)],
        name="sinc(x) = sin(x)/x",
        max_order=6,
        verbose=True,
    )

    # ════════════════════════════════════════════════════════════
    # SEKCJA 6: Bieguny logarytmiczne QCD — Kierunek 4
    # ════════════════════════════════════════════════════════════
    print("\n▶  BIEGUNY LOGARYTMICZNE QCD — propagator gluonu\n")
    print("   (czynnik log zeruje mianownik — nowy typ w wheel_calculus)\n")

    k2      = sp.Symbol("k2",      positive=True)
    alpha_s = sp.Symbol("alpha_s", positive=True)
    mu_r    = sp.Symbol("mu_r",    positive=True)

    gluon_prop = sp.Integer(1) / (k2 * (1 + alpha_s * sp.log(k2 / mu_r**2)))
    k2_ir      = sp.S.Zero
    k2_landau  = mu_r**2 * sp.exp(-sp.Integer(1) / alpha_s)

    cases_qcd = [
        {
            "name": "Gluon prop. — biegun IR (k²=0)",
            "expr": gluon_prop,
            "variables": [(k2, k2_ir)],
            "expected_limit": sp.zoo,
        },
        {
            "name": "Gluon prop. — biegun Landaua",
            "expr": gluon_prop,
            "variables": [(k2, k2_landau)],
            "expected_limit": sp.zoo,
        },
    ]

    print(f"  {'Równanie':<42} {'Typ':<16} {'res':<20} {'OK?'}")
    print(f"  {'─'*42} {'─'*16} {'─'*20} {'─'*4}")
    classify_batch(cases_qcd, max_order=8, verbose=True)

    print()
    analyse_singularity(
        gluon_prop,
        [(k2, k2_landau)],
        name="Propagator gluonu QCD — biegun Landaua (pełna analiza)",
        max_order=6,
        verbose=True,
    )

    # ════════════════════════════════════════════════════════════
    # PODSUMOWANIE
    # ════════════════════════════════════════════════════════════
    print("\n" + "═"*62)
    print("  PODSUMOWANIE — wheel_calculus.py")
    print("═"*62)
    print("""
  Pięciopodział:
    WheelFinite(v)            → regularny punkt (Wheel OK)
    WheelNumber(⊥)            → ⊥ bez struktury (fallback)
    RemovableSingularity      → Wheel=⊥ ale lim=v (osobliwość usuwalna)
    PoleSingularity           → biegun algebraiczny: rząd + residuum + Laurent
    LogarithmicSingularity    → biegun log: czynnik log zeruje mianownik

  Residue analysis:
    1/(p²-m²) przy p=m  → POLE[1], res=1/(2m)
    1/x² przy x=0       → POLE[2], res=N/A (Cauchy tylko dla rzędu 1)
    g_rr przy r=r_s     → POLE[1], res=r_s
    Gluon @ Landau      → LOG_POLE[1], res=1/αs  ← NOWE

  Związek z QFT:
    residuum propagatora = amplituda przejścia on-shell
    residuum bieguna Landaua = 1/αs (siła sprzężenia QCD)
    twierdzenie Cauchy'ego: ∮ f(z) dz = 2πi · Σ res(f, zₖ)

  Architektura:
    wheel_algebra.py  — aksjomatyczna Wheel Algebra, bez zmian
    wheel_calculus.py — rozszerzenie analityczne (ten moduł)

  Kluczowa różnica (ważna dla preprintu!):
    Wheel Algebra:   sin(0)/0 = 0/0 = ⊥   (algebra punktowa)
    wheel_calculus:  sin(x)/x przy x→0  = 1  (rozwinięcie Taylora)
    Oba są POPRAWNE — opisują różne pytania:
      ⊥ → "co się dzieje w tym punkcie algebraicznie?"
      1 → "co jest wartością graniczną analizy matematycznej?"
""")