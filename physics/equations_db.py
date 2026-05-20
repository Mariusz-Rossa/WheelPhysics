# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
equations_db.py — baza danych równań fizycznych

Katalog znanych równań z metadanymi:
  - priorytet (1=dzielenie, 2=osobliwość, 3=granica)
  - zmienne i wartości krytyczne
  - znane osobliwości i ich fizyczny sens
  - wynik analizy Wheel (uzupełniany przez system)

Strategia zgodna z instrukcją projektu:
  Prio 1: równania z dzieleniem
  Prio 2: osobliwości (wynik → ∞)
  Prio 3: granice i pochodne (0/0)
  Pomijamy: bez dzielenia (identyczne w Wheel)
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import sympy as sp


class Priority(IntEnum):
    DIVISION    = 1   # zawiera dzielenie
    SINGULARITY = 2   # wynik → ∞ w pewnych punktach
    LIMIT       = 3   # forma 0/0 lub granica


@dataclass
class PhysicsEquation:
    """Opis pojedynczego równania w bazie."""
    name:             str
    domain:           str                    # "GR", "QFT", "thermo", ...
    expression:       sp.Basic
    variables:        list[sp.Symbol]
    parameters:       list[sp.Symbol]
    priority:         Priority
    known_singular:   list[dict]             # [{var, value, description}]
    physical_meaning: str
    wheel_result:     Optional[str] = None   # wypełniane po analizie
    notes:            str = ""

    def one_liner(self) -> str:
        prio_str = f"P{int(self.priority)}"
        sing_count = len(self.known_singular)
        return (
            f"[{prio_str}] [{self.domain:<5}] {self.name:<40} "
            f"| osobliwości: {sing_count}"
        )


# ─── Baza równań ──────────────────────────────────────────────────────────────

def build_database() -> list[PhysicsEquation]:
    """Buduje i zwraca pełną bazę równań."""

    # Symbole
    r, r_s         = sp.symbols("r r_s",             positive=True)
    theta          = sp.Symbol("theta",               positive=True)
    t_s            = sp.Symbol("t",                   positive=True)
    a              = sp.Symbol("a",                   positive=True)
    p, m, k, q     = sp.symbols("p m k q",            real=True)
    m_e            = sp.Symbol("m_e",                 positive=True)
    G, M, c        = sp.symbols("G M c",              positive=True)
    H              = sp.Symbol("H",                   real=True)
    Lambda         = sp.Symbol("Lambda",              positive=True)
    rho            = sp.Symbol("rho",                 positive=True)
    k_curv         = sp.Symbol("k_curv",              real=True)
    T              = sp.Symbol("T",                   positive=True)
    kB             = sp.Symbol("k_B",                 positive=True)
    omega, omega0  = sp.symbols("omega omega0",       positive=True)
    hbar           = sp.Symbol("hbar",                positive=True)
    epsilon        = sp.Symbol("epsilon",             positive=True)
    x              = sp.Symbol("x",                   real=True)
    n              = sp.Symbol("n",                   positive=True, integer=True)
    k_e            = sp.Symbol("k_e",                 positive=True)
    e_charge       = sp.Symbol("e_charge",            positive=True)
    q_mom          = sp.Symbol("q_mom",               real=True)
    V              = sp.Symbol("V",                   positive=True)
    s              = sp.Symbol("s",                   positive=True)

    db = []

    # ── PRIORYTET 1: Ogólna teoria względności ────────────────────────────────

    db.append(PhysicsEquation(
        name="g_rr Schwarzschilda",
        domain="GR",
        expression=1 / (1 - r_s / r),
        variables=[r],
        parameters=[r_s],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": r, "value": r_s,        "description": "Horyzont zdarzeń (artefakt układu)"},
            {"var": r, "value": sp.S.Zero,  "description": "Osobliwość fizyczna"},
        ],
        physical_meaning="Składowa radialna tensora metrycznego Schwarzschilda",
        notes="K(r_s) skończone — horyzont NIE jest osobliwością fizyczną",
    ))

    db.append(PhysicsEquation(
        name="g_tt Schwarzschilda",
        domain="GR",
        expression=-(1 - r_s / r) * c**2,
        variables=[r],
        parameters=[r_s, c],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": r, "value": sp.S.Zero, "description": "Osobliwość fizyczna r=0"},
        ],
        physical_meaning="Składowa czasowa tensora metrycznego Schwarzschilda",
    ))

    db.append(PhysicsEquation(
        name="Niezmiennik Kretschmanna",
        domain="GR",
        expression=12 * r_s**2 / r**6,
        variables=[r],
        parameters=[r_s],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": sp.S.Zero, "description": "Prawdziwa osobliwość — K→∞"},
        ],
        physical_meaning="K = R_abcd R^abcd — niezmiennik skalarny krzywizny. K(r_s) skończony.",
        notes="Kluczowy test: odróżnia artefakt układu od osobliwości fizycznej",
    ))

    db.append(PhysicsEquation(
        name="Symbol Christoffela Γ^t_tr",
        domain="GR",
        expression=r_s / (2 * r * (r - r_s)),
        variables=[r],
        parameters=[r_s],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": r, "value": r_s,       "description": "Horyzont — osobliwość układu"},
            {"var": r, "value": sp.S.Zero, "description": "Osobliwość fizyczna"},
        ],
        physical_meaning="Symbol Christoffela — związek z przyspieszeniem geodezyjnym",
    ))

    # ── PRIORYTET 1: Kosmologia ───────────────────────────────────────────────

    db.append(PhysicsEquation(
        name="Friedmann — człon krzywiznowy",
        domain="COSMO",
        expression=k_curv * c**2 / a**2,
        variables=[a],
        parameters=[k_curv, c],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": a, "value": sp.S.Zero, "description": "Wielki Wybuch / Wielki Ścisk"},
        ],
        physical_meaning="Człon krzywiznowy w równaniu Friedmanna H² = 8πGρ/3 - kc²/a² + Λc²/3",
        notes="Symetria a→-a sugeruje istnienie 'przed-wszechświata' w Wheel",
    ))

    db.append(PhysicsEquation(
        name="Gęstość materii ρ~1/a³",
        domain="COSMO",
        expression=rho / a**3,
        variables=[a],
        parameters=[rho],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": a, "value": sp.S.Zero, "description": "Wielki Wybuch — gęstość → ∞"},
        ],
        physical_meaning="Ewolucja gęstości materii z czynnikiem skali",
    ))

    db.append(PhysicsEquation(
        name="Gęstość promieniowania ρ~1/a⁴",
        domain="COSMO",
        expression=rho / a**4,
        variables=[a],
        parameters=[rho],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": a, "value": sp.S.Zero, "description": "Wielki Wybuch — promieniowanie → ∞"},
        ],
        physical_meaning="Ewolucja gęstości promieniowania (+ człon ciśnienia relatywistycznego)",
    ))

    # ── PRIORYTET 1: QFT ──────────────────────────────────────────────────────

    db.append(PhysicsEquation(
        name="Propagator skalarny Feynmana",
        domain="QFT",
        expression=sp.Integer(1) / (p**2 - m**2),
        variables=[p],
        parameters=[m],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": p, "value":  m, "description": "On-shell (p=+m) — cząstka rzeczywista"},
            {"var": p, "value": -m, "description": "On-shell (p=-m) — antycząstka"},
        ],
        physical_meaning="Propagator Kleina-Gordona. Biegun = stan asymptotyczny (obserwowalny).",
        notes="Hipoteza: ⊥ on-shell = algebraiczna definicja obserwowalności",
    ))

    db.append(PhysicsEquation(
        name="Propagator fotonowy",
        domain="QFT",
        expression=sp.Integer(1) / k**2,
        variables=[k],
        parameters=[],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": k, "value": sp.S.Zero, "description": "Foton bezmasowy on-shell (k=0)"},
        ],
        physical_meaning="Propagator fotonu w gauge Lorenza. IR osobliwość przy k=0.",
    ))

    db.append(PhysicsEquation(
        name="Propagator fermionowy (uproszczony)",
        domain="QFT",
        expression=(p + m_e) / (p**2 - m_e**2),
        variables=[p],
        parameters=[m_e],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": p, "value":  m_e, "description": "On-shell elektron"},
            {"var": p, "value": -m_e, "description": "On-shell pozyton"},
        ],
        physical_meaning="Propagator Diraca (skalaryzowany). Licznik: p̸+m po śladzie.",
    ))

    db.append(PhysicsEquation(
        name="Propagator bezmasowego fermiona",
        domain="QFT",
        expression=sp.Integer(1) / p,
        variables=[p],
        parameters=[],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": p, "value": sp.S.Zero, "description": "IR — bezmasowy fermion nie może mieć p=0"},
        ],
        physical_meaning="Propagator Diraca dla m=0 (neutrina, kwarki chiralne).",
        notes="Wheel algebraicznie zakazuje p=0 dla bezmasowych — fizycznie poprawne",
    ))

    # ── PRIORYTET 2: Termodynamika / statystyczna ─────────────────────────────

    db.append(PhysicsEquation(
        name="Rozkład Boltzmanna 1/T",
        domain="THERMO",
        expression=sp.exp(-epsilon / (kB * T)) / T,
        variables=[T],
        parameters=[epsilon, kB],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": T, "value": sp.S.Zero, "description": "Zero bezwzględne — osobliwość termodynamiczna"},
        ],
        physical_meaning="Prawdopodobieństwo obsadzenia stanu energetycznego ε przy temp. T",
    ))

    db.append(PhysicsEquation(
        name="Rozkład Plancka",
        domain="THERMO",
        expression=hbar * omega / (sp.exp(hbar * omega / (kB * T)) - 1),
        variables=[T],
        parameters=[hbar, omega, kB],
        priority=Priority.LIMIT,
        known_singular=[
            {"var": T, "value": sp.S.Zero, "description": "T=0 — mianownik exp(∞)-1 → ∞"},
        ],
        physical_meaning="Energia fotonów w promieniowaniu ciała doskonale czarnego",
        notes="Forma 0/0 gdy T→∞ (klasyczna granica Rayleigha-Jeansa)",
    ))

    # ── PRIORYTET 3: Granice i pochodne ──────────────────────────────────────

    db.append(PhysicsEquation(
        name="sinc(x) = sin(x)/x",
        domain="MATH",
        expression=sp.sin(x) / x,
        variables=[x],
        parameters=[],
        priority=Priority.LIMIT,
        known_singular=[
            {"var": x, "value": sp.S.Zero, "description": "Forma 0/0 — granica = 1"},
        ],
        physical_meaning="Funkcja sinc — pojawia się w dyfrakcji, transformacie Fouriera",
        notes="Klasycznie: lim(x→0) sin(x)/x = 1. Wheel: ⊥. To istotna różnica!",
    ))

    # ── NOWE: Tensor Riemanna — Schwarzschild ─────────────────────────────────

    db.append(PhysicsEquation(
        name="Tensor Riemanna R^r_trt",
        domain="GR",
        expression=-r_s / r**3,
        variables=[r],
        parameters=[r_s],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "Osobliwość fizyczna — krzywizna → ∞"},
        ],
        physical_meaning=(
            "R^r_trt = -r_s/r³ — składowa tensora Riemanna dla Schwarzschilda. "
            "Przy r=r_s: -1/r_s² (skończony — horyzont regularny). "
            "Przy r=0: ⊥ (osobliwość fizyczna)."
        ),
        notes="R(r_s) skończony — tensor Riemanna potwierdza horyzont jako artefakt",
    ))

    db.append(PhysicsEquation(
        name="Tensor Riemanna R^θ_rθr",
        domain="GR",
        expression=-r_s / (2 * r**3),
        variables=[r],
        parameters=[r_s],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "Osobliwość fizyczna — krzywizna kątowa → ∞"},
        ],
        physical_meaning=(
            "R^θ_rθr = -r_s/(2r³) — składowa kątowo-radialna. "
            "Mierzy krzywiznę w kierunkach kątowych. "
            "Przy r=r_s: -1/(2r_s²) skończona. Przy r=0: ⊥."
        ),
        notes="Ta sama struktura co R^r_trt — osobliwość tylko przy r=0",
    ))

    db.append(PhysicsEquation(
        name="Tensor Riemanna R^φ_tφt",
        domain="GR",
        expression=(r_s / r**3) * (1 - r_s / r),
        variables=[r],
        parameters=[r_s],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "Osobliwość fizyczna"},
            {"var": r, "value": r_s,
             "description": "Horyzont — czynnik f(r)=0 kasuje r_s/r³, wynik 0 nie ⊥"},
        ],
        physical_meaning=(
            "R^φ_tφt = (r_s/r³)·(1-r_s/r). "
            "Wyjątkowy: przy r=r_s mamy 0·∞ — f(r)→0 ale r_s/r³→∞. "
            "Wheel przez rekurencję na /r w f(r) daje ⊥ przy r=0. "
            "Przy r=r_s: (1-r_s/r)→0 kasuje dywergencję, wynik = 0."
        ),
        notes="Przypadek 0·∞ przy r=r_s — fizycznie V_eff=0 na horyzoncie",
    ))

    # ── NOWE: Równanie Diraca ─────────────────────────────────────────────────

    db.append(PhysicsEquation(
        name="Dirac m=0 — propagator Weylla",
        domain="QFT",
        expression=sp.Integer(1) / p,
        variables=[p],
        parameters=[],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": p, "value": sp.S.Zero,
             "description": "Bezmasowy fermion on-shell — p=0 fizycznie nieosiągalne"},
        ],
        physical_meaning=(
            "lim(m→0) S_F(p,m) = 1/p. "
            "Równanie Diraca dla m=0 to dwa niezależne równania Weylla. "
            "Biegun przy p=0 — stan spoczynku bezmasowej cząstki niedostępny (porusza się z c)."
        ),
        notes="Wheel algebraicznie wyprowadza zakaz p=0 dla bezmasowych",
    ))

    db.append(PhysicsEquation(
        name="Dirac — energia relatywistyczna 1/√(p²+m²)",
        domain="QFT",
        expression=sp.Integer(1) / sp.sqrt(p**2 + m**2),
        variables=[p, m],
        parameters=[],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": m, "value": sp.S.Zero,
             "description": "Granica bezmasowa przy p=0 — 1/|p|, osobliwość IR"},
        ],
        physical_meaning=(
            "1/E = 1/√(p²+m²) — normalizacja stanu relativistycznego. "
            "Przy m=0 i p=0: 1/0 → ⊥. "
            "Przy m>0 i p=0: 1/m (skończone — masa regularyzuje IR)."
        ),
        notes="Wheel odróżnia: m>0 brak osobliwości w p=0, m=0 daje ⊥",
    ))

    # ── NOWE: Klein-Gordon w zakrzywionej czasoprzestrzeni ────────────────────

    l_sym = sp.Symbol("l", nonneg=True, integer=True)

    db.append(PhysicsEquation(
        name="Klein-Gordon w Schwarzschildzie V_eff",
        domain="QFT",
        expression=(1 - r_s/r) * (m**2 + l_sym*(l_sym+1)/r**2 + r_s/r**3),
        variables=[r],
        parameters=[r_s, m, l_sym],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "Osobliwość fizyczna — V_eff → ∞"},
            {"var": r, "value": r_s,
             "description": "Horyzont — f(r)=0 zeruje V_eff (bariera znika)"},
        ],
        physical_meaning=(
            "V_eff = f(r)·[m² + l(l+1)/r² + r_s/r³], f=1-r_s/r. "
            "Efektywny potencjał KG w Schwarzschildzie (wsp. żółwia). "
            "Pierwsze połączenie OTW+QFT — obie osobliwości nakładają się przy r=0. "
            "Przy r=r_s: V_eff=0 (horyzont = bariera znika — fizycznie poprawne). "
            "Przy r=0: V_eff=⊥."
        ),
        notes="r=r_s daje V_eff=0, nie ⊥ — horyzont jest tu zanikiem bariery, nie osobliwością",
    ))

    db.append(PhysicsEquation(
        name="Klein-Gordon euklidesowy 1/(p²+m²)",
        domain="QFT",
        expression=sp.Integer(1) / (p**2 + m**2),
        variables=[p],
        parameters=[m],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": p, "value": sp.S.Zero,
             "description": "Przy m=0: IR biegun przy p=0"},
        ],
        physical_meaning=(
            "Euklidesowy propagator KG: 1/(p²+m²). "
            "Dla m>0: brak bieguna na osi rzeczywistej — stąd użyteczność obrotu Wicka. "
            "Dla m=0 przy p=0: ⊥. "
            "Porównanie z Minkowskim 1/(p²-m²): obrót Wicka usuwa bieguny rzeczywiste."
        ),
        notes="m=0,p=0 → ⊥. Dla m>0 i p=0: 1/m² skończone",
    ))


    # ── GR: Metryka Kerr ─────────────────────────────────────────────────────

    a_kerr = sp.Symbol("a_kerr", positive=True)   # moment obrotowy / masę
    Delta  = r**2 - r_s*r + a_kerr**2             # funkcja Kerr
    Sigma  = r**2 + a_kerr**2 * sp.cos(theta)**2  # czynnik kształtu

    db.append(PhysicsEquation(
        name="Kerr g_rr — Δ(r) w mianowniku",
        domain="GR",
        expression=Sigma / Delta,
        variables=[r],
        parameters=[r_s, a_kerr, theta],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": r_s/2 + sp.sqrt(r_s**2/4 - a_kerr**2),
             "description": "Zewnętrzny horyzont r+ — Δ(r+)=0"},
            {"var": r, "value": r_s/2 - sp.sqrt(r_s**2/4 - a_kerr**2),
             "description": "Wewnętrzny horyzont r- — Δ(r-)=0"},
        ],
        physical_meaning=(
            "Składowa g_rr metryki Kerr (obracająca się czarna dziura). "
            "Δ = r²-r_s·r+a² — zeruje się na dwóch horyzontach r±. "
            "Osobliwość pierścieniowa Kerr: r=0, θ=π/2 (Σ→0 i Δ→a²≠0). "
            "Bardziej realistyczny model niż Schwarzschild — każda astrofizyczna BH rotuje."
        ),
        notes="Dwa horyzonty zamiast jednego — bogatsza struktura osobliwości niż Schwarzschild",
    ))

    # ── GR: Metryka Reissnera-Nordströma ─────────────────────────────────────

    r_Q = sp.Symbol("r_Q", positive=True)   # promień ładunku: r_Q²=GQ²/(4πε₀c⁴)
    f_RN = 1 - r_s/r + r_Q**2/r**2

    db.append(PhysicsEquation(
        name="Reissner-Nordström g_rr",
        domain="GR",
        expression=sp.Integer(1) / f_RN,
        variables=[r],
        parameters=[r_s, r_Q],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": r_s/2 + sp.sqrt(r_s**2/4 - r_Q**2),
             "description": "Zewnętrzny horyzont r+ — naładowana BH"},
            {"var": r, "value": r_s/2 - sp.sqrt(r_s**2/4 - r_Q**2),
             "description": "Wewnętrzny horyzont r- (Cauchy'ego)"},
            {"var": r, "value": sp.S.Zero,
             "description": "Osobliwość fizyczna r=0"},
        ],
        physical_meaning=(
            "g_rr metryki Reissnera-Nordströma (czarna dziura z ładunkiem Q). "
            "f_RN = 1 - r_s/r + r_Q²/r². Trzy osobliwości: r+, r-, r=0. "
            "Gdy r_Q = r_s/2: horyzont ekstremalny (r+=r-). "
            "Gdy r_Q > r_s/2: nagie osobliwości (bez horyzontu)."
        ),
        notes="Trzy osobliwości — najbogatsza struktura wśród metryk sferycznych",
    ))

    # ── GR: Promień Hubble'a ──────────────────────────────────────────────────

    db.append(PhysicsEquation(
        name="Promień Hubble'a r_H = c/H",
        domain="COSMO",
        expression=c / H,
        variables=[H],
        parameters=[c],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": H, "value": sp.S.Zero,
             "description": "H=0 — statyczny wszechświat, brak horyzontu kosmologicznego"},
        ],
        physical_meaning=(
            "r_H = c/H — rozmiar horyzontu Hubble'a. "
            "H=0: wszechświat statyczny (model Einsteina), horyzont → ∞. "
            "W Wheel: c/0 = ⊥. "
            "Powiązanie z Friedmannem: H² → 0 gdy a → stałe."
        ),
        notes="Łączy się z równaniami Friedmanna — gdy H²=0, r_H=⊥",
    ))

    # ── Temperatura Hawkinga ──────────────────────────────────────────────────

    db.append(PhysicsEquation(
        name="Temperatura Hawkinga T_H",
        domain="GR",
        expression=hbar * c**3 / (8 * sp.pi * G * M * kB),
        variables=[M],
        parameters=[hbar, c, G, kB],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": M, "value": sp.S.Zero,
             "description": "M=0 — brak czarnej dziury, T_H → ∞"},
        ],
        physical_meaning=(
            "T_H = ħc³/(8πGMk_B) — temperatura promieniowania Hawkinga. "
            "Im mniejsza masa BH, tym wyższa temperatura (paradoks). "
            "M→0: T_H→∞ — końcowe stadium ewaporacji BH. "
            "Wheel: T_H(M=0) = ⊥. Powiązanie z Kretschmannem: K~1/r⁶, M~r_s."
        ),
        notes="Łączy OTW z QFT — promieniowanie termiczne z horyzontu zdarzeń",
    ))

    # ── Mechanika klasyczna ───────────────────────────────────────────────────

    db.append(PhysicsEquation(
        name="Potencjał Coulomba ke²/r",
        domain="CLASS",
        expression=k_e * e_charge**2 / r,
        variables=[r],
        parameters=[k_e, e_charge],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "Punkt ładunku — potencjał → ∞"},
        ],
        physical_meaning=(
            "V_C = ke²/r — potencjał elektrostatyczny punktowego ładunku. "
            "Archetyp wszystkich osobliwości 1/r w fizyce. "
            "W QED zastępowany propagatorem fotonowym 1/q² (już w bazie). "
            "Wheel: V_C(r=0) = ⊥ — ładunek punktowy jest osobliwością."
        ),
        notes="Archetyp osobliwości 1/r — fundament elektrodynamiki klasycznej",
    ))

    db.append(PhysicsEquation(
        name="Potencjał grawitacyjny -GM/r",
        domain="CLASS",
        expression=-G * M / r,
        variables=[r],
        parameters=[G, M],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "Punkt masy — potencjał → -∞"},
        ],
        physical_meaning=(
            "V_g = -GM/r — Newtonowski potencjał grawitacyjny. "
            "Limit nierelatywistyczny metryki Schwarzschilda (g_tt ≈ -1 + r_s/r). "
            "Wheel: V_g(r=0) = ⊥. "
            "Ta sama osobliwość co w OTW — Wheel spójny w obu limitach."
        ),
        notes="Spójność: ta sama ⊥ co w Schwarzschildzie dla r=0",
    ))

    db.append(PhysicsEquation(
        name="Siła Keplera -GM/r²",
        domain="CLASS",
        expression=-G * M / r**2,
        variables=[r],
        parameters=[G, M],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "Kolizja — siła → ∞ (osobliwość kolizyjne)"},
        ],
        physical_meaning=(
            "F = -GM/r² — siła grawitacji Newtona / siła Keplera. "
            "Osobliwość kolizyjne przy r=0 — fundament problemu N-ciał. "
            "Wheel: F(r=0) = ⊥. "
            "Powiązanie z ChaosEngine — w PTC osobliwości kolizyjne to te same ⊥."
        ),
        notes="Osobliwości kolizyjne w PTC — most między WheelPhysics a ChaosEngine",
    ))

    db.append(PhysicsEquation(
        name="Rezonans oscylatora 1/(ω²-ω₀²)",
        domain="CLASS",
        expression=sp.Integer(1) / (omega**2 - omega0**2),
        variables=[omega],
        parameters=[omega0],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": omega, "value": omega0,
             "description": "Rezonans — amplituda → ∞ (bez tłumienia)"},
        ],
        physical_meaning=(
            "Amplituda wymuszonego oscylatora harmonicznego: A ~ 1/(ω²-ω₀²). "
            "Rezonans przy ω=ω₀: A→∞ (bez tłumienia). "
            "Ta sama struktura co propagator: 1/(p²-m²) ↔ 1/(ω²-ω₀²). "
            "Wheel: A(ω=ω₀) = ⊥. Rezonans = stan 'on-shell' oscylatora."
        ),
        notes="Izomorfizm z propagatorem QFT: rezonans klasyczny = on-shell kwantowy",
    ))

    # ── Termodynamika ─────────────────────────────────────────────────────────

    a_vdw, b_vdw, R_gas = sp.symbols("a_vdw b_vdw R_gas", positive=True)

    db.append(PhysicsEquation(
        name="van der Waals — ciśnienie P(V,T)",
        domain="THERMO",
        expression=R_gas * T / (V - b_vdw) - a_vdw / V**2,
        variables=[V],
        parameters=[T, R_gas, a_vdw, b_vdw],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": V, "value": b_vdw,
             "description": "V=b — objętość własna cząsteczek (twarde jądra)"},
            {"var": V, "value": sp.S.Zero,
             "description": "V=0 — osobliwość niefizyczna (gaz nie może się zwinąć)"},
        ],
        physical_meaning=(
            "P = RT/(V-b) - a/V² — równanie stanu van der Waalsa. "
            "V=b: cząsteczki dotykają się, ciśnienie → ∞ (twarde rdzenie). "
            "V=0: czysto matematyczna osobliwość, niefizyczna. "
            "Wheel poprawnie daje ⊥ w obu punktach."
        ),
        notes="V=b to fizyczna granica gazu rzeczywistego, V=0 to artefakt matematyczny",
    ))

    Tc_sym = sp.Symbol("Tc", positive=True)

    db.append(PhysicsEquation(
        name="Ciepło właściwe przy przejściu fazowym ~1/|T-Tc|",
        domain="THERMO",
        expression=sp.Integer(1) / sp.Abs(T - Tc_sym),
        variables=[T],
        parameters=[Tc_sym],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": T, "value": Tc_sym,
             "description": "T=Tc — punkt krytyczny, C → ∞ (wykładnik krytyczny α)"},
        ],
        physical_meaning=(
            "C ~ |T-Tc|^(-α) — dywergencja ciepła właściwego przy przejściu fazowym. "
            "α ≈ 0.11 dla 3D Ising, α=0 (log) dla He-4. "
            "Uproszczenie: α=1 (mean field). "
            "Wheel: C(T=Tc) = ⊥ — punkt krytyczny jest osobliwością termodynamiczną."
        ),
        notes="Wykładniki krytyczne opisują jak szybko zbliżamy się do ⊥",
    ))

    # ── QFT: dodatkowe amplitudy ──────────────────────────────────────────────

    s_var = sp.Symbol("s", positive=True)   # zmienna Mandelstama

    db.append(PhysicsEquation(
        name="Amplituda Comptona 1/(s-m²)",
        domain="QFT",
        expression=sp.Integer(1) / (s_var - m**2),
        variables=[s_var],
        parameters=[m],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": s_var, "value": m**2,
             "description": "s=m² — biegun w kanale s (cząstka pośrednia on-shell)"},
        ],
        physical_meaning=(
            "Amplituda rozpraszania Comptona: A ~ 1/(s-m²). "
            "s = (p+k)² — zmienna Mandelstama. "
            "Biegun przy s=m²: cząstka pośrednia staje się rzeczywista (on-shell). "
            "Identyczna struktura co propagator skalarny — ⊥ = cząstka asymptotyczna."
        ),
        notes="Potwierdza: bieguny w zmiennych Mandelstama to on-shell, czyli ⊥",
    ))

    db.append(PhysicsEquation(
        name="QED wymiana fotonu 1/q²",
        domain="QFT",
        expression=sp.Integer(1) / q_mom**2,
        variables=[q_mom],
        parameters=[],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": q_mom, "value": sp.S.Zero,
             "description": "q=0 — foton o zerowym pędzie (IR, potencjał Coulomba)"},
        ],
        physical_meaning=(
            "Amplituda wymiany fotonu w QED: M ~ e²/q². "
            "Limit q→0: potencjał Coulomba (odzyskujemy fizykę klasyczną). "
            "Wheel: M(q=0) = ⊥. "
            "Połączenie QED z klasyczną elektrodynamiką przez granicę q→0."
        ),
        notes="q→0 to granica klasyczna QED — Wheel daje ⊥ gdzie klasycznie V_C=∞",
    ))

    # ── Matematyka ────────────────────────────────────────────────────────────

    db.append(PhysicsEquation(
        name="Funkcja Gamma Γ(n) ~ 1/n przy n→0",
        domain="MATH",
        expression=sp.Integer(1) / n,
        variables=[n],
        parameters=[],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": n, "value": sp.S.Zero,
             "description": "n=0 — pierwszy biegun Γ(n), residuum = 1"},
        ],
        physical_meaning=(
            "Γ(n) ~ 1/n przy n→0 (i przy n=-1,-2,...). "
            "W regularyzacji wymiarowej QFT: Γ(ε) ~ 1/ε przy ε→0. "
            "To jest źródło dywergencji UV w wymiarowej regularyzacji! "
            "Wheel: Γ_pole(n=0) = ⊥. "
            "Hipoteza: regularyzacja wymiarowa to substytut Wheel dla całek."
        ),
        notes="Most między Wheel a regularyzacją wymiarową — Γ(ε)=⊥ gdy ε→0",
    ))

    db.append(PhysicsEquation(
        name="Funkcja ζ Riemanna — biegun przy s=1",
        domain="MATH",
        expression=sp.Integer(1) / (s - sp.Integer(1)),
        variables=[s],
        parameters=[],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": s, "value": sp.Integer(1),
             "description": "s=1 — biegun ζ(s), residuum = 1"},
        ],
        physical_meaning=(
            "ζ(s) ~ 1/(s-1) przy s→1 — jedyny biegun funkcji Riemanna. "
            "Wheel: ζ_pole(s=1) = ⊥. "
            "W fizyce: ζ-regularyzacja sumy 1+2+3+... = -1/12 omija biegun. "
            "Hipoteza: ζ-regularyzacja to klasyczny substytut Wheel dla szeregów rozbieżnych."
        ),
        notes="ζ-regularyzacja i Wheel — dwa różne sposoby na tę samą osobliwość",
    ))

    db.append(PhysicsEquation(
        name="Transformata Fouriera IR — 1/ω",
        domain="MATH",
        expression=sp.Integer(1) / omega,
        variables=[omega],
        parameters=[],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": omega, "value": sp.S.Zero,
             "description": "ω=0 — IR dywergencja w przestrzeni częstości"},
        ],
        physical_meaning=(
            "F[1/t](ω) ~ 1/ω — transformata Fouriera funkcji 1/t. "
            "IR dywergencja przy ω=0 pojawia się w: "
            "akustyce (mody zerowe), QFT (miękkie fotony/gluony), "
            "turbulencji (widmo Kołmogorowa). "
            "Wheel: (1/ω)(ω=0) = ⊥."
        ),
        notes="IR dywergencje w QFT i Wheel — ω=0 to brak energii, stan nieosiągalny",
    ))


    # ══════════════════════════════════════════════════════════════════════════
    # NOWE RÓWNANIA — v0.9
    # GR/COSMO: Milne/de Sitter, ADM, foton V_eff
    # QFT: propagator gluonu QCD, Green oscylatora z tłumieniem, t-kanał Mandelstama
    # THERMO: entropia Bekenstein-Hawkinga (pochodna), van Hove D(E)
    # MATH: sinc², (1-cos)/x² — kontrprzykłady dla wheel_calculus.py
    # ══════════════════════════════════════════════════════════════════════════

    # ── GR/COSMO: Metryka de Sittera — horyzont kosmologiczny ────────────────

    db.append(PhysicsEquation(
        name="Metryka de Sittera — g_rr = 1/(1 - r²/R_H²)",
        domain="COSMO",
        expression=sp.Integer(1) / (1 - r**2 / r_s**2),   # r_s pełni rolę R_H
        variables=[r],
        parameters=[r_s],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": r, "value": r_s,
             "description": "r = R_H — horyzont kosmologiczny de Sittera (odpowiednik r_s w Schwarzschildzie)"},
        ],
        physical_meaning=(
            "g_rr metryki de Sittera: 1/(1 - r²/R_H²) gdzie R_H = c/H = √(3/Λ). "
            "Horyzont kosmologiczny przy r=R_H — dokładna analogia do horyzontu Schwarzschilda. "
            "Kluczowa różnica: tutaj wnętrze (r<R_H) jest dostępne obserwatorowi, "
            "a zewnętrze (r>R_H) jest za horyzontem (odwrotnie niż BH). "
            "H może zmieniać znak w bardziej ogólnych modelach → dwa horyzonty. "
            "Wheel: g_rr(r=R_H) = ⊥ — ta sama algebra co Schwarzschild."
        ),
        notes="Izomorfizm Schwarzschild ↔ de Sitter: ta sama struktura algebraiczna ⊥, odwrócona fizyka",
    ))

    # ── GR: Energia ADM — osobliwość na granicy (r→∞) ─────────────────────────

    db.append(PhysicsEquation(
        name="Energia ADM — człon 1/r przy r→∞",
        domain="GR",
        expression=sp.Integer(1) / r,
        variables=[r],
        parameters=[],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "r=0 — osobliwość centralna (tu jako osobliwość całkowa przy r→∞ to odwrotny problem)"},
        ],
        physical_meaning=(
            "Energia ADM (Arnowitt-Deser-Misner) — całkowa energia układu w OTW. "
            "E_ADM = -(c²/16πG) ∮ (∂_j h_ii - ∂_i h_ij) dS^j. "
            "Asymptotycznie: h_ij ≈ δ_ij(1 + 2GM/rc²) — człon 1/r dominuje przy r→∞. "
            "ODWRÓCONY PROBLEM: osobliwość nie w centrum (r=0) ale na granicy całkowania. "
            "Wheel operuje punktowo — 1/r przy r=0 daje ⊥, przy r→∞ wyraz 1/r→0 (regularny). "
            "Kontrast z poprzednimi: tu ⊥ w centrum nie jest problemem ADM, "
            "lecz zachowanie przy r→∞ decyduje o energii."
        ),
        notes="Odwrócony problem: fizyka w granicy r→∞, nie r→0. Wheel działa punktowo — inna logika.",
    ))

    # ── GR: Potencjał efektywny dla fotonów (fotosfera Schwarzschilda) ─────────

    db.append(PhysicsEquation(
        name="Potencjał fotonu V_ph = (1-r_s/r)/r²",
        domain="GR",
        expression=(1 - r_s / r) / r**2,
        variables=[r],
        parameters=[r_s],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": r, "value": sp.S.Zero,
             "description": "r=0 — osobliwość fizyczna (V_ph → ∞)"},
            {"var": r, "value": r_s,
             "description": "r=r_s — horyzont, f(r)=0 kasuje 1/r², wynik V_ph=0"},
        ],
        physical_meaning=(
            "V_ph = f(r)/r² = (1-r_s/r)/r² — efektywny potencjał dla fotonów (l=0, m=0) "
            "w Schwarzschildzie (współrzędna żółwia). "
            "Fotosfera (niestabilna orbita kołowa fotonów) przy r_ph = 3r_s/2, "
            "gdzie dV_ph/dr = 0: V_ph(r_ph) = 4/(27r_s²) — skończony. "
            "Uzupełnienie Klein-Gordon V_eff dla bezmasowych (m=0, l=0). "
            "Przy r=r_s: V_ph = 0 (bariera znika na horyzoncie — analogia do KG). "
            "Przy r=0: V_ph = ⊥ (osobliwość fizyczna)."
        ),
        notes="Uzupełnia KG V_eff dla bezmasowych — m=0, l=0. Fotosfera r_ph=3r_s/2 jest regularna.",
    ))

    # ── QFT: Propagator gluonu z samooddziaływaniem QCD ───────────────────────

    alpha_s = sp.Symbol("alpha_s", positive=True)   # stała sprzężenia QCD
    mu_r    = sp.Symbol("mu_r",    positive=True)   # skala renormalizacji
    k2      = sp.Symbol("k2",      positive=True)   # k² (pęd²)

    db.append(PhysicsEquation(
        name="Propagator gluonu QCD z poprawką pętlową",
        domain="QFT",
        expression=sp.Integer(1) / (k2 * (1 + alpha_s * sp.log(k2 / mu_r**2))),
        variables=[k2],
        parameters=[alpha_s, mu_r],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": k2, "value": sp.S.Zero,
             "description": "k²=0 — biegun IR (foton o zerowym pędzie, jak w QED)"},
            {"var": k2, "value": mu_r**2 * sp.exp(-sp.Integer(1) / alpha_s),
             "description": "Biegun Landaua QCD — k²=μ²·exp(-1/αs), nieperturbacyjny"},
        ],
        physical_meaning=(
            "Propagator gluonu z jednostronną poprawką pętlową w QCD: 1/(k²(1+αs·log(k²/μ²))). "
            "DWA typy biegunów — oba nieznane Wheel do tej pory: "
            "(1) k²=0: standardowy biegun IR jak w QED (oczekiwane ⊥). "
            "(2) Biegun Landaua: k²=μ²·exp(-1/αs) — biegun LOGARYTMICZNY, "
            "inny typ niż bieguny algebraiczne 1/xⁿ. "
            "W QED odpowiednik jest niefizyczny (10^280 GeV). "
            "W QCD biegun Landaua jest nieperturbacyjny — pojawia się w skali konfinementu (~ΛQCD). "
            "Pytanie: czy Wheel poprawnie wykrywa bieguny logarytmiczne? Niezbadany teren."
        ),
        notes="NOWY TYP: biegun logarytmiczny. Wheel dotąd testowany tylko na biegunach algebraicznych.",
    ))

    # ── QFT: Funkcja Greena oscylatora z tłumieniem (pomost rezonans↔QM) ──────

    gamma_d = sp.Symbol("gamma_d", positive=True)   # współczynnik tłumienia

    db.append(PhysicsEquation(
        name="Green oscylatora z tłumieniem G(ω) = 1/(ω²-ω₀²+iγω)",
        domain="QFT",
        expression=sp.Integer(1) / (omega**2 - omega0**2 + sp.I * gamma_d * omega),
        variables=[omega],
        parameters=[omega0, gamma_d],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": gamma_d, "value": sp.S.Zero,
             "description": "γ→0 — biegun wraca na oś rzeczywistą: ω=±ω₀ (rezonans klasyczny)"},
            {"var": omega, "value": sp.I * gamma_d / 2 + sp.sqrt(omega0**2 - gamma_d**2 / 4),
             "description": "Biegun w górnej półpłaszczyźnie zespolonej (dla γ>0)"},
        ],
        physical_meaning=(
            "G(ω) = 1/(ω²-ω₀²+iγω) — funkcja Greena oscylatora tłumionego. "
            "Biegun przesuwa się do płaszczyzny zespolonej gdy γ>0: "
            "ω_± = ±√(ω₀²-γ²/4) + iγ/2. "
            "POMOST między trzema obiektami: "
            "(1) γ=0: rezonans klasyczny 1/(ω²-ω₀²) — już w bazie. "
            "(2) γ→0⁺: recepta iε Feynmana! (iγω pełni rolę iε). "
            "(3) γ>0: fizyczna szerokość rezonansu = czas życia stanu kwantowego (reguła Breit-Wignera). "
            "Wheel operuje na liczbach rzeczywistych — ω jest rzeczywiste. "
            "Biegun jest zespolony → Wheel NIE trafi w niego przez podstawienie rzeczywiste. "
            "To otwiera pytanie: jak Wheel radzi sobie z biegunami zespolonymi?"
        ),
        notes="KLUCZOWY: γ→0 to ciągłe przejście do rezonans↔iε. Bieguny zespolone to nowe terytorium dla Wheel.",
    ))

    # ── QFT: t-kanał Mandelstama — osobliwość przy pędzie przekazanym t=0 ──────

    t_man = sp.Symbol("t_man", real=True)   # zmienna Mandelstama t

    db.append(PhysicsEquation(
        name="Amplituda t-kanału 1/(t-m²)",
        domain="QFT",
        expression=sp.Integer(1) / (t_man - m**2),
        variables=[t_man],
        parameters=[m],
        priority=Priority.DIVISION,
        known_singular=[
            {"var": t_man, "value": m**2,
             "description": "t=m² — cząstka wymieniana on-shell (limit Coulomba gdy m→0, t→0)"},
            {"var": t_man, "value": sp.S.Zero,
             "description": "t=0 przy m=0 — Coulomb limit: wymiana bezmasowego bozonu przy zerowym pędzie"},
        ],
        physical_meaning=(
            "Amplituda w t-kanale rozpraszania 2→2: M ~ 1/(t-m²), t=(p1-p3)². "
            "t jest pędem przekazanym — zawsze t≤0 dla fizycznego rozpraszania. "
            "RÓŻNICA od s-kanału (amplituda Comptona): "
            "s > 0 (energia w środku masy), t ≤ 0 (pęd przekazany). "
            "Przy m→0: biegun przy t=0 — limit Coulomba QED (potencjał 1/q²). "
            "Wheel: 1/(t-m²) przy t=m² → ⊥. Przy t=0, m=0 → ⊥. "
            "Uzupełnia s-kanał (amplituda Comptona) — pełny obraz zmiennych Mandelstama s,t."
        ),
        notes="Dopełnia s-kanał: teraz mamy s i t. u-kanał = s+t+u=Σm² — nie dodaje nowej struktury.",
    ))

    # ── THERMO: Entropia Bekenstein-Hawkinga — pochodna dS/dM ─────────────────

    db.append(PhysicsEquation(
        name="Entropia BH — dS/dM = 1/T_H ~ M",
        domain="THERMO",
        expression=8 * sp.pi * G * M * kB / (hbar * c**3),
        variables=[M],
        parameters=[G, kB, hbar, c],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": M, "value": sp.S.Zero,
             "description": "M=0 — brak czarnej dziury, dS/dM → 0 (ale T_H=1/(dS/dM)→⊥)"},
        ],
        physical_meaning=(
            "Entropia Bekenstein-Hawkinga: S_BH = A/(4l_P²) = 4πGM²k_B/(ħc). "
            "dS/dM = 8πGMk_B/(ħc³) = 1/T_H — termodynamiczna definicja temperatury. "
            "Samo S_BH jest skończone i dobrze zdefiniowane (brak osobliwości w S). "
            "Ale: T_H = (dS/dM)^(-1) = ħc³/(8πGMk_B) → ⊥ przy M=0. "
            "POWIĄZANIE z temperaturą Hawkinga (już w bazie): dS/dM = 1/T_H. "
            "dS/dM → 0 gdy M→0: entropia rośnie coraz wolniej przy ewaporacji. "
            "Wheel: dS/dM(M=0) = 0 (skończone!), ale T_H = 1/(dS/dM) = ⊥. "
            "To pokazuje jak ⊥ propaguje przez odwracanie: skończone → ⊥ przez /0."
        ),
        notes="dS/dM jest SKOŃCZONE przy M=0 (=0). Ale 1/(dS/dM)=T_H=⊥. Ćwiczenie z propagacji ⊥.",
    ))

    # ── THERMO: Gęstość stanów van Hove'a ─────────────────────────────────────

    E_sym  = sp.Symbol("E",   real=True)
    E_c    = sp.Symbol("E_c", real=True)   # energia krytyczna

    db.append(PhysicsEquation(
        name="Gęstość stanów van Hove'a D(E) ~ 1/√|E-E_c|",
        domain="THERMO",
        expression=sp.Integer(1) / sp.sqrt(sp.Abs(E_sym - E_c)),
        variables=[E_sym],
        parameters=[E_c],
        priority=Priority.SINGULARITY,
        known_singular=[
            {"var": E_sym, "value": E_c,
             "description": "E=E_c — punkt van Hove'a: krzywizna pasma energetycznego = 0, D(E)→∞"},
        ],
        physical_meaning=(
            "D(E) ~ 1/√|E-E_c| — gęstość stanów przy punktach van Hove'a w ciele stałym. "
            "Pojawia się gdy ∇_k E(k) = 0 (płaskie dno lub wierzchołek pasma). "
            "Fizycznie: nieskończona gęstość stanów = nagromadzenie orbitali o tej samej energii. "
            "Bezpośrednia konsekwencja dla: nadprzewodnictwa (BCS), efektu van Hove'a w optyce, "
            "anomalii w cieple właściwym. "
            "PORÓWNANIE z ciepłem właściwym ~1/|T-Tc|: "
            "tamto dywerguje jako 1/x (wykładnik α=1, mean field), "
            "to jako 1/√x (wykładnik α=1/2 — inne universality class). "
            "Wheel: D(E_c) = ⊥ — punkt niestabilności w strukturze pasmowej."
        ),
        notes="Wykładnik 1/2 (van Hove) vs 1 (mean field ciepło właściwe) — Wheel traktuje tak samo: ⊥.",
    ))

    # ── MATH: sinc²(x) = sin²(x)/x² — kontrprzykład kwadratowy ───────────────

    db.append(PhysicsEquation(
        name="sinc²(x) = sin²(x)/x²",
        domain="MATH",
        expression=sp.sin(x)**2 / x**2,
        variables=[x],
        parameters=[],
        priority=Priority.LIMIT,
        known_singular=[
            {"var": x, "value": sp.S.Zero,
             "description": "Forma 0/0 — granica = 1 (tak samo jak sinc, bo lim sinc²=lim²sinc=1²=1)"},
        ],
        physical_meaning=(
            "sinc²(x) = sin²(x)/x² — kwadrat funkcji sinc. "
            "Pojawia się w: intensywności dyfrakcji przez szczelinę (I ~ sinc²), "
            "widmie mocy sygnału prostokątnego, korelacji sygnałów. "
            "Granica: lim(x→0) sin²(x)/x² = [lim sinc(x)]² = 1² = 1. "
            "Wheel: sin²(0)/0² = 0/0 → ⊥. "
            "TEST dla wheel_calculus.py: czy moduł Taylora uogólnia się na potęgi? "
            "sin(x) ≈ x - x³/6 → sin²(x) ≈ x² - x⁴/3 → sin²(x)/x² ≈ 1 - x²/3 → 1. "
            "Wheel_calculus powinien zwrócić 1, nie ⊥."
        ),
        notes="Kontrprzykład #2 dla wheel_calculus.py — Taylor wyższego rzędu. Wynik powinien: ⊥→1.",
    ))

    # ── MATH: (1-cos(x))/x² — kontrprzykład kosinusowy ───────────────────────

    db.append(PhysicsEquation(
        name="(1 - cos(x))/x²",
        domain="MATH",
        expression=(1 - sp.cos(x)) / x**2,
        variables=[x],
        parameters=[],
        priority=Priority.LIMIT,
        known_singular=[
            {"var": x, "value": sp.S.Zero,
             "description": "Forma 0/0 — granica = 1/2 (inny wynik niż sinc!)"},
        ],
        physical_meaning=(
            "(1-cos(x))/x² — pojawia się w: "
            "rozwinięciu energii kinetycznej (1-cos(ka)) w modelu tight-binding, "
            "fazie akumulowanej przez qubit (geometryczna faza Berriego), "
            "poprawkach do dyspersji fal. "
            "Granica: lim(x→0) (1-cos(x))/x² = 1/2. "
            "Dowód: cos(x) ≈ 1 - x²/2 + x⁴/24 → 1-cos(x) ≈ x²/2 → (1-cos)/x² → 1/2. "
            "Wheel: (1-cos(0))/0² = 0/0 → ⊥. "
            "KLUCZOWA RÓŻNICA od sinc i sinc²: granica ≠ 1, granica = 1/2. "
            "Test dla wheel_calculus.py: czy moduł zwróci 1/2 a nie ⊥?"
        ),
        notes="Kontrprzykład #3 dla wheel_calculus.py — granica = 1/2, nie 1. Inny rozwinięcie Taylora.",
    ))

    return db


# ─── Interfejs bazy ───────────────────────────────────────────────────────────

class EquationsDB:
    """Interfejs do bazy równań fizycznych."""

    def __init__(self):
        self._db = build_database()

    def all(self) -> list[PhysicsEquation]:
        return self._db

    def by_priority(self, p: Priority) -> list[PhysicsEquation]:
        return [eq for eq in self._db if eq.priority == p]

    def by_domain(self, domain: str) -> list[PhysicsEquation]:
        return [eq for eq in self._db if eq.domain == domain.upper()]

    def with_singularities(self) -> list[PhysicsEquation]:
        return [eq for eq in self._db if eq.known_singular]

    def print_catalogue(self) -> None:
        print("═" * 68)
        print("  WHEELPHYSICS — Katalog równań fizycznych")
        print("═" * 68)

        domains = {}
        for eq in self._db:
            domains.setdefault(eq.domain, []).append(eq)

        total_singular = sum(len(eq.known_singular) for eq in self._db)

        for domain, eqs in sorted(domains.items()):
            print(f"\n  [{domain}]")
            for eq in eqs:
                print(f"    {eq.one_liner()}")

        print(f"\n  Razem: {len(self._db)} równań | {total_singular} znanych osobliwości")
        print("═" * 68)

    def stats(self) -> dict:
        return {
            "total":          len(self._db),
            "priority_1":     len(self.by_priority(Priority.DIVISION)),
            "priority_2":     len(self.by_priority(Priority.SINGULARITY)),
            "priority_3":     len(self.by_priority(Priority.LIMIT)),
            "total_singular": sum(len(eq.known_singular) for eq in self._db),
            "domains":        list({eq.domain for eq in self._db}),
        }


if __name__ == "__main__":
    db = EquationsDB()
    db.print_catalogue()
    print()
    stats = db.stats()
    print(f"  Statystyki: {stats}")