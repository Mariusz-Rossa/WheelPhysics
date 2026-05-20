# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
results_log.py — zapis i odczyt wyników analizy Wheel

Wyniki są GENEROWANE przez faktyczną analizę równań z equations_db,
nie hardcodowane ręcznie.
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
    Wpis z analizy wheel_calculus — czwórpodział osobliwości.
    Zapisywany do calculus_results.json.
    """
    timestamp:      str
    name:           str
    expression_str: str
    variables:      str          # np. "x→0" lub "p→m, m→0"
    wheel_result:   str          # zawsze "⊥" (calculus uruchamiany gdy Wheel dał ⊥)
    calculus_type:  str          # "REMOVABLE" | "POLE" | "BOTTOM" | "FINITE"
    limit_value:    str          # wartość granicy lub "⊥"
    taylor_order:   int | None   # rząd rozwinięcia (REMOVABLE) lub rząd bieguna (POLE)
    series_hint:    str          # fragment rozwinięcia Taylora lub Laurent hint
    pole_order:     int | None   # rząd bieguna (tylko POLE), inaczej None
    residue:        str          # residuum jako string (tylko POLE rząd=1), inaczej ""
    note:           str = ""


@dataclass
class AnalysisEntry:
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
        BOTTOM  — wszystkie znane osobliwości dają ⊥
        FINITE  — żadna nie daje ⊥ (Wheel nie wykrył)
        MIXED   — część ⊥, część skończona I te skończone nie są artefaktami
        
        Jeśli skończone wyniki mają w nocie słowo 'artefakt'/'regularny'/
        'skończony' — traktujemy je jako poprawne (nie obniżamy verdyktu).
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

        # Mamy mix — sprawdź czy skończone to artefakty
        artifact_keywords = ("artefakt", "regularny", "skończony", "horyzont regularny")
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
            print("  Log jest pusty. Uruchom: python main.py --regen-log")
            return

        print("═" * 64)
        print(f"  WHEELPHYSICS — Log wyników  [{len(self._entries)} wpisów]")
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
        print(f"\n  Verdykty: ⊥={b}  ✓={f}  ±={m}")
        print("═" * 64)


class CalculusLog:
    """
    Log wyników wheel_calculus — trójpodział osobliwości.
    Zapisuje do calculus_results.json obok wheel_results.json.
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
            print("  Log calculus jest pusty. Uruchom: python main.py --module calculus")
            return

        rem  = self.by_type("REMOVABLE")
        pol  = self.by_type("POLE")
        bot  = self.by_type("BOTTOM")
        fin  = self.by_type("FINITE")
        total = len(self._entries)

        print("═" * 64)
        print(f"  WHEEL CALCULUS — Log wyników  [{total} wpisów]")
        print(f"  {self.path}")
        print("═" * 64)

        if rem:
            print(f"\n  [USUWALNE — Wheel=⊥ ale lim istnieje]  ({len(rem)})")
            for e in rem:
                print(f"    ⊥→{e.limit_value:<8}  {e.name:<40}  rząd={e.taylor_order}")
                if e.series_hint:
                    hint = e.series_hint[:60] + "…" if len(e.series_hint) > 60 else e.series_hint
                    print(f"             Taylor: {hint}")

        if pol:
            print(f"\n  [BIEGUNY — rząd + residuum (Cauchy)]  ({len(pol)})")
            for e in pol:
                res_str = f"res={e.residue}" if e.residue else "res=N/A"
                print(f"    ⊥  rząd={e.pole_order}  {res_str:<20}  {e.name}")
                if e.series_hint:
                    hint = e.series_hint[:60] + "…" if len(e.series_hint) > 60 else e.series_hint
                    print(f"             Laurent: {hint}")

        if bot:
            print(f"\n  [BOTTOM — ⊥ bez struktury (fallback)]  ({len(bot)})")
            for e in bot:
                print(f"    ⊥         {e.name:<40}  {e.variables}")

        if fin:
            print(f"\n  [REGULARNE — Wheel OK]  ({len(fin)})")
            for e in fin:
                print(f"    ✓={e.limit_value:<8}  {e.name:<40}")

        score = (len(rem) + len(pol) + len(fin)) / total if total else 0
        print(f"\n  Czwórpodział: USUWALNE={len(rem)}  BIEGUNY={len(pol)}  BOTTOM={len(bot)}  REGULARNE={len(fin)}")
        ts = self._entries[-1].timestamp[:10] if self._entries else "—"
        print(f"  Ostatnia analiza: {ts}")
        print("═" * 64)


# ─── Generator — produkuje log z faktycznej analizy ───────────────────────────

# Hipotezy badawcze per równanie — jedyne miejsce gdzie są hardcoded,
# bo to interpretacja, nie wynik obliczeniowy.
_HYPOTHESES = {
    "g_rr Schwarzschilda": (
        "g_rr(r_s)=⊥ i g_rr(0)=⊥, ale K(r_s) skończony. "
        "Wheel sam w sobie nie rozróżnia artefaktu od osobliwości fizycznej — "
        "dopiero zestawienie g_rr z K to pokazuje.",
        "Kluczowy wynik: potrzeba niezmiennika K do odróżnienia typów osobliwości"
    ),
    "g_tt Schwarzschilda": (
        "g_tt(r=0)=⊥. Przy r=r_s: g_tt=0 (nie ⊥) — horyzont jest regularny dla g_tt.",
        ""
    ),
    "Niezmiennik Kretschmanna": (
        "K(r=0)=⊥ — prawdziwa osobliwość. K(r=r_s)=12/r_s⁴ — skończony. "
        "Wheel przez K poprawnie odróżnia osobliwość fizyczną od artefaktu układu. "
        "To najsilniejszy wynik projektu w obszarze OTW.",
        "Wynik 100% zgodny z OTW — K jest niezmiennikiem tensorowym"
    ),
    "Symbol Christoffela Γ^t_tr": (
        "Γ=⊥ zarówno przy r=r_s jak i r=0. Symbole Christoffela zależą od układu "
        "współrzędnych — ich ⊥ przy r=r_s to artefakt, nie fizyka.",
        "Potwierdza: Wheel na obiektach zależnych od układu wymaga ostrożności"
    ),
    "Friedmann — człon krzywiznowy": (
        "H²(a=0)=⊥. Wheel dopuszcza a<0 matematycznie — symetria przez a=0. "
        "Hipoteza: a<0 = pre-big-bang. Równania formalne są ciągłe przez osobliwość.",
        "Hipoteza pre-big-bang wymaga interpretacji fizycznej"
    ),
    "Gęstość materii ρ~1/a³": (
        "ρ_materia(a=0)=⊥. Spójne z Friedmannem — Wielki Wybuch jest osobliwością "
        "we wszystkich składnikach równania jednocześnie.",
        ""
    ),
    "Gęstość promieniowania ρ~1/a⁴": (
        "ρ_rad(a=0)=⊥. Silniejsza dywergencja niż materia (a⁻⁴ vs a⁻³), "
        "Wheel traktuje obie tak samo — ⊥.",
        ""
    ),
    "Propagator skalarny Feynmana": (
        "Biegun on-shell (p=±m) → ⊥. Off-shell: skończony i obliczalny. "
        "Recepta iε Feynmana to klasyczny trik omijający biegun w zespolonej płaszczyźnie. "
        "Wheel przechodzi przez biegun bezpośrednio. "
        "Hipoteza: ⊥ on-shell = algebraiczna definicja obserwowalności.",
        "On-shell=⊥ może być głębszą strukturą QFT"
    ),
    "Propagator fotonowy": (
        "k=0 → ⊥. Foton bezmasowy on-shell przy k=0. "
        "IR osobliwość — fizycznie: foton o zerowym pędzie nie istnieje.",
        ""
    ),
    "Propagator fermionowy (uproszczony)": (
        "p=±m_e → ⊥. Elektron i pozyton on-shell. "
        "Licznik (p+m_e) przy p=-m_e daje 0, mianownik też 0 → forma 0/0 → ⊥.",
        "Zarówno cząstka jak i antycząstka oznaczone tym samym ⊥"
    ),
    "Propagator bezmasowego fermiona": (
        "p=0 → ⊥. Bezmasowy fermion (neutryno) nie może mieć p=0 — "
        "porusza się zawsze z prędkością c. "
        "Wheel algebraicznie zakazuje tego stanu, nie postuluje zakazu.",
        "Najczystszy przykład: ⊥ = stan fizycznie niemożliwy"
    ),
    "Rozkład Boltzmanna 1/T": (
        "T=0 → ⊥. Zero bezwzględne jest osobliwością termodynamiczną. "
        "Wheel: przy T=0 prawdopodobieństwo obsadzenia → ⊥ zamiast 0 lub ∞.",
        "Trzecia zasada termodynamiki — T=0 nieosiągalne"
    ),
    "Rozkład Plancka": (
        "T=0 → ⊥ (mianownik exp(∞)-1 → ∞). "
        "Forma 0/0 gdy T→∞ (klasyczna granica Rayleigha-Jeansa = 2k_BT) — "
        "Wheel daje ⊥ zamiast skończonej granicy. "
        "To samo ograniczenie co sinc: Wheel ≠ teoria granic.",
        "Edge case: granica T→∞ istnieje klasycznie, Wheel daje ⊥"
    ),
    "sinc(x) = sin(x)/x": (
        "KLUCZOWY KONTRPRZYKŁAD: sin(0)/0 = 0/0 → ⊥ w Wheel. "
        "Klasycznie: lim(x→0) sin(x)/x = 1 (granica istnieje i jest skończona). "
        "Wheel NIE oblicza granic — podstawia wartość dosłownie. "
        "Fundamentalna granica metody: Wheel jest algebrą (podstawienie punktowe), "
        "nie analizą matematyczną (granice). "
        "Formy 0/0 gdzie granica jest skończona leżą POZA zakresem Wheel.",
        "WAŻNY KONTRPRZYKŁAD — Wheel ≠ teoria granic"
    ),
}


def generate_log(log: ResultsLog, verbose: bool = True) -> None:
    """
    Generuje log przez faktyczną analizę wszystkich równań z equations_db.
    Czyści poprzedni log przed generowaniem.
    """
    from physics.equations_db import EquationsDB
    from core.sympy_extension import wheel_subs
    import sympy as sp

    log.clear()
    db = EquationsDB()

    if verbose:
        print(f"  Analizuję {len(db.all())} równań...\n")

    for eq in db.all():
        singular_points = []

        for sing in eq.known_singular:
            var = sing["var"]
            val = sing["value"]

            # Uruchom faktyczną analizę Wheel
            w = wheel_subs(eq.expression, {var: val})
            w_str = "⊥" if w.is_bottom else str(w.value)[:40]

            # Klasyczny wynik
            try:
                classical = sp.simplify(eq.expression.subs({var: val}))
                cl_str = str(classical)[:40]
            except Exception:
                cl_str = "błąd"

            singular_points.append({
                "var":          str(var),
                "value":        str(val),
                "wheel_result": w_str,
                "classical":    cl_str,
                "note":         sing["description"],
            })

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
            print(f"  [{icon}] {eq.name}")

    if verbose:
        print(f"\n  Zapisano {len(log.all())} wpisów → {log.path}")


def generate_calculus_log(log: "CalculusLog", verbose: bool = True) -> None:
    """
    Uruchamia wheel_calculus na zestawie testowym i zapisuje wyniki.

    Testuje: kontrprzykłady z bazy + prawdziwe bieguny + punkty regularne.
    Wyniki lądują w calculus_results.json.
    """
    from core.wheel_calculus import wheel_limit, RemovableSingularity, PoleSingularity
    import sympy as sp

    log.clear()

    x, m, p, r, r_s = sp.symbols("x m p r r_s", real=True)
    omega, omega0    = sp.symbols("omega omega0", real=True)

    cases = [
        # ── Usuwalne (kontrprzykłady z bazy) ─────────────────────────────
        dict(name="sinc(x) = sin(x)/x",
             expr=sp.sin(x)/x, variables=[(x, sp.S.Zero)],
             note="Kluczowy kontrprzykład — Wheel=⊥, lim=1"),
        dict(name="sinc²(x) = sin²(x)/x²",
             expr=sp.sin(x)**2/x**2, variables=[(x, sp.S.Zero)],
             note="Kwadrat sinc — ta sama granica, wyższy rząd Taylor"),
        dict(name="(1 - cos(x))/x²",
             expr=(1 - sp.cos(x))/x**2, variables=[(x, sp.S.Zero)],
             note="Granica = 1/2 (nie 1!) — inny typ rozwinięcia"),
        dict(name="Rayleigh-Jeans: x/(e^x - 1) gdy x→0",
             expr=x/(sp.exp(x) - 1), variables=[(x, sp.S.Zero)],
             note="x = ħω/kT → 0 gdy T→∞: granica klasyczna Rayleigha-Jeansa"),
        dict(name="(e^x - 1)/x",
             expr=(sp.exp(x) - 1)/x, variables=[(x, sp.S.Zero)],
             note="Pochodna exp w x=0 przez iloraz różnicowy"),
        dict(name="tan(x)/x",
             expr=sp.tan(x)/x, variables=[(x, sp.S.Zero)],
             note="Usuwalna — lim=1"),
        dict(name="(sin(3x))/(sin(5x)) przy x→0",
             expr=sp.sin(3*x)/sp.sin(5*x), variables=[(x, sp.S.Zero)],
             note="Granica = 3/5 — reguła L'Hôpitala lub Taylor"),
        dict(name="(x² - 1)/(x - 1) przy x→1",
             expr=(x**2 - 1)/(x - 1), variables=[(x, sp.S.One)],
             note="Klasyczny przykład osobliwości usuwalnej — lim=2"),
        # ── Bieguny (residue analysis) ────────────────────────────────────
        dict(name="Propagator skalarny 1/(p²-m²) [on-shell]",
             expr=1/(p**2 - m**2), variables=[(p, m)],
             note="Prosty biegun on-shell — res=1/(2m), izomorfizm z rezonansem"),
        dict(name="g_rr Schwarzschilda przy r=r_s",
             expr=1/(1 - r_s/r), variables=[(r, r_s)],
             note="Horyzont zdarzeń — biegun układu współrzędnych, res=r_s"),
        dict(name="Rezonans 1/(ω²-ω₀²) przy ω=ω₀",
             expr=1/(omega**2 - omega0**2), variables=[(omega, omega0)],
             note="Rezonans klasyczny — res=1/(2ω₀), izomorfizm z propagatorem"),
        dict(name="Propagator fotonowy 1/k² przy k=0",
             expr=1/x**2, variables=[(x, sp.S.Zero)],
             note="IR biegun rzędu 2 — residuum niezdefiniowane (Cauchy tylko rząd=1)"),
        # ── Regularne ────────────────────────────────────────────────────
        dict(name="KG euklidesowy 1/(p²+m²) przy p=0, m=1",
             expr=1/(p**2 + m**2), variables=[(p, sp.S.Zero), (m, sp.S.One)],
             note="Brak bieguna na osi rzeczywistej — obrót Wicka"),
        dict(name="Potencjał fotonu V_ph przy r=3r_s/2",
             expr=(1 - r_s/r)/r**2, variables=[(r, sp.Rational(3,2)*r_s)],
             note="Fotosfera Schwarzschilda — punkt regularny"),
    ]

    if verbose:
        print(f"  Analizuję {len(cases)} przypadków wheel_calculus...\n")

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
            order     = result.pole_order   # rząd bieguna
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
                label = f"⊥ BIEGUN[{pole_ord}]{res_part}"
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
        print(f"\n  Zapisano {len(log.all())} wpisów → {log.path}")
        print(f"  Czwórpodział: USUWALNE={rem}  BIEGUNY={pol}  BOTTOM={bot}  REGULARNE={fin}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regen", action="store_true",
                        help="Wygeneruj log od nowa z faktycznej analizy")
    args = parser.parse_args()

    log = ResultsLog()

    if args.regen or not log.all():
        print("  Generuję log z analizy...\n")
        generate_log(log)

    log.print_summary()