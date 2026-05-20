# Copyright (c) 2026 Mariusz Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
main.py — WheelPhysics: punkt wejścia

Użycie:
  python main.py                  # pełna analiza
  python main.py --quick          # tylko podsumowania
  python main.py --module gr      # tylko OTW
  python main.py --module qft     # tylko QFT
  python main.py --module viz     # tylko wykresy
  python main.py --module calculus  # wheel_calculus + zapis wyników
  python main.py --log            # pokaż log wyników (wheel_results.json)
  python main.py --calculus-log   # pokaż log calculus (calculus_results.json)
  python main.py --regen-log      # przelicz log od nowa
  python main.py --db             # katalog równań
"""

import sys
import argparse

from scanner.consistency_checker import ConsistencyChecker
from physics.general_relativity  import run_full_analysis   as run_gr
from physics.quantum             import run_quantum_analysis as run_qft
from physics.equations_db        import EquationsDB
from physics.results_log         import ResultsLog, CalculusLog, generate_log, generate_calculus_log
from viz.comparator              import Comparator


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║           W H E E L P H Y S I C S                           ║
║                                                              ║
║  Badanie osobliwości fizyki teoretycznej                     ║
║  przez pryzmat Wheel Algebra (Carlström 2004)                ║
║                                                              ║
║  Hipoteza: osobliwości to artefakt złego aparatu            ║
║  matematycznego — Wheel Algebra przechodzi przez nie         ║
║  płynnie, przypisując im wartość ⊥ zamiast ∞               ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_consistency() -> None:
    print("\n" + "▓" * 64)
    print("  MODUŁ: Spójność systemu")
    print("▓" * 64)
    ConsistencyChecker().run_standard_suite()


def run_calculus(max_order: int = 8) -> None:
    print("\n" + "▓" * 64)
    print("  MODUŁ: wheel_calculus — rozszerzenie analityczne Wheel")
    print("  Trójpodział: skończone | ⊥ nieusuwalna | ⊥→val usuwalna")
    print("▓" * 64)
    stats = ConsistencyChecker().run_calculus_suite(max_order=max_order)
    print(f"\n  wheel_calculus wynik globalny: {stats['correct']}/{stats['total']}  ({stats['score']:.0%})")

    # Zapisz wyniki do calculus_results.json
    print("\n  Zapisuję wyniki do calculus_results.json...")
    calc_log = CalculusLog()
    generate_calculus_log(calc_log, verbose=True)
    calc_log.print_summary()


def run_viz() -> None:
    print("\n" + "▓" * 64)
    print("  MODUŁ: Porównania wizualne")
    print("▓" * 64)
    Comparator().run_showcase()


def run_log(regen: bool = False) -> None:
    print("\n" + "▓" * 64)
    print("  MODUŁ: Log wyników")
    print("▓" * 64)
    log = ResultsLog()
    if regen or not log.all():
        print("  Generuję log z faktycznej analizy...\n")
        generate_log(log)
    log.print_summary()


def run_db() -> None:
    print("\n" + "▓" * 64)
    print("  MODUŁ: Katalog równań")
    print("▓" * 64)
    EquationsDB().print_catalogue()


def run_research_summary() -> None:
    print("\n" + "╔" + "═" * 62 + "╗")
    print("║  PODSUMOWANIE BADAWCZE — WheelPhysics v0.1" + " " * 19 + "║")
    print("╚" + "═" * 62 + "╝")

    # Wczytaj verdykty z faktycznego logu
    log = ResultsLog()
    if not log.all():
        generate_log(log, verbose=False)

    b = len(log.by_verdict("BOTTOM"))
    f = len(log.by_verdict("FINITE"))
    m = len(log.by_verdict("MIXED"))
    total = len(log.all())

    print(f"""
  Przeanalizowano : {total} równań
  Wynik ⊥         : {b} ({b/total:.0%}) — osobliwości poprawnie wykryte
  Wynik skończony : {f} ({f/total:.0%}) — brak osobliwości
  Wynik MIXED     : {m} ({m/total:.0%}) — mieszany (np. Kretschmann)
    """)

    findings = {
        "Potwierdzone": [
            "K(r=0)=⊥, K(r=r_s) skończony — Wheel rozróżnia typy osobliwości przez niezmiennik",
            "Każdy propagator Feynmana ma ⊥ dokładnie on-shell",
            "Propagator m=0: ⊥ przy p=0 — algebraiczny zakaz stanu spoczynku",
            "Friedmann ciągły przez a=0 (z ⊥) — a<0 matematycznie dozwolone",
            "Rozkład Boltzmanna: T=0 → ⊥ (spójne z III zasadą termodynamiki)",
        ],
        "Hipotezy robocze": [
            "⊥ on-shell = algebraiczna definicja obserwowalności w QFT",
            "a<0 w Friedmannie = czas przed Wielkim Wybuchem",
            "Recepta iε Feynmana = klasyczny substytut za brak Wheel w algebrze",
            "K(-r) = K(+r) — symetria osobliwości sugeruje lustrzane rozwiązanie",
        ],
        "Granice metody (odkryte i zaadresowane)": [
            "Wheel ≠ teoria granic — rozwiązane przez wheel_calculus.py (trójpodział)",
            "sinc: Wheel=⊥, wheel_calculus=1  |  (1-cos)/x²: Wheel=⊥, wheel_calculus=1/2",
            "Dywergencje UV żyją w całkach ∫d⁴k — Wheel operuje punktowo (otwarte)",
            "Symbole Christoffela zależne od układu: ich ⊥ może być artefaktem",
        ],
        "Następne kroki": [
            "Tensor metryczny w algebrze koła — naturalny odpowiednik?",
            "Wheel + regularyzacja wymiarowa — ⊥ jako substytut ε=(4-d)/2?",
            "Wheel na rozmaitościach Riemanna — geometria + algebra koła",
            "Interpretacja fizyczna ⊥: czy to stan, zakaz, czy coś trzeciego?",
            "Bieguny logarytmiczne QCD i zespolone (Green z tłumieniem) — otwarte",
        ],
    }

    for section, items in findings.items():
        icon = {"Potwierdzone": "✓", "Hipotezy robocze": "?",
                "Granice metody (odkryte)": "!", "Następne kroki": "→"}.get(section, "•")
        print(f"  [{icon}] {section}:")
        for item in items:
            print(f"      • {item}")

    print("\n" + "═" * 64)
    print("  Stan projektu: core/ ✓  scanner/ ✓  physics/ ✓  viz/ ✓  calculus/ ✓")
    print("═" * 64 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", "-m",
                        choices=["gr", "qft", "viz", "consistency", "log", "db", "calculus"])
    parser.add_argument("--quick",    "-q", action="store_true")
    parser.add_argument("--log",            action="store_true")
    parser.add_argument("--regen-log",      action="store_true",
                        help="Przelicz log od nowa z faktycznej analizy")
    parser.add_argument("--calculus-log",   action="store_true",
                        help="Pokaż log wyników wheel_calculus")
    parser.add_argument("--db",             action="store_true")
    args = parser.parse_args()

    print(BANNER)

    if args.regen_log:
        run_log(regen=True); return
    if args.calculus_log:
        CalculusLog().print_summary(); return
    if args.log or args.module == "log":
        run_log(); return
    if args.db or args.module == "db":
        run_db(); return
    if args.module == "gr":
        run_gr(); run_research_summary(); return
    if args.module == "qft":
        run_qft(); run_research_summary(); return
    if args.module == "viz":
        run_viz(); return
    if args.module == "consistency":
        run_consistency(); return
    if args.module == "calculus":
        run_calculus(); return
    if args.quick:
        run_db(); run_log(); run_research_summary(); return

    # Pełna analiza
    run_consistency()
    run_gr()
    run_qft()
    run_calculus()        # ← nowy moduł
    run_viz()
    run_log(regen=True)   # ← po analizie zawsze przelicz log od nowa
    run_research_summary()


if __name__ == "__main__":
    main()