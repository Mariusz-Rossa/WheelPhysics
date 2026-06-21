# Copyright (c) 2026 Mariusz "Vidi" Rossa
# Licensed under the MIT License — see LICENSE file for details.

"""
main.py — WheelPhysics: entry point

Usage:
  python main.py                  # full analysis
  python main.py --quick          # only summaries
  python main.py --module gr      # only GR (General Relativity)
  python main.py --module qft     # only QFT
  python main.py --module viz     # only plots
  python main.py --module calculus  # wheel_calculus + save results
  python main.py --log            # show results log (wheel_results.json)
  python main.py --calculus-log   # show calculus log (calculus_results.json)
  python main.py --regen-log      # recompute log from scratch
  python main.py --db             # equations catalogue
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
║  Investigating theoretical physics singularities             ║
║  through the lens of Wheel Algebra (Carlström 2004)          ║
║                                                              ║
║  Hypothesis: singularities are an artifact of a bad          ║
║  mathematical framework — Wheel Algebra passes through them  ║
║  smoothly, assigning them the value ⊥ instead of ∞           ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_consistency() -> None:
    print("\n" + "▓" * 64)
    print("  MODULE: System consistency")
    print("▓" * 64)
    ConsistencyChecker().run_standard_suite()


def run_calculus(max_order: int = 8) -> None:
    print("\n" + "▓" * 64)
    print("  MODULE: wheel_calculus — Wheel analytical extension")
    print("  Five-fold division: finite | removable | alg. pole | log. pole | ⊥ fallback")
    print("▓" * 64)
    stats = ConsistencyChecker().run_calculus_suite(max_order=max_order)
    print(f"\n  wheel_calculus global result: {stats['correct']}/{stats['total']}  ({stats['score']:.0%})")

    # Save results to calculus_results.json
    print("\n  Saving results to calculus_results.json...")
    calc_log = CalculusLog()
    generate_calculus_log(calc_log, verbose=True)
    calc_log.print_summary()


def run_viz() -> None:
    print("\n" + "▓" * 64)
    print("  MODULE: Visual comparisons")
    print("▓" * 64)
    Comparator().run_showcase()


def run_log(regen: bool = False) -> None:
    print("\n" + "▓" * 64)
    print("  MODULE: Results log")
    print("▓" * 64)
    log = ResultsLog()
    if regen or not log.all():
        print("  Generating log from actual analysis...\n")
        generate_log(log)
    log.print_summary()


def run_db() -> None:
    print("\n" + "▓" * 64)
    print("  MODULE: Equations catalogue")
    print("▓" * 64)
    EquationsDB().print_catalogue()


def run_research_summary() -> None:
    print("\n" + "╔" + "═" * 62 + "╗")
    print("║  RESEARCH SUMMARY — WheelPhysics v0.1" + " " * 23 + "║")
    print("╚" + "═" * 62 + "╝")

    # Load verdicts from actual log
    log = ResultsLog()
    if not log.all():
        generate_log(log, verbose=False)

    b = len(log.by_verdict("BOTTOM"))
    f = len(log.by_verdict("FINITE"))
    m = len(log.by_verdict("MIXED"))
    total = len(log.all())

    print(f"""
  Analyzed        : {total} equations
  Result ⊥        : {b} ({b/total:.0%}) — singularities correctly detected
  Finite result   : {f} ({f/total:.0%}) — no singularities
  MIXED result    : {m} ({m/total:.0%}) — mixed (e.g., Kretschmann)
    """)

    findings = {
        "Confirmed": [
            "K(r=0)=⊥, K(r=r_s) finite — Wheel distinguishes singularity types via invariant",
            "Every Feynman propagator yields ⊥ exactly on-shell",
            "m=0 propagator: ⊥ at p=0 — algebraic prohibition of rest state",
            "Friedmann continuous through a=0 (with ⊥) — a<0 mathematically permitted",
            "Boltzmann distribution: T=0 → ⊥ (consistent with 3rd law of thermodynamics)",
        ],
        "Working hypotheses": [
            "⊥ on-shell = algebraic definition of observability in QFT",
            "a<0 in Friedmann = time before the Big Bang",
            "Feynman's iε prescription = classical substitute for the lack of Wheel in algebra",
            "K(-r) = K(+r) — singularity symmetry suggests mirror solution",
        ],
        "Method limitations (discovered and addressed)": [
            "Wheel ≠ limit theory — solved by wheel_calculus.py (five-fold division)",
            "sinc: Wheel=⊥, wheel_calculus=1  |  (1-cos)/x²: Wheel=⊥, wheel_calculus=1/2",
            "UV divergences live in ∫d⁴k integrals — Wheel operates pointwise (open)",
            "Christoffel symbols are frame-dependent: their ⊥ might be an artifact",
        ],
        "Next steps": [
            "Metric tensor in wheel algebra — a natural counterpart?",
            "Wheel + dimensional regularization — ⊥ as a substitute for ε=(4-d)/2?",
            "Wheel on Riemannian manifolds — geometry + wheel algebra",
            "Physical interpretation of ⊥: is it a state, a prohibition, or something else?",
            "Complex poles (damped Green) — open",
        ],
    }

    for section, items in findings.items():
        icon = {"Confirmed": "✓", "Working hypotheses": "?",
                "Method limitations (discovered and addressed)": "!", "Next steps": "→"}.get(section, "•")
        print(f"  [{icon}] {section}:")
        for item in items:
            print(f"      • {item}")

    print("\n" + "═" * 64)
    print("  Project status: core/ ✓  scanner/ ✓  physics/ ✓  viz/ ✓  calculus/ ✓  log. poles/ ✓")
    print("═" * 64 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", "-m",
                        choices=["gr", "qft", "viz", "consistency", "log", "db", "calculus"])
    parser.add_argument("--quick",    "-q", action="store_true")
    parser.add_argument("--log",            action="store_true")
    parser.add_argument("--regen-log",      action="store_true",
                        help="Recompute log from scratch from actual analysis")
    parser.add_argument("--calculus-log",   action="store_true",
                        help="Show wheel_calculus results log")
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

    # Full analysis
    run_consistency()
    run_gr()
    run_qft()
    run_calculus()        
    run_viz()
    run_log(regen=True)   # ← always recompute log from scratch after analysis
    run_research_summary()


if __name__ == "__main__":
    main()