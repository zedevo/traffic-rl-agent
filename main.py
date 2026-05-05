"""
main.py — Script principal : entraînement + évaluation complète
Projet IAD & SMA 2025-2026
Usage : python main.py
"""

import numpy as np
import os

from train import train, plot_training, LAMBDAS_BALANCED, LAMBDAS_ASYMMETRIC
from evaluate import compare_and_plot, run_sensitivity_analysis

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Gestion intelligente du trafic — Q-Learning Agent       ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── Entraînement ────────────────────────────────────────────────────────
    print("▶ PHASE 1 : Entraînement\n")
    rewards_bal, wait_bal, eps_hist, agent_bal = train(LAMBDAS_BALANCED, "balanced")
    rewards_asym, wait_asym, _, agent_asym     = train(LAMBDAS_ASYMMETRIC, "asymmetric")
    plot_training(rewards_bal, rewards_asym, eps_hist)

    # ── Évaluation & comparaison ─────────────────────────────────────────────
    print("\n▶ PHASE 2 : Évaluation comparative\n")
    results_bal  = compare_and_plot("balanced",   LAMBDAS_BALANCED)
    results_asym = compare_and_plot("asymmetric", LAMBDAS_ASYMMETRIC)

    # ── Analyse de sensibilité ───────────────────────────────────────────────
    print("\n▶ PHASE 3 : Analyse de sensibilité\n")
    run_sensitivity_analysis(LAMBDAS_BALANCED,   "balanced")
    run_sensitivity_analysis(LAMBDAS_ASYMMETRIC, "asymmetric")

    # ── Résumé final ─────────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║                  RÉSUMÉ DES RÉSULTATS                    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Trafic équilibré  — Gain Q-Learning vs Baseline : "
          f"{results_bal['gain']:.1f}%")
    print(f"  Trafic asymétrique — Gain Q-Learning vs Baseline : "
          f"{results_asym['gain']:.1f}%")
    print(f"\n  Figures générées dans : {OUTPUT_DIR}/")
    print("  Entraînement complet terminé ✓")
