"""
train.py — Entraînement de l'agent Q-Learning
Projet IAD & SMA 2025-2026
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from simulation import IntersectionEnv
from agent import QLearningAgent

# ── Hyperparamètres ──────────────────────────────────────────────────────────
N_EPISODES    = 1000       # nombre d'épisodes d'entraînement
STEPS_PER_EP  = 200        # durée d'un épisode (pas de temps)
ALPHA         = 0.1        # taux d'apprentissage
GAMMA         = 0.95       # facteur d'actualisation
EPSILON_START = 1.0        # exploration initiale
EPSILON_MIN   = 0.05       # exploration minimale
EPSILON_DECAY = 0.995      # décroissance de ε

# Taux d'arrivée Poisson [N, S, E, O]
LAMBDAS_BALANCED   = (0.4, 0.4, 0.4, 0.4)   # trafic équilibré
LAMBDAS_ASYMMETRIC = (0.7, 0.7, 0.2, 0.2)   # charge asymétrique N/S dominante

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train(lambdas, scenario_name, seed=42):
    """
    Entraîne l'agent Q-Learning sur un scénario donné.
    Retourne les récompenses par épisode et l'agent entraîné.
    """
    np.random.seed(seed)
    env   = IntersectionEnv(lambdas=lambdas)
    agent = QLearningAgent(
        alpha=ALPHA, gamma=GAMMA,
        epsilon=EPSILON_START, epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY
    )

    episode_rewards = []
    episode_avg_wait = []
    epsilon_history  = []

    for ep in range(N_EPISODES):
        state = env.reset()
        total_reward = 0
        total_wait   = 0

        for _ in range(STEPS_PER_EP):
            action = agent.select_action(state)
            next_state, reward, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            total_wait   += env.get_total_waiting()

        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_avg_wait.append(total_wait / STEPS_PER_EP)
        epsilon_history.append(agent.epsilon)

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(episode_rewards[-100:])
            print(f"[{scenario_name}] Épisode {ep+1:4d} | "
                  f"Récompense moy. (100 ep) : {avg_r:8.1f} | "
                  f"ε={agent.epsilon:.3f} | "
                  f"États Q explorés : {agent.get_q_table_size()}")

    # Sauvegarde de l'agent
    with open(f"{OUTPUT_DIR}/agent_{scenario_name}.pkl", "wb") as f:
        pickle.dump(agent, f)

    return episode_rewards, episode_avg_wait, epsilon_history, agent


def smooth(data, window=20):
    """Moyenne glissante pour lisser les courbes."""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_training(rewards_bal, rewards_asym, eps_hist, output_dir=OUTPUT_DIR):
    """Génère les figures de convergence."""

    # ── Figure 1 : Récompense cumulée par épisode ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Convergence de l'agent Q-Learning", fontsize=14, fontweight='bold')

    for ax, rewards, title, color in zip(
        axes,
        [rewards_bal, rewards_asym],
        ["Trafic équilibré (λ=0.4)", "Trafic asymétrique (λ_NS=0.7, λ_EW=0.2)"],
        ["steelblue", "darkorange"]
    ):
        episodes = np.arange(1, len(rewards) + 1)
        ax.plot(episodes, rewards, alpha=0.3, color=color, linewidth=0.8)
        smoothed = smooth(rewards, 30)
        ax.plot(np.arange(30, len(rewards) + 1), smoothed,
                color=color, linewidth=2, label="Moy. glissante (30 ep)")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Épisode")
        ax.set_ylabel("Récompense cumulée")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_convergence.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Sauvegardé : {output_dir}/fig1_convergence.png")

    # ── Figure 2 : Décroissance de ε ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eps_hist, color='crimson', linewidth=2)
    ax.set_title("Décroissance de ε (exploration → exploitation)", fontsize=12)
    ax.set_xlabel("Épisode")
    ax.set_ylabel("ε (epsilon)")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.05, linestyle='--', color='gray', label='ε_min = 0.05')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2_epsilon_decay.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Sauvegardé : {output_dir}/fig2_epsilon_decay.png")


if __name__ == "__main__":
    print("=" * 60)
    print("ENTRAÎNEMENT — Scénario 1 : Trafic équilibré")
    print("=" * 60)
    rewards_bal, wait_bal, eps_hist, agent_bal = train(
        LAMBDAS_BALANCED, "balanced"
    )

    print()
    print("=" * 60)
    print("ENTRAÎNEMENT — Scénario 2 : Trafic asymétrique")
    print("=" * 60)
    rewards_asym, wait_asym, _, agent_asym = train(
        LAMBDAS_ASYMMETRIC, "asymmetric"
    )

    print()
    print("Génération des figures...")
    plot_training(rewards_bal, rewards_asym, eps_hist)
    print("Entraînement terminé.")
