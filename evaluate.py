"""
evaluate.py — Évaluation et comparaison Q-Learning vs Baseline
Projet IAD & SMA 2025-2026
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from simulation import IntersectionEnv
from agent import QLearningAgent, BaselineAgent

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_EVAL_EPISODES = 200
STEPS_PER_EP    = 200
LAMBDAS_BALANCED   = (0.4, 0.4, 0.4, 0.4)
LAMBDAS_ASYMMETRIC = (0.7, 0.7, 0.2, 0.2)


def evaluate_agent(agent, env, n_episodes=N_EVAL_EPISODES, steps=STEPS_PER_EP,
                   use_greedy=True):
    """
    Évalue un agent sur n_episodes épisodes.
    use_greedy=True : politique greedy (pas d'exploration).
    Retourne : (récompenses moyennes par épisode, attentes moyennes par pas)
    """
    rewards_list = []
    wait_list    = []

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        total_wait   = 0

        for _ in range(steps):
            if use_greedy and hasattr(agent, 'epsilon'):
                # Sauvegarde et mise à zéro de ε pour éval pure
                old_eps = agent.epsilon
                agent.epsilon = 0.0
                action = agent.select_action(state)
                agent.epsilon = old_eps
            else:
                action = agent.select_action(state)

            state, reward, _ = env.step(action)
            total_reward += reward
            total_wait   += env.get_total_waiting()

        rewards_list.append(total_reward)
        wait_list.append(total_wait / steps)

    return rewards_list, wait_list


def load_agent(scenario_name):
    path = f"{OUTPUT_DIR}/agent_{scenario_name}.pkl"
    if not os.path.exists(path):
        print(f"[!] Agent non trouvé : {path}. Lancez d'abord train.py")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def run_sensitivity_analysis(lambdas, scenario_name, seed=99):
    """
    Analyse de sensibilité sur α et ε (epsilon_start).
    Entraîne plusieurs agents avec différentes valeurs et compare.
    """
    np.random.seed(seed)
    N_EP    = 400
    STEPS   = 150

    alphas   = [0.01, 0.1, 0.3, 0.5]
    colors_a = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Analyse de sensibilité — {scenario_name}", fontsize=13, fontweight='bold')

    # ── Sensibilité à α ──────────────────────────────────────────────────────
    ax = axes[0]
    for alpha, color in zip(alphas, colors_a):
        env   = IntersectionEnv(lambdas=lambdas)
        agent = QLearningAgent(alpha=alpha, gamma=0.95, epsilon=1.0,
                               epsilon_min=0.05, epsilon_decay=0.99)
        ep_rewards = []
        for ep in range(N_EP):
            state = env.reset()
            total_r = 0
            for _ in range(STEPS):
                a       = agent.select_action(state)
                ns, r, _ = env.step(a)
                agent.update(state, a, r, ns)
                state   = ns
                total_r += r
            agent.decay_epsilon()
            ep_rewards.append(total_r)

        smoothed = np.convolve(ep_rewards, np.ones(20)/20, mode='valid')
        ax.plot(smoothed, label=f"α={alpha}", color=color, linewidth=1.8)

    ax.set_title("Influence du taux d'apprentissage α")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Récompense (moy. glissante 20 ep)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Sensibilité à ε (decay) ──────────────────────────────────────────────
    ax = axes[1]
    decays   = [0.990, 0.995, 0.999]
    colors_e = ['#9467bd', '#8c564b', '#17becf']

    for decay, color in zip(decays, colors_e):
        env   = IntersectionEnv(lambdas=lambdas)
        agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=1.0,
                               epsilon_min=0.05, epsilon_decay=decay)
        ep_rewards = []
        for ep in range(N_EP):
            state = env.reset()
            total_r = 0
            for _ in range(STEPS):
                a       = agent.select_action(state)
                ns, r, _ = env.step(a)
                agent.update(state, a, r, ns)
                state   = ns
                total_r += r
            agent.decay_epsilon()
            ep_rewards.append(total_r)

        smoothed = np.convolve(ep_rewards, np.ones(20)/20, mode='valid')
        ax.plot(smoothed, label=f"decay={decay}", color=color, linewidth=1.8)

    ax.set_title("Influence de la décroissance de ε")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Récompense (moy. glissante 20 ep)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig3_sensitivity_{scenario_name}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Sauvegardé : {path}")


def compare_and_plot(scenario_name, lambdas, seed=77):
    """Compare Q-Learning vs Baseline et génère la figure de comparaison."""
    np.random.seed(seed)

    agent_ql = load_agent(scenario_name)
    if agent_ql is None:
        # Entraîne un nouvel agent si non trouvé
        from train import train
        _, _, _, agent_ql = train(lambdas, scenario_name)

    baseline = BaselineAgent(period=5)
    env_ql   = IntersectionEnv(lambdas=lambdas)
    env_bl   = IntersectionEnv(lambdas=lambdas)

    rewards_ql, wait_ql = evaluate_agent(agent_ql, env_ql, use_greedy=True)
    rewards_bl, wait_bl = evaluate_agent(baseline, env_bl, use_greedy=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    title_map = {
        "balanced": "Trafic équilibré (λ=0.4)",
        "asymmetric": "Trafic asymétrique (λ_NS=0.7, λ_EW=0.2)"
    }
    fig.suptitle(f"Comparaison Q-Learning vs Baseline — {title_map.get(scenario_name, scenario_name)}",
                 fontsize=13, fontweight='bold')

    episodes = np.arange(1, N_EVAL_EPISODES + 1)

    # Récompense
    ax = axes[0]
    ax.plot(episodes, rewards_ql, color='steelblue', alpha=0.6, linewidth=1)
    ax.axhline(np.mean(rewards_ql), color='steelblue', linewidth=2.5,
               linestyle='--', label=f"Q-Learning (moy={np.mean(rewards_ql):.0f})")
    ax.plot(episodes, rewards_bl, color='tomato', alpha=0.6, linewidth=1)
    ax.axhline(np.mean(rewards_bl), color='tomato', linewidth=2.5,
               linestyle='--', label=f"Baseline fixe (moy={np.mean(rewards_bl):.0f})")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Récompense cumulée")
    ax.set_title("Récompense cumulée par épisode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Attente moyenne
    ax = axes[1]
    ax.plot(episodes, wait_ql, color='steelblue', alpha=0.6, linewidth=1)
    ax.axhline(np.mean(wait_ql), color='steelblue', linewidth=2.5,
               linestyle='--', label=f"Q-Learning (moy={np.mean(wait_ql):.2f})")
    ax.plot(episodes, wait_bl, color='tomato', alpha=0.6, linewidth=1)
    ax.axhline(np.mean(wait_bl), color='tomato', linewidth=2.5,
               linestyle='--', label=f"Baseline fixe (moy={np.mean(wait_bl):.2f})")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Véhicules en attente (moyenne/pas)")
    ax.set_title("Attente moyenne par pas de temps")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fig4_comparison_{scenario_name}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Sauvegardé : {path}")

    print(f"\n── Résultats [{scenario_name}] ──")
    print(f"  Q-Learning   | Récompense moy. : {np.mean(rewards_ql):8.1f} | "
          f"Attente moy. : {np.mean(wait_ql):.3f} veh/pas")
    print(f"  Baseline     | Récompense moy. : {np.mean(rewards_bl):8.1f} | "
          f"Attente moy. : {np.mean(wait_bl):.3f} veh/pas")
    gain = (np.mean(wait_bl) - np.mean(wait_ql)) / np.mean(wait_bl) * 100
    print(f"  Gain Q-Learning vs Baseline : {gain:.1f}% de réduction de l'attente\n")

    return {
        "rewards_ql": rewards_ql, "rewards_bl": rewards_bl,
        "wait_ql": wait_ql, "wait_bl": wait_bl, "gain": gain
    }


if __name__ == "__main__":
    print("=" * 60)
    print("ÉVALUATION — Scénario 1 : Trafic équilibré")
    print("=" * 60)
    compare_and_plot("balanced", LAMBDAS_BALANCED)

    print("=" * 60)
    print("ÉVALUATION — Scénario 2 : Trafic asymétrique")
    print("=" * 60)
    compare_and_plot("asymmetric", LAMBDAS_ASYMMETRIC)

    print("=" * 60)
    print("ANALYSE DE SENSIBILITÉ")
    print("=" * 60)
    run_sensitivity_analysis(LAMBDAS_BALANCED, "balanced")
    run_sensitivity_analysis(LAMBDAS_ASYMMETRIC, "asymmetric")

    print("\nÉvaluation terminée. Figures dans le dossier outputs/")
