"""
simulation.py — Simulation du carrefour à 4 branches
Gestion intelligente du trafic urbain — Projet IAD & SMA 2025-2026
"""

import numpy as np

# Phases du feu
PHASE_NS = 0   # Nord-Sud autorisé
PHASE_EW = 1   # Est-Ouest autorisé
PHASE_ORANGE = 2  # Transition (orange)

PHASE_NAMES = {PHASE_NS: "N/S", PHASE_EW: "E/O", PHASE_ORANGE: "Orange"}

# Directions (indices des files)
NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3
DIRECTION_NAMES = ["Nord", "Sud", "Est", "Ouest"]

# Capacité max d'une file (discrétisation 0..MAX_QUEUE)
MAX_QUEUE = 5
# Débit max de véhicules écoulés par pas de temps (quand feu vert)
FLOW_RATE = 2


class IntersectionEnv:
    """
    Environnement simulant un carrefour à 4 branches avec feux tricolores.

    État : tuple (q_N, q_S, q_E, q_O, phase_courante)
      - q_i ∈ {0,1,2,3,4,5} : file discrétisée (5 = saturée)
      - phase ∈ {0,1,2} : N/S vert, E/O vert, orange

    Actions : {0 = Maintenir la phase, 1 = Changer de phase}
    """

    def __init__(self, lambdas=(0.4, 0.4, 0.4, 0.4), orange_duration=1):
        """
        lambdas : taux d'arrivée Poisson pour [N, S, E, O]
        orange_duration : durée (en pas) de la phase orange
        """
        self.lambdas = np.array(lambdas, dtype=float)
        self.orange_duration = orange_duration
        self.reset()

    def reset(self):
        """Réinitialise l'environnement."""
        self.queues = np.zeros(4, dtype=int)      # [N, S, E, O]
        self.phase = PHASE_NS                      # on démarre en N/S
        self.orange_timer = 0                      # compteur phase orange
        self.total_wait = 0
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        """Retourne l'état courant comme tuple hashable."""
        return tuple(self.queues) + (self.phase,)

    def _arrivals(self):
        """Génère les arrivées de véhicules selon un processus de Poisson."""
        arrivals = np.random.poisson(self.lambdas)
        self.queues = np.minimum(self.queues + arrivals, MAX_QUEUE)

    def _flow(self):
        """
        Écoule les véhicules autorisés selon la phase active.
        Phase NS : écoulement des files Nord et Sud
        Phase EW : écoulement des files Est et Ouest
        Phase orange : aucun écoulement
        """
        if self.phase == PHASE_NS:
            flow_dirs = [NORTH, SOUTH]
        elif self.phase == PHASE_EW:
            flow_dirs = [EAST, WEST]
        else:
            flow_dirs = []  # orange : personne ne passe

        for d in flow_dirs:
            self.queues[d] = max(0, self.queues[d] - FLOW_RATE)

    def _compute_reward(self):
        """
        Récompense = opposé du nombre total de véhicules en attente.
        Formule : R = -sum(q_i) — pénalise chaque véhicule immobilisé.
        On ajoute un malus supplémentaire si une file est saturée (MAX_QUEUE).
        """
        total_waiting = np.sum(self.queues)
        saturation_penalty = np.sum(self.queues == MAX_QUEUE) * 2
        return -(total_waiting + saturation_penalty)

    def step(self, action):
        """
        Exécute un pas de temps.
        action : 0 = Maintenir, 1 = Changer de phase

        Retourne : (next_state, reward, done)
        done est toujours False (environnement continu, épisodes artificiels)
        """
        self.step_count += 1

        # --- Gestion du changement de phase ---
        if self.phase == PHASE_ORANGE:
            # En phase orange : on attend la fin du timer
            self.orange_timer -= 1
            if self.orange_timer <= 0:
                # Commutation effective
                self.phase = PHASE_EW if self._prev_phase == PHASE_NS else PHASE_NS
        else:
            if action == 1:  # Changer
                self._prev_phase = self.phase
                self.phase = PHASE_ORANGE
                self.orange_timer = self.orange_duration
            # action == 0 : on maintient

        # --- Dynamique ---
        self._arrivals()
        self._flow()

        # --- Récompense ---
        reward = self._compute_reward()
        self.total_wait += np.sum(self.queues)

        return self._get_state(), reward, False

    def get_total_waiting(self):
        return int(np.sum(self.queues))

    def get_phase_name(self):
        return PHASE_NAMES[self.phase]


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    N_STEPS = 20
    np.random.seed(42)

    env = IntersectionEnv(lambdas=(0.4, 0.4, 0.4, 0.4), orange_duration=1)
    state = env.reset()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Simulation du carrefour — démonstration (20 pas)        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"{'Pas':>4}  {'Phase':<8}  {'N':>3} {'S':>3} {'E':>3} {'O':>3}  "
          f"{'Attente':>7}  {'Récompense':>10}  {'Action':<12}")
    print("─" * 65)

    total_reward = 0
    for step in range(1, N_STEPS + 1):
        # Action aléatoire pour la démo (0 = maintenir, 1 = changer)
        action = random.choice([0, 1])
        action_label = "Changer" if action == 1 else "Maintenir"

        next_state, reward, _ = env.step(action)
        q = env.queues
        total_reward += reward

        print(f"{step:>4}  {env.get_phase_name():<8}  "
              f"{q[0]:>3} {q[1]:>3} {q[2]:>3} {q[3]:>3}  "
              f"{env.get_total_waiting():>7}  {reward:>10.1f}  {action_label}")
        state = next_state

    print("─" * 65)
    print(f"  Récompense totale : {total_reward:.1f}")
    print(f"  Véhicules en attente (fin) : {env.get_total_waiting()}")
    print("\nPour l'entraînement complet, lancez : python main.py")
