"""
agent.py — Agent Q-Learning pour la gestion des feux de signalisation
Implémentation from scratch — Projet IAD & SMA 2025-2026
"""

import numpy as np
from simulation import MAX_QUEUE, PHASE_NS, PHASE_EW, PHASE_ORANGE


class QLearningAgent:
    """
    Agent Q-Learning tabulaire pour le contrôle d'un carrefour.

    Table Q : dict {state -> array[n_actions]}
    Politique : ε-greedy avec décroissance de ε.

    Paramètres :
        alpha (float) : taux d'apprentissage ∈ (0,1]
        gamma (float) : facteur d'actualisation ∈ [0,1]
        epsilon (float) : exploration initiale
        epsilon_min (float) : exploration minimale
        epsilon_decay (float) : facteur de décroissance de ε par épisode
        n_actions (int) : nombre d'actions (2 : maintenir / changer)
    """

    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995, n_actions=2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions

        # Table Q : initialisée à 0 pour chaque état rencontré (lazy init)
        self.Q = {}

    def _init_state(self, state):
        """Initialise les valeurs Q d'un état à zéro s'il est nouveau."""
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions)

    def select_action(self, state):
        """
        Politique ε-greedy.
        Avec probabilité ε : action aléatoire (exploration)
        Sinon : action greedy (exploitation)
        """
        self._init_state(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state):
        """
        Mise à jour de la table Q selon la règle Q-Learning :
        Q(s,a) ← Q(s,a) + α · [R + γ · max_a' Q(s',a') - Q(s,a)]
        """
        self._init_state(state)
        self._init_state(next_state)

        td_target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        """Décroissance de ε après chaque épisode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self, state):
        """Retourne l'action greedy pour un état donné."""
        self._init_state(state)
        return int(np.argmax(self.Q[state]))

    def get_q_table_size(self):
        return len(self.Q)


class BaselineAgent:
    """
    Politique de référence : feu à durée fixe (alternance périodique).
    Change de phase tous les `period` pas de temps, quelle que soit la situation.
    """

    def __init__(self, period=5):
        self.period = period
        self.counter = 0

    def select_action(self, state):
        self.counter += 1
        if self.counter >= self.period:
            self.counter = 0
            return 1  # Changer
        return 0  # Maintenir

    def reset(self):
        self.counter = 0
