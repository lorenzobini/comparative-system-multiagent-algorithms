# NOTE: Execute the replicator dynamic on the grand table also visualize it as a graph.
# You may change everything in this file.

from typing import List
import matplotlib.pyplot as plt
import numpy as np
import math

from GrandTable import GrandTable


# Proportions you will have to use:
class Proportions:
    uniform_with_own_strat = [1/9] * 9
    uniform_without_own_strat = [1/8] * 8
    non_uniform_with_own_strat = [0.12, 0.08, 0.06, 0.15, 0.05, 0.21, 0.06, 0.09, 0.18]
    non_uniform_without_own_strat = [0.22, 0.19, 0.04, 0.06, 0.13, 0.10, 0.05, 0.21]


class ReplicatorDynamic:

    history: List[List[float]]
    rounds: List[int]
    grand_table: GrandTable

    def __init__(self, grand_table: GrandTable):
        self.history = [[] for i in grand_table.row_strategies]
        self.grand_table = grand_table
        self.rounds = []

    def to_graph(self):
        """Visualize the evolution of proportions."""
        for strategy in self.history:
            plt.plot(self.rounds, strategy, linewidth=2, color=np.random.rand(3,))

        plt.legend()
        plt.show()

    def check_stability(self, old_proportions, new_proportions) -> bool:
        delta = 0.001
        diff = new_proportions - old_proportions
        diff = abs(diff)

        if sum(diff) <= delta:
            return True
        else:
            return False

    def update_history(self, proportions: List[float], round: int) -> None:
        for i in range(0, len(proportions)):
            self.history[i].append(proportions[i])
        self.rounds.append(round)

    def run(self, proportions: List[float]) -> None:

        score_table = np.array(self.grand_table.grand_table)
        prop_vector = np.array(proportions)
        round = 0

        while(True):
            self.update_history(prop_vector, round)
            expected_score = np.multiply(np.dot(score_table, prop_vector), prop_vector)
            new_prop_vector = expected_score / np.sum(expected_score)

            if self.check_stability(prop_vector, new_prop_vector):
                break

            round += 1
            prop_vector = new_prop_vector

        self.update_history(new_prop_vector, round + 1)
        self.to_graph()




