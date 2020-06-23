# NOTE: Execute the replicator dynamic on the grand table also visualize it as a graph.
# You may change everything in this file.
import os
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
    stability_count: int

    def __init__(self, grand_table: GrandTable):
        self.history = [[] for i in grand_table.row_strategies]
        self.grand_table = grand_table
        self.rounds = []
        self.stability_count = 0

    def to_graph(self, name: str): # TODO: fix overlapping
        """Visualize the evolution of proportions."""
        colors = ["#cc0000", "#ff8000", "#00994c", "#00cc00", "#cccc00", "#00cccc", "#0080ff",
                  "#9999ff", "#cc00cc"]
        path = os.path.dirname(__file__)
        plt.figure()
        for i, strategy in enumerate(self.history):
            plt.plot(self.rounds, strategy, linewidth=2, color=colors[i],
                     label=self.grand_table.row_strategies[i].name)
        plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15), prop={'size': 6})
        plt.savefig(path + "\\img\\" + name + ".png")
        #plt.show()




    def check_stability(self, old_proportions, new_proportions) -> bool:
        """
        Check the stability of the proportions given a threshold for maximum difference
        Parameters:
            *old_proportions*: Set of proportions from the previous round
            *new_proportions*: Set of newly computed proportions from the current round
        Returns:
             True if the absolute difference between the sets is lower than the threshold
             False otherwise
        """
        delta = 0.00001
        diff = new_proportions - old_proportions
        diff = abs(diff)

        if sum(diff) <= delta:
            self.stability_count += 1
            if self.stability_count == 50:
                return True
            else:
                return False
        else:
            self.stability_count = 0
            return False

    def update_history(self, proportions: List[float], round: int) -> None:
        """
        Updates the history of proportions
        Parameters:
            *proportions*: Proportions associated to each strategy
            *round*: The current round
        """
        for i in range(0, len(proportions)):
            self.history[i].append(proportions[i])
        self.rounds.append(round)

    def run(self, proportions: List[float], name="") -> None:
        """
        Runs the replicator dynamic until stability between the strategies is reached
        Parameters:
            *proportions*: Proportions associated to each strategy
        """
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
        self.to_graph(name)




