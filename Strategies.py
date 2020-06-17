# Note: You may not change methods in the Strategy class, nor their input parameters.
# Information about the entire game is given in the *initialize* method, as it gets the entire MatrixSuite.
# During play the payoff matrix doesn't change so if your strategy needs that information,
#  you can save it to Class attributes of your strategy. (like the *actions* attribute of Aselect)

import abc
import random
import numpy as np
from typing import List

import MatrixSuite
from MatrixSuite import Action, Payoff


class Strategy(metaclass=abc.ABCMeta):
    """Abstract representation of a what a Strategy should minimally implement.

    Class attributes:
        name: A string representing the name of the strategy.
    """
    name: str

    def __repr__(self) -> str:
        """The string representation of a strategy is just it's name.
        So it you call **print()** it will output the name.
        """
        return self.name

    @abc.abstractmethod
    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        """Initialize/reset the strategy with a new game.
        :param matrix_suite: The current MatrixSuite,
        so the strategy can extract the information it needs from the payoff matrix.
        :param player: A string of either 'row' or 'col',
        representing which player the strategy is currently playing as.
        """
        pass

    @abc.abstractmethod
    def get_action(self, round_: int) -> Action:
        """Calculate the action for this round.
        :param round_: The current round number.
        """
        pass

    @abc.abstractmethod
    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """Update the strategy with the result of this round.
        :param round_: The current round number.
        :param action: The action this strategy played this round.
        :param payoff: The payoff this strategy received this round.
        :param opp_action: The action the opposing strategy played this round.
        """
        pass


# As an example we have implemented the first strategy for you.


class Aselect(Strategy):
    """Implements the Aselect (random play) algorithm."""
    actions: List[Action]

    def __init__(self):
        """The __init__ method will be called when you create an object.
        This should set the name of the strategy and handle any additional parameters.

        Example:
            To create an object of the Aselect class you use, "Aselect()"

            If you want to pass it parameters on creation, e.g. "Aselect(0.1)",

            you need to modify "__init__(self)" to "__init__(self, x: float)",
            and you probably also want to save "x", or whatever you name it,
            to a class attribute, so you can use it in other methods.
            """
        self.name = "Aselect"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        """Just save the actions as that's the only thing we need."""
        self.actions = matrix_suite.get_actions(player)

    def get_action(self, round_: int) -> Action:
        """Pick the next action randomly from the possible actions."""
        return random.choice(self.actions)

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """Aselect has no update mechanic."""
        pass


# Add the other strategies below

    class EGreedy(Strategy):
        """Implements the Aselect (random play) algorithm."""
        actions: List[Action]
        payoffs: List[Payoff]

        def __init__(self):

            self.name = "EGreedy"
            self.epsilon = 0.1
            self.prob = 1 - self.epsilon
            self.actionlist = []
            self.payofflist = []

        def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
            """Just save the actions as that's the only thing we need."""
            self.actions = matrix_suite.get_actions(player)

        def get_action(self, round_: int) -> Action:
            print(round_)
            if (round_ == 1):
                return random.choice(self.actions)

            if (round_ == 2):
                highestpayoff = self.payofflist[0]

            np_player_payoffs = np.asarray(self.payofflist, dtype=np.float32)
            np_actions = np.asarray(self.actionlist, dtype=np.int)

            print(len(self.actionlist), "actionlist elements")
            print(len(self.payofflist), "payofflist elements")
            print(highestpayoff, "highestpayoff")

            temp1 = []

            for each in np.unique(np_actions):
                indices = np.argwhere(np_actions == each)
                for each_ in indices:
                    temp = np_player_payoffs[each_]
                    temp1.append(temp)
                    temp2 = sum(temp1) / len(indices)
                    if temp2 >= highestpayoff:
                        highestpayoff = temp2
                        beststrategy = each

            probarray = np.zeros(len(set(self.actionlist)) + 1)

            print(beststrategy, "beststrategy")
            range_ = len(set(self.actionlist)) + 1
            probarray[beststrategy] = self.prob
            choiceindex = choice([i for i in range(0, range_) if i not in [beststrategy, -1]])

            print(choiceindex, "epsilonindex")
            probarray[choiceindex] = self.epsilon

            print(probarray, "probarray")
            print(self.actions, "actions")

            return np.random.choice(
                [self.actions],
                p=probarray
            )

        def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
            print(action, "update action")
            self.actionlist.append(action)
            self.payofflist.append(payoff)




    class UCB(Strategy):
        """Implements the Aselect (random play) algorithm."""
        actions: List[Action]

        def __init__(self):
            self.name = "UCB"

        def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
            """Just save the actions as that's the only thing we need."""
            self.actions = matrix_suite.get_actions(player)

        def get_action(self, round_: int) -> Action:
            """Pick the next action randomly from the possible actions."""
            return random.choice(self.actions)

        def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
            """Aselect has no update mechanic."""
            pass

