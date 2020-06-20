# Note: You may not change methods in the Strategy class, nor their input parameters.
# Information about the entire game is given in the *initialize* method, as it gets the entire MatrixSuite.
# During play the payoff matrix doesn't change so if your strategy needs that information,
#  you can save it to Class attributes of your strategy. (like the *actions* attribute of Aselect)

import abc
import math
import random
import numpy as np
from numpy.random import choice
from typing import List
from operator import sub

from MatrixSuite import Action, Payoff
from MatrixSuite import MatrixSuite


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
    """Implements the Egreedy (exploration-exploitation) algorithm."""
    actions: List[Action]
    payoffs: List[Payoff]
    prob: float
    epsilon: float
    probarray: np.ndarray


    def __init__(self):
        self.name = "EGreedy"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        """Just save the actions as that's the only thing we need."""
        self.actions = matrix_suite.get_actions(player)
        self.epsilon = 0.1
        self.prob = 1 - self.epsilon
        self.actions = []
        self.payoffs = []
        self.probarray = np.zeros(len(self.actions))

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
                temp2 = sum(temp1)/len(indices)
                if temp2 >= highestpayoff:
                    highestpayoff = temp2
                    beststrategy = each


        print(beststrategy, "beststrategy")
        range_ = len(set(self.actionlist)) + 1
        self.probarray[beststrategy] = self.prob
        choiceindex = choice([i for i in range(0, range_) if i not in [beststrategy, -1]])

        print(choiceindex, "epsilonindex")
        self.probarray[choiceindex] = self.epsilon

        ind = np.where(np.random.multinomial(1, self.probarray))[0][0]


        return self.actions[ind]

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        print (action, "update action")
        self.actionlist.append(action)
        self.payofflist.append(payoff)

class UCB(Strategy):
    """Implements the Aselect (random play) algorithm."""
    actions: List[Action]
    action_history: List[Action]
    payoff_history: List[Payoff]

    def __init__(self):
        self.name = "UCB"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        """Just save the actions as that's the only thing we need."""
        self.actions = matrix_suite.get_actions(player)
        self.action_history = []
        self.payoff_history = []

    def get_action(self, round_: int) -> Action:
        # TODO: description
        for action in self.actions:
            if action not in self.action_history:
                return self.actions[action]

        avgscores = np.zeros(len(self.actions))

        for each in set(self.action_history):
            indices = [i for i, j in enumerate(self.action_history) if j == each]
            avg = sum([self.payoff_history[i] for i in indices]) / len(indices)
            avgscores[each] = avg

        for each in set(self.actions):
            indices = len([i for i, j in enumerate(self.action_history) if j == each])
            avgscores[each] = avgscores[each] + math.sqrt((2 * math.log(round_)) / indices)

        return self.actions[np.argmax(avgscores)]

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        self.action_history.append(action)
        self.payoff_history.append(payoff)


class Softmax(Strategy):
    """Implements the Softmax algorithm."""
    actions: List[Action]
    qmatrix: np.ndarray
    learning_rate: float
    discount: float
    payoff: float
    temperature: float
    action_history: List[Action]
    probmatrix: np.ndarray

    def __init__(self):
        self.name = "Softmax"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        # TODO: description
        self.actions = matrix_suite.get_actions(player)
        self.qmatrix = np.ones(len(self.actions))
        self.probmatrix = np.zeros(len(self.actions))
        self.learning_rate = 0.1
        self.temperature = 1.0
        self.payoff = 0.0
        self.action_history = []

    def get_action(self, round_: int) -> Action:
        # TODO: description
        # first round pick a random action
        if round_ == 0:
            return random.choice(self.actions)
        # check last action and its respective q-value
        lastaction = self.action_history[-1]
        curq = self.qmatrix[lastaction]
        # update q value
        self.qmatrix[lastaction] = (1 - self.learning_rate) * curq + self.learning_rate * (self.payoff)

        # normalize using softmax
        i = 0
        for each in self.qmatrix:
            self.probmatrix[i] = np.exp(each / self.temperature) / sum(np.exp(self.qmatrix / self.temperature))
            i += 1
        # get action based on probabilities
        ind = np.where(np.random.multinomial(1, self.probmatrix))[0][0]

        return self.actions[ind]

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        self.payoff = payoff
        self.action_history.append(action)


class Satisficing(Strategy):
    """Implements the Satisficing (gamma=0.1) algorithm."""
    alpha: float
    gamma: float
    actions: List[Action]
    past_actions: List[Action]
    past_payoffs: List[Payoff]

    def __init__(self):
        self.name = "Satisficing Play"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        self.actions = matrix_suite.get_actions(player)
        self.alpha = 1 + matrix_suite.get_max_payoff(0 if player == "row" else 1)
        self.gamma = 0.1
        self.past_actions = []
        self.past_payoffs = []

    def get_action(self, round_: int) -> Action:
        """Exploit if the last action satisfies the aspiration level, otherwise explore."""
        action: Action

        if len(self.past_payoffs) == 0 or self.past_payoffs[-1] < self.alpha:
            action = random.randint(0, len(self.actions)-1)
        elif self.past_payoffs[-1] >= self.alpha:
            action = self.actions[-1]

        return action

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        # TODO: description
        self.alpha = self.alpha * self.gamma + (1 - self.gamma) * payoff
        self.past_actions.append(action)
        self.past_payoffs.append(payoff)


class Bully(Strategy):
    """Implements the Bully strategy algorithm."""
    actions: List[Action]
    bestAction: Action
    player: int
    opponent: int

    def __init__(self):
        self.name = "Bully"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        self.actions = matrix_suite.get_actions(player)
        self.player = 0 if player == "row" else 1
        self.opponent = 1 - self.player
        self.bestAction = self.find_best_action(matrix_suite)

    def get_action(self, round_: int) -> Action:
        """Stubbornly returns the action with the highest potential payoff"""
        return self.bestAction

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """Bully does not need any update"""
        pass

    def find_best_action(self, matrix_suite: MatrixSuite) -> Action:
        # TODO: description
        player_actions = matrix_suite.get_actions("row" if self.player==0 else "col")
        opponent_actions = matrix_suite.get_actions("row" if self.opponent==0 else "col")
        max_payoff = 0

        for player_action in player_actions:
            total = 0
            for opponent_action in opponent_actions:
                if self.player == 0:
                    ''' Row Player: summing through the left-hand payoffs of each row'''
                    total += matrix_suite.get_payoffs(player_action, opponent_action)[self.player]
                else:
                    '''Column Player: summing through the right-hand payoffs of each column'''
                    total += matrix_suite.get_payoffs(opponent_action, player_action)[self.player]
            if total > max_payoff:
                max_payoff = total
                best_action = player_action

        return best_action


class FictitiousPlay(Strategy):
    """Implements the Fictitious Play algorithm"""
    actions: List[Action]
    player: int
    opponent: int
    opponent_actions: List[Action]
    opp_action_frequency: List[int]
    most_frequent_actions: List[int]
    matrix_suite: MatrixSuite

    def __init__(self):
        self.name = "Fictitious Play"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        # TODO: description
        self.player = 0 if player == "row" else 1
        self.opponent = 1 - self.player
        self.actions = matrix_suite.get_actions(player)
        self.matrix_suite = matrix_suite
        self.opponent_actions = matrix_suite.get_actions("row" if player == "col" else "col")
        self.opp_action_frequency = [0 for action in self.opponent_actions]
        self.most_frequent_actions = []

    def get_action(self, round_: int) -> Action:
        """Returns the best response at the opponent's most chosen action"""

        if len(self.most_frequent_actions) == 0:
            action = random.choice(self.actions)
        elif len(self.most_frequent_actions) == 1:
            action = self.find_best_response(self.most_frequent_actions[0])
        else:
            action = self.find_best_response(random.choice(self.most_frequent_actions))

        return action

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """Updating frequency table"""
        self.opp_action_frequency[opp_action] += 1
        self.most_frequent_actions = self.find_most_frequent_actions()

    def find_most_frequent_actions(self) -> List[Action]:
        """Determines the action(s) that  is(are) most frequently chosen by the opponent"""
        max_payoff = 0
        most_frequent_actions = []
        for action in range(0, len(self.opponent_actions)-1):
            if self.opp_action_frequency[action] > max_payoff:
                most_frequent_actions = [action]
            elif self.opp_action_frequency[action] == max_payoff:
                most_frequent_actions.append(action)

        return most_frequent_actions

    def find_best_response(self, opp_action: Action) -> Action:
        # TODO: description
        max_payoff = 0
        best_response = random.choice(self.actions)

        for action in self.actions:
            if self.player == 0:
                ''' Row Player: given a column, determine the action with the highest payoff'''
                payoff = self.matrix_suite.get_payoffs(action, opp_action)[self.player]
            else:
                '''Column Player: given a row, determine the action with the highest payoff'''
                payoff = self.matrix_suite.get_payoffs(opp_action, action)[self.player]

            if payoff > max_payoff:
                max_payoff = payoff
                best_response = action

        return best_response


class ProportionalRegretMatching(Strategy):
    """Implements the Proportional Regret Matching algorithm for no-regret"""
    actions: List[Action]
    player: int
    opponent: int
    actual_payoff: Payoff
    potential_payoffs: List[Payoff]
    regret_matching: List[float]
    matrix_suite: MatrixSuite

    def __init__(self):
        self.name = "Proportional Regret Matching"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        # TODO: description
        self.player = 0 if player == "row" else 1
        self.opponent = 1 - self.player
        self.actions = matrix_suite.get_actions(player)
        self.matrix_suite = matrix_suite
        self.actual_payoff = 0
        self.potential_payoffs = [0 for action in self.actions]
        self.regret_matching = [0 for action in self.actions]

    def get_action(self, round_: int) -> Action:
        """Returns new action according to regret matching probability distribution"""
        if all(prob == 0 for prob in self.regret_matching):
            action = random.choice(self.actions)
        else:
            action = np.random.choice(self.actions, p=self.regret_matching)

        return action


    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """Updating payoffs and regret"""
        total_regret = 0

        self.actual_payoff += payoff
        for player_action in self.actions:
            if self.player == 0:
                ''' Row Player: we consider the left-hand side of the payoffs'''
                self.potential_payoffs[player_action] += self.matrix_suite.get_payoffs(player_action, opp_action)[self.player]
            else:
                ''' Row Player: we consider the right-hand side of the payoffs'''
                self.potential_payoffs[player_action] += self.matrix_suite.get_payoffs(opp_action, player_action)[self.player]

        for potential_payoff in self.potential_payoffs:
            """ Regret subsists only when potential payoff is higher than actual payoff """
            if potential_payoff > self.actual_payoff:
                total_regret += potential_payoff - self.actual_payoff

        if total_regret != 0:
            self.regret_matching = [(potential_payoff - self.actual_payoff) / total_regret
                                    for potential_payoff in self.potential_payoffs]

            # check for negative payoffs and set them to 0 probability
            for player_action in self.actions:
                if self.regret_matching[player_action] < 0:
                    self.regret_matching[player_action] = 0


'''Custom Strategy'''


class EFictitiousPlay(Strategy):
    """Egreedy with UCB played 10% of times and FictitiousPlay played 90% of times"""
    actions: List[Action]  # TODO: possibly a window
    payoffs: List[Payoff]
    prob: float
    epsilon: float
    #FictitiousPlay
    player: int
    opponent: int
    opponent_actions: List[Action]
    opp_action_frequency: List[int]
    most_frequent_actions: List[int]
    matrix_suite: MatrixSuite
    #UCB
    action_history: List[Action]
    payoff_history: List[Payoff]

    def __init__(self):
        self.name = "EFictitiousPlay"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        # TODO: description
        self.actions = matrix_suite.get_actions(player)
        self.payoffs = []
        self.epsilon = 0.2
        self.prob = 1 - self.epsilon
        # Fictitious Play parameters
        self.player = 0 if player == "row" else 1
        self.opponent = 1 - self.player
        self.matrix_suite = matrix_suite
        self.opponent_actions = matrix_suite.get_actions("row" if player == "col" else "col")
        self.opp_action_frequency = [0 for action in self.opponent_actions]
        self.most_frequent_actions = []
        #UCB
        self.action_history = []
        self.payoff_history = []

    def get_action(self, round_: int) -> Action:
        # TODO: description
        strategy = np.random.choice(["exploit", "explore"], p=[self.prob, self.epsilon])
        if strategy == "exploit":
            action = self.play_FictitiousPlay(round_)
        else:
            action = self.play_UCB(round_)

        return action

    def play_UCB(self, round_: int):
        # TODO: description
        for action in self.actions:
            if action not in self.action_history:
                return self.actions[action]

        avgscores = np.zeros(len(self.actions))

        for each in set(self.action_history):
            indices = [i for i, j in enumerate(self.action_history) if j == each]
            avg = sum([self.payoff_history[i] for i in indices]) / len(indices)
            avgscores[each] = avg

        for each in set(self.actions):
            indices = len([i for i, j in enumerate(self.action_history) if j == each])
            avgscores[each] = avgscores[each] + math.sqrt((2 * math.log(round_)) / indices)

        return self.actions[np.argmax(avgscores)]

    def play_FictitiousPlay(self, round_: int) -> Action:
        # TODO: description
        if len(self.most_frequent_actions) == 0:
            action = random.choice(self.actions)
        elif len(self.most_frequent_actions) == 1:
            action = self.find_best_response(self.most_frequent_actions[0])
        else:
            action = self.find_best_response(random.choice(self.most_frequent_actions))

        return action

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """Updating frequency table for FictitiousPlay"""
        self.opp_action_frequency[opp_action] += 1
        self.most_frequent_actions = self.find_most_frequent_actions()
        """Updating actions and payoffs for UCB"""
        self.action_history.append(action)
        self.payoff_history.append(payoff)

    def find_most_frequent_actions(self) -> List[Action]:
        """Determines the action(s) that  is(are) most frequently chosen by the opponent"""
        max_payoff = 0
        most_frequent_actions = []
        for action in range(0, len(self.opponent_actions)-1):
            if self.opp_action_frequency[action] > max_payoff:
                most_frequent_actions = [action]
            elif self.opp_action_frequency[action] == max_payoff:
                most_frequent_actions.append(action)

        return most_frequent_actions

    def find_best_response(self, opp_action: Action) -> Action:
        # TODO: description
        max_payoff = 0
        best_response = random.choice(self.actions)

        for action in self.actions:
            if self.player == 0:
                ''' Row Player: given a column, determine the action with the highest payoff'''
                payoff = self.matrix_suite.get_payoffs(action, opp_action)[self.player]
            else:
                '''Column Player: given a row, determine the action with the highest payoff'''
                payoff = self.matrix_suite.get_payoffs(opp_action, action)[self.player]

            if payoff > max_payoff:
                max_payoff = payoff
                best_response = action

        return best_response
