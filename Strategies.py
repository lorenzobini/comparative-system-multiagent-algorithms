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
    """
    Implements the Egreedy (exploration-exploitation) algorithm.
    Class attributes:
        *actions*: List of playable actions for the current player
        *prob*: Probability of choosing exploitation
        *epsilon*: Probability of choosing exploration
        *action_history*: History of all played actions in the current session
        *payoff_history*: History of all obtained payoffs associated to the played actions
    """
    actions: List[Action]
    prob: float
    epsilon: float
    action_history: List[Action]
    payoff_history: List[Payoff]

    def __init__(self):
        self.name = "EGreedy"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        """
        Initialisation of actions and parameters.
        Parameters:
           *matrix_suite*: The MatrixSuite the game is played upon
        """
        self.actions = matrix_suite.get_actions(player)
        self.epsilon = 0.1
        self.prob = 1 - self.epsilon
        self.action_history = []
        self.payoff_history = []

    def get_action(self, round_: int) -> Action:
        """
        Determines the next action to play. Exploits the action with the highest average payoff
        with probability *prob*, explore a random action with probability *epsilon*
        """
        if round_ == 0:
            return random.choice(self.actions)

        avg_scores = np.zeros(len(self.actions))

        # calculate average for each
        for each in set(self.action_history):
            indices = [i for i, j in enumerate(self.action_history) if j == each]
            avg = sum([self.payoff_history[i] for i in indices]) / len(indices)
            avg_scores[each] = avg

        # retrieve highest average
        best_strategy = np.argmax(avg_scores)

        # chose explore or exploit based on probabilities
        strategy = np.random.choice(['Explore', 'Exploit'], p=[self.epsilon, self.prob])

        if strategy == "Explore":
            range_ = len(set(self.actions))
            choiceindex = choice([i for i in range(0, range_) if i not in [best_strategy, -1]])
            return self.actions[choiceindex]
        else:
            return self.actions[best_strategy]

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """
        Updates the last played action and the relative payoff obtained
        """
        self.action_history.append(action)
        self.payoff_history.append(payoff)



class UCB(Strategy):
    """
    Implements the Upper Confidence Bound algorithm.
    Class attributes:
        *actions*: List of playable actions for the current player
        *action_history*: History of all played actions in the current session
        *payoff_history*: History of all obtained payoffs associated to the played actions
    """
    actions: List[Action]
    action_history: List[Action]
    payoff_history: List[Payoff]

    def __init__(self):
        self.name = "UCB"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        """
        Initialisation of actions and histories.
        Parameters:
           *matrix_suite*: The MatrixSuite the game is played upon
        """
        self.actions = matrix_suite.get_actions(player)
        self.action_history = []
        self.payoff_history = []

    def get_action(self, round_: int) -> Action:
        """
        Determines the next action to play. At first every action at least once.
        It computes the average reward obtained by playing each action. At every subsequent round, it plays the action
        with the highest average reward.
        """
        if(round_ == 0):
            return random.choice(self.actions)

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

        max_score = np.argmax(avgscores)
        return self.actions[max_score]

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """
        Updates the last played action and the relative payoff obtained
        """
        self.action_history.append(action)
        self.payoff_history.append(payoff)


class Softmax(Strategy):
    """
    Implements the Softmax Q-Learning algorithm.
    Class attributes:
        *actions*: List of playable actions for the current player
        *learning_rate*: The learning rate for the Q-Learning moving average
        *temperature*: The temperature for the softmax distribution computation
        *action_history*: History of all played actions in the current session
        *qmatrix*: Moving average of payoffs for each instance of *actions*
        *probmatrix*: Softmax probability distribution for each instance of *actions*
    """
    actions: List[Action]
    learning_rate: float
    temperature: float
    action_history: List[Action]
    qmatrix: np.ndarray
    probmatrix: np.ndarray

    def __init__(self):
        self.name = "Softmax"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        """
        Initialisation of actions and histories.
        Parameters:
           *matrix_suite*: The MatrixSuite the game is played upon
        """
        self.actions = matrix_suite.get_actions(player)
        self.qmatrix = np.ones(len(self.actions))
        self.probmatrix = np.zeros(len(self.actions))
        self.learning_rate = 0.1
        self.temperature = 1.0
        self.action_history = []

    def get_action(self, round_: int) -> Action:
        """
        Determines the next action to play. The first action is chosen at random.
        Every subsequent action is chosen according to the Softmax probability distribution *probmatrix*
        of the Q-Learning moving averages *qmatrix* for each instance of *actions*
        """
        if round_ == 0:
            # first round pick a random action
            action = random.choice(self.actions)
        else:
            # get action based on probabilities
            action = np.where(np.random.multinomial(1, self.probmatrix))[0][0]

        return action

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """
        Updates the last played action, the moving average associated with the last played action
        following the Q-Learning approach, the Softmax probability distribution given the updated
        moving averages.
        """
        self.action_history.append(action)
        self.qmatrix[action] = (1 - self.learning_rate) * self.qmatrix[action] + self.learning_rate * payoff
        for i, each in enumerate(self.qmatrix):
            self.probmatrix[i] = np.exp(each / self.temperature) / sum(np.exp(self.qmatrix / self.temperature))



class Satisficing(Strategy):
    """
    Implements the Satisficing (gamma=0.1) algorithm.
    Class attributes:
        *actions*: List of playable actions for the current player
        *alpha*: Aspiration level
        *gamma*: Learning rate
        *action_history*: History of all played actions in the current session
        *payoff_history*: History of all obtained payoffs associated to the played actions
    """
    actions: List[Action]
    alpha: float
    gamma: float
    action_history: List[Action]
    payoff_history: List[Payoff]

    def __init__(self):
        self.name = "Satisficing"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        """
        Initialisation of actions, parameters and histories.
        Parameters:
           *matrix_suite*: The MatrixSuite the game is played upon
        """
        self.actions = matrix_suite.get_actions(player)
        self.alpha = 1 + matrix_suite.get_max_payoff(0 if player == "row" else 1)
        self.gamma = 0.1
        self.action_history = []
        self.payoff_history = []

    def get_action(self, round_: int) -> Action:
        """
        Determines the next action to play. Exploits the last played action if it satisfies the aspiration level,
        otherwise explores a random action.
        """
        action = random.choice(self.actions)

        if len(self.payoff_history) == 0 or self.payoff_history[-1] < self.alpha:
            action = random.choice(self.actions)
        elif self.payoff_history[-1] >= self.alpha:
            action = self.actions[-1]

        return action

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """
        Updates the last played action, the respective obtained payoff and the aspiration level.
        """
        self.alpha = self.alpha * self.gamma + (1 - self.gamma) * payoff
        self.action_history.append(action)
        self.payoff_history.append(payoff)


class Bully(Strategy):
    """
    Implements the Bully strategy algorithm.
    Class attributes:
        *actions*: List of playable actions for the current player
        *bestAction*: The action with the highest potential reward according the the MatrixSuite
        *player*: 0 if row player, 1 if column player
        *opponent*: 0 if row player: 1 if column player
    """
    actions: List[Action]
    bestAction: Action
    player: int
    opponent: int

    def __init__(self):
        self.name = "Bully"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        """
        Initialisation of actions and parameters.
        Parameters:
           *matrix_suite*: The MatrixSuite the game is played upon
        """
        self.actions = matrix_suite.get_actions(player)
        self.player = 0 if player == "row" else 1
        self.opponent = 1 - self.player
        self.bestAction = self.find_best_action(matrix_suite)

    def get_action(self, round_: int) -> Action:
        """
        Determines the next action to play. Stubbornly returns the action with the highest potential payoff.
        """
        return self.bestAction

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """Bully does not need any update"""
        pass

    def find_best_action(self, matrix_suite: MatrixSuite) -> Action:
        """
        Determine the best action given the matrix suite. The best action is determined by  summing through
        the potential rewards obtainable by playing each possible action and choosing the one with the highest sum.
        It considers left-hand rewards if player is the row player, the right-hand rewards if player is column player.
        """
        player_actions = matrix_suite.get_actions("row" if self.player==0 else "col")
        opponent_actions = matrix_suite.get_actions("row" if self.opponent==0 else "col")
        max_payoff = 0
        best_action = random.choice(self.actions)

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
    """
    Implements the Fictitious Play algorithm
    Class attributes:
        *actions*: List of playable actions for the current player
        *player*: 0 if row player, 1 if column player
        *opponent*: 0 if row player: 1 if column player
        *opponent_actions*: The opponent's playable actions
        *opp_action_frequency*: List of frequencies for each action
        *most_frequent_action*: Action with the highest frequency in *opp_action_frequency*
        *matrix_suite*: The game's matrix suite
    """
    actions: List[Action]
    player: int
    opponent: int
    opponent_actions: List[Action]
    opp_action_frequency: List[int]
    most_frequent_actions: List[int]
    matrix_suite: MatrixSuite

    def __init__(self):
        self.name = "Fictitious"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        """
        Initialisation of actions and parameters.
        Parameters:
           *matrix_suite*: The MatrixSuite the game is played upon
        """
        self.player = 0 if player == "row" else 1
        self.opponent = 1 - self.player
        self.actions = matrix_suite.get_actions(player)
        self.matrix_suite = matrix_suite
        self.opponent_actions = matrix_suite.get_actions("row" if player == "col" else "col")
        self.opp_action_frequency = [0 for action in self.opponent_actions]
        self.most_frequent_actions = []

    def get_action(self, round_: int) -> Action:
        """
        Determines the next action to play. Chooses a random action for the first round. At each subsequent round,
        it chooses the best response to the opponent's most chosen action
        """

        if len(self.most_frequent_actions) == 0:
            action = random.choice(self.actions)
        elif len(self.most_frequent_actions) == 1:
            action = self.find_best_response(self.most_frequent_actions[0])
        else:
            action = self.find_best_response(random.choice(self.most_frequent_actions))

        return action

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """
        Updates the frequency table
        """
        self.opp_action_frequency[opp_action] += 1
        self.most_frequent_actions = self.find_most_frequent_actions()

    def find_most_frequent_actions(self) -> List[Action]:
        """
        Determines the action(s) that  is(are) most frequently chosen by the opponent
        """
        max_payoff = 0
        most_frequent_actions = []
        for action in self.opponent_actions:
            if self.opp_action_frequency[action] > max_payoff:
                most_frequent_actions = [action]
            elif self.opp_action_frequency[action] == max_payoff:
                most_frequent_actions.append(action)

        return most_frequent_actions

    def find_best_response(self, opp_action: Action) -> Action:
        """
        Given the opponent's most frequently played action, returns the best response to it.
        The best response is defined as the playable action that ensures the highest payoff given the
        opponent's choice.
        It considers left-hand rewards if player is the row player, the right-hand rewards if player is column player.
        """
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
    """
    Implements the Proportional Regret Matching algorithm for no-regret
    Class attributes:
        *actions*: List of playable actions for the current player
        *player*: 0 if row player, 1 if column player
        *opponent*: 0 if row player: 1 if column player
        *actual_payoff*: Last payoff obtained
        *potential_payoof: List of potential payoffs obtainable by playing each action on the last round
        *regret_matching*: List of normalised regrets for each action
        *matrix_suite*: The game's matrix suite
    """
    actions: List[Action]
    player: int
    opponent: int
    actual_payoff: Payoff
    potential_payoffs: List[Payoff]
    regret_matching: List[float]
    matrix_suite: MatrixSuite

    def __init__(self):
        self.name = "PRM"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        """
        Initialisation of actions and parameters.
        Parameters:
           *matrix_suite*: The MatrixSuite the game is played upon
        """
        self.player = 0 if player == "row" else 1
        self.opponent = 1 - self.player
        self.actions = matrix_suite.get_actions(player)
        self.matrix_suite = matrix_suite
        self.actual_payoff = 0
        self.potential_payoffs = [0 for action in self.actions]
        self.regret_matching = [0 for action in self.actions]

    def get_action(self, round_: int) -> Action:
        """
        Determines the next action to play. Returns new action according to regret matching probability distribution.
        If the regret matching probability distribution is not available, it returns a random action.
        """
        if all(prob == 0 for prob in self.regret_matching):
            action = random.choice(self.actions)
        else:
            action = np.random.choice(self.actions, p=self.regret_matching)

        return action

    def update(self, round_: int, action: Action, payoff: Payoff, opp_action: Action) -> None:
        """
        Updates actual payoff and potential payoffs for the last round. Computes the regret and the
        regret matching probability distribution for the current round.
        """
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
    """
    Implements the Egreedy algorithm with UCB as exploration strategy and FictitiousPlay as exploitation strategy.
    Class attributes:
        *actions*: List of playable actions for the current player
        *prob*: Probability of choosing exploitation
        *epsilon*: Probability of choosing exploration
        *player*: 0 if row player, 1 if column player
        *opponent*: 0 if row player: 1 if column player
        *opponent_actions*: The opponent's playable actions
        *opp_action_frequency*: List of frequencies for each action
        *most_frequent_action*: Action with the highest frequency in *opp_action_frequency*
        *matrix_suite*: The game's matrix suite
        *action_history*: History of all played actions in the current session
        *payoff_history*: History of all obtained payoffs associated to the played actions
    """
    actions: List[Action]
    prob: float
    epsilon: float
    # FictitiousPlay
    player: int
    opponent: int
    opponent_actions: List[Action]
    opp_action_frequency: List[int]
    most_frequent_actions: List[int]
    matrix_suite: MatrixSuite
    # UCB
    action_history: List[Action]
    payoff_history: List[Payoff]

    def __init__(self):
        self.name = "EFictitious"

    def initialize(self, matrix_suite: MatrixSuite, player: str) -> None:
        """
        Initialisation of actions and parameters.
        Parameters:
           *matrix_suite*: The MatrixSuite the game is played upon
        """
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
        # UCB
        self.action_history = []
        self.payoff_history = []

    def get_action(self, round_: int) -> Action:
        """
        Determines the next action to play. Exploits by playing Fictitious Play with probability *prob*,
        explores by playing Upper Confidence Bound with probability *epsilon*
        """
        strategy = np.random.choice(["exploit", "explore"], p=[self.prob, self.epsilon])
        if strategy == "exploit":
            action = self.play_FictitiousPlay(round_)
        else:
            action = self.play_UCB(round_)

        return action

    def play_UCB(self, round_: int):
        """
        Determines the next action to play. At first every action at least once.
        It computes the average reward obtained by playing each action. At every subsequent round, it plays the action
        with the highest average reward.
        """
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

        max_score = np.argmax(avgscores)
        return self.actions[max_score]

    def play_FictitiousPlay(self, round_: int) -> Action:
        """
        Determines the next action to play. Chooses a random action for the first round. At each subsequent round,
        it chooses the best response to the opponent's most chosen action
        """
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
        """
        Given the opponent's most frequently played action, returns the best response to it.
        The best response is defined as the playable action that ensures the highest payoff given the
        opponent's choice.
        It considers left-hand rewards if player is the row player, the right-hand rewards if player is column player.
        """
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
