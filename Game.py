# NOTE: This is a suggestion of how you could begin implementing a 2 player game,
# feel free to come up with your own way. You may change almost everything of this class,
# it just has to play 2 strategies against each other on a matrix game.

from typing import List

import MatrixSuite
from Strategies import Strategy
from MatrixSuite import Payoff, Action


class Game:
    """Play a 2 player game on the payoff matrix of the given MatrixSuite,
    keep track of all the actions and payoffs during the game.

    Class attributes:
        *matrix_suite*: The MatrixSuite that the game is played on, should NOT be updated during the game,
        as the payoff matrix should stay the same.

        *round_*: Keeps track of the current round number.

        *row_player*: Instance of a Strategy subclass for the row player. (i.e. the Aselect class)

        *col_player*: Instance of a Strategy subclass for the column player. (i.e. the Aselect class)

        *row_player_actions*: History of the actions played by the row player.

        *col_player_actions*: History of the actions played by the column player.

        *row_player_payoffs*: History of the payoffs received by the row player.

        *col_player_payoffs*: History of the payoffs received by the column player.
    """
    matrix_suite: MatrixSuite
    round_: int
    row_player: Strategy
    col_player: Strategy
    row_player_payoffs: List[Payoff]

    def __init__(self, game_suite: MatrixSuite, row_player: Strategy, col_player: Strategy) -> None:
        """Set all the variables and call the initialize method."""
        self.row_player = row_player
        self.col_player = col_player
        self.initialize(game_suite)

    def initialize(self, game_suite: MatrixSuite) -> None:
        """(Re-) initialize the game with an updated matrix suite."""
        self.matrix_suite = game_suite
        self.round_ = 0
        self.row_player_payoffs = []

        # Call initialize on the strategies at the start of the game.
        self.row_player.initialize(self.matrix_suite, "row")
        self.col_player.initialize(self.matrix_suite, "col")

    def play_round(self, round) -> None:
        """
        Plays the round by generating both players' action and determining the relative reward
        """
        row_action = self.row_player.get_action(round)  # Get the next action
        col_action = self.col_player.get_action(round)

        row_payoff, col_payoff = self.matrix_suite.get_payoffs(row_action, col_action)

        self.row_player_payoffs.append(row_payoff)

        self.row_player.update(round, row_action, row_payoff, col_action)
        self.col_player.update(round, col_action, col_payoff, row_action)

    def play(self, rounds: int) -> float:
        """
        Plays the game for the specified number of rounds
        Returns:
            The average payoff for the row player
        """
        for round in range(0, rounds-1):
            self.play_round(round)

        avg = sum(self.row_player_payoffs) / len(self.row_player_payoffs)

        return avg

    # Add methods that implement the logic of playing a game, so play one round and play x rounds.
