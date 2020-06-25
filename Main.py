# Note: From this script the program can be run.
# You may change everything about this file.

import MatrixSuite
from MatrixSuite import FixedMatrixSuite
from MatrixSuite import RandomIntMatrixSuite
from MatrixSuite import RandomFloatMatrixSuite
import Strategies
from  Strategies import Strategy
import Game
from GrandTable import GrandTable
from ReplicatorDynamic import ReplicatorDynamic, Proportions
import Nash
from typing import List


'''Setting up strategies and relative proportions'''
'''
strategies = [Strategies.Aselect(), Strategies.EGreedy(), Strategies.UCB(), Strategies.Satisficing(),
              Strategies.Softmax(), Strategies.FictitiousPlay(), Strategies.Bully(),
              Strategies.ProportionalRegretMatching()]
strategies_ext = [Strategies.Aselect(), Strategies.EGreedy(), Strategies.UCB(), Strategies.Satisficing(),
                  Strategies.Softmax(), Strategies.FictitiousPlay(), Strategies.Bully(),
                  Strategies.ProportionalRegretMatching(), Strategies.EFictitiousPlay()]

proportions = [Proportions.uniform_without_own_strat, Proportions.non_uniform_without_own_strat]
proportions_ext = [Proportions.uniform_with_own_strat, Proportions.non_uniform_with_own_strat]


def game_session(game_id: int, matrix_suite: MatrixSuite, strategies: List[Strategy],
                 proportions: List[List[float]], restarts: int, rounds=1000) -> None:
    """
    Builds the game session according to the given parameters.
    Parameters:
        *game_id*: Identifier, it gets printed on console
        *matrix_suite: The MatrixSuite that the game is played on
        *strategies*: The set of strategies that will play against each other
        *proportions*: The probability distribution associated to each strategy for replicator dynamic
        *restarts*: The number of restart per game session
        *rounds* The number of rounds per restart, default is 1000
    """

    print("\n==============================================\n")

    print("GAME ", game_id, " \n")
    grand_table = GrandTable(matrix_suite, strategies, restarts, rounds)

    grand_table.play_games()
    print(grand_table)
    grand_table.to_latex("Game_" + str(game_id))

    i = 1
    for proportion in proportions:
        replicator_dynamic = ReplicatorDynamic(grand_table)
        replicator_dynamic.run(proportion, name="Game_" + str(game_id) + "_" + str(i))
        i+=1

    print("Gambit test result:")
    Nash.nash_equilibria(strategies, grand_table)


# FIRST GAME:Standard set of strategies, Fixed Matrix Suite
game_session(1, FixedMatrixSuite(), strategies, proportions, 9)
# SECOND GAME: Extended set of strategies, Fixed Matrix Suite
game_session(2, FixedMatrixSuite(), strategies_ext, proportions_ext, 9)
# THIRD GAME: Standard set of strategies, Random Int Matrix Suite
game_session(3, RandomIntMatrixSuite(), strategies, proportions, 19)
# FOURTH GAME: Extended set of strategies, Random Int Matrix Suite
game_session(4, RandomIntMatrixSuite(), strategies_ext, proportions_ext, 19)
# FIFTH GAME: Standard set of strategies, Random Float Matrix Suite
game_session(5, RandomFloatMatrixSuite(), strategies, proportions, 19)
# SIXTH GAME: Extended set of strategies, Random Float Matrix Suite
game_session(6, RandomFloatMatrixSuite(), strategies_ext, proportions_ext, 19)



'''
############# TEST AREA
print("BEGIN  ########################################\n")


matrix_suite = RandomIntMatrixSuite()  # Create a matrix suite

row_strat = Strategies.Softmax()  # Create the strategy you want to test.
col_strat = Strategies.Softmax()
row_strat.initialize(matrix_suite, "row")  # Initialise it with the game suite and as either "row" or "col" player.
col_strat.initialize(matrix_suite, "col")

for round in range(0, 500):
    row_action = row_strat.get_action(round)  # Get the next action
    col_action = col_strat.get_action(round)
    print("Row plays action:" + row_action.__repr__())
    print("Col plays action:" + col_action.__repr__())

    row_payoff, col_payoff = matrix_suite.get_payoffs(row_action, col_action)
    print("Payoff: " + str((row_payoff,col_payoff)))

    row_strat.update(round, row_action, row_payoff, col_action)
    col_strat.update(round, col_action, col_payoff, row_action)# Update the strategy with a fake payoff and opponent action.
# Now you might want to look at the class attributes of the strategy,
# which you can call the same as functions, just without any parentheses.
print("UCB actions:")
print(row_strat.actions, " ROW")
print(col_strat.actions, " COL")
print()


print("END  ########################################\n")
########################

