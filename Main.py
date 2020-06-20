# Note: From this script the program can be run.
# You may change everything about this file.

import MatrixSuite
from MatrixSuite import FixedMatrixSuite
from MatrixSuite import RandomIntMatrixSuite
from MatrixSuite import RandomFloatMatrixSuite
import Strategies
import Game
from GrandTable import GrandTable
from ReplicatorDynamic import ReplicatorDynamic, Proportions
import Nash
from typing import List



''' Setting up strategies and relative proportions '''
strategies = [Strategies.Aselect(), Strategies.EGreedy(), Strategies.UCB(), Strategies.Satisficing(),
              Strategies.Softmax(), Strategies.FictitiousPlay(), Strategies.Bully,
              Strategies.ProportionalRegretMatching()]
strategies_ext = strategies.append(Strategies.Custom())  #TODO: replace name
proportions = [Proportions.uniform_without_own_strat, Proportions.non_uniform_without_own_strat]
proportions_ext = [Proportions.uniform_with_own_strat, Proportions.non_uniform_with_own_strat]



def game_session(game_id: int, matrix_suite: MatrixSuite, strategies: List[Strategies],
                 proportions: List[float], restarts: int, rounds=1000) -> None:
    print("\n==============================================\n")

    print("GAME ", game_id, " \n")
    grand_table = GrandTable(matrix_suite, strategies, restarts, rounds)

    grand_table.play_games()
    print(grand_table)

    replicator_dynamic = ReplicatorDynamic(grand_table)
    for proportion in proportions:
        replicator_dynamic.run(proportion)

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
game_session(3, RandomFloatMatrixSuite(), strategies, proportions, 19)
# SIXTH GAME: Extended set of strategies, Random Float Matrix Suite
game_session(3, RandomFloatMatrixSuite(), strategies_ext, proportions_ext, 19)


'''

############# TEST AREA
print("BEGIN  ########################################\n")


matrix_suite = RandomIntMatrixSuite()  # Create a matrix suite

row_strat = Strategies.ProportionalRegretMatching()  # Create the strategy you want to test.
col_strat = Strategies.ProportionalRegretMatching()
row_strat.initialize(matrix_suite, "row")  # Initialise it with the game suite and as either "row" or "col" player.
col_strat.initialize(matrix_suite, "col")

for round in range(0, 10):
    row_action = row_strat.get_action(round)  # Get the next action
    col_action = col_strat.get_action(round)
    print("Row plays action:" + row_action.__repr__())

    row_payoff, col_payoff = matrix_suite.get_payoffs(row_action, col_action)

    row_strat.update(round, row_action, row_payoff, col_action)
    col_strat.update(round, col_action, col_payoff, row_action)# Update the strategy with a fake payoff and opponent action.
# Now you might want to look at the class attributes of the strategy,
# which you can call the same as functions, just without any parentheses.
print("Bully actions:")
print(row_strat.actions, " ROW")
print(col_strat.actions, " COL")
print()


print("END  ########################################\n")
########################


'''