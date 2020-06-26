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
    # grand_table.to_latex("Game_" + str(game_id))

    for i, proportion in enumerate(proportions):
        replicator_dynamic = ReplicatorDynamic(grand_table)
        replicator_dynamic.run(proportion, name="Game_" + str(game_id) + "_" + str(i+1))


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




