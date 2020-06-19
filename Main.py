# Note: From this script the program can be run.
# You may change everything about this file.

from MatrixSuite import FixedMatrixSuite
from MatrixSuite import RandomIntMatrixSuite
from MatrixSuite import RandomFloatMatrixSuite
import Strategies
import Game
from GrandTable import GrandTable
from ReplicatorDynamic import ReplicatorDynamic, Proportions
import Nash

# Output some basic things to show how to call the classes.
matrix_suite = FixedMatrixSuite()
strategies = [Strategies.Aselect(), Strategies.Aselect(), Strategies.Aselect()]
grand_table = GrandTable(matrix_suite, strategies, 9, 1000)

table = grand_table.play_games()

ReplicatorDynamic.run(table, strategies)

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



# Example of how to test a strategy:
matrix_suite = FixedMatrixSuite()  # Create a matrix suite

strat = Strategies.Aselect()  # Create the strategy you want to test.

strat.initialize(matrix_suite, "row")  # Initialise it with the game suite and as either "row" or "col" player.

action = strat.get_action(1)  # Get the next action
print("Strategy plays action:" + action.__repr__())

strat.update(1, action, 1.5, 1)  # Update the strategy with a fake payoff and opponent action.
# Now you might want to look at the class attributes of the strategy,
# which you can call the same as functions, just without any parentheses.
print("Aselect actions:")
print(strat.actions)
print()


# Test to see if gambit runs properly, see Section 5 of the assignment and Nash.py
m = [[3, 0, 5],
     [1, 0, 1],
     [3, 1, 3]]
print("Gambit test result:")
Nash.run_gambit(strategies, m)
# The output should be this:
# ======================|
#  Aselect: 1.00 | 1.00 |
#  Aselect: ---- | ---- |
#  Aselect: ---- | ---- |
# ======================|
#  Aselect: 1.00 | ---- |
#  Aselect: ---- | ---- |
#  Aselect: ---- | 1.00 |
# ======================|
#  Aselect: ---- | 1.00 |
#  Aselect: ---- | ---- |
#  Aselect: 1.00 | ---- |
# ======================|
