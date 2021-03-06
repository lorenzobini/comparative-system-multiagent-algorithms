# NOTE: You may change the way actions and the payoff matrix are represented.
#  However if you do so, you'll have to update the __repr__ method accordingly.
#  Also FixedMatrixSuite will have to be updated to reflect your changes, which can be quite a bit of work.
#
# You may not change the actions and payoffs of the matrix games in FixedMatrixSuite, only their representation.

import abc
import random
from typing import List, Tuple, Dict

# Define custom types for actions and payoffs.
Payoff = float
Action = int


class MatrixSuite(metaclass=abc.ABCMeta):
    """Abstract representation of a suite of matrix games.

    Class attributes:
        *name*: Name of the matrix suite.
                If it only generates one type of matrix,
                it can also be the name of that matrix type. (i.e. Constant Sum)

        *row_actions*: List of actions that the row player has in the current matrix.

        *col_actions*: List of actions that the column player has in the current matrix.

        *payoff_matrix*: A 2D list containing tuples with the payoffs of the row and column players.

        Indices of the outer list are row actions.

        Indices of the inner list are column actions.

        The first item of the tuple is the row player payoff,
        the second item is the column player payoff.
    """
    name: str
    row_actions: List[Action]
    col_actions: List[Action]
    payoff_matrix: List[List[Tuple[Payoff, Payoff]]]

    @abc.abstractmethod
    def generate_new_payoff_matrix(self) -> None:
        """Generate a new payoff matrix and update the row and column actions accordingly."""
        pass

    def __repr__(self) -> str:
        """Prettified string representation of the matrix, useful for testing.
        This will show if you use **print()** on an instance of this class."""
        out: str = ""
        for ra in self.row_actions:
            for ca in self.col_actions:
                out += self.payoff_matrix[ra][ca].__repr__() + " "
            out += "\n"
        return out

    # Here you can add more methods to make implementing strategies easier,
    # for example a method that returns a given players actions.
    def get_actions(self, player: str) -> List[Action]:
        """Return the actions of the given player.
        :param player: A string of either 'row' or 'col',
        representing which player the strategy is currently playing as.
        """
        if player == "row":
            return self.row_actions
        elif player == "col":
            return self.col_actions
        else:
            raise Exception("ERROR: *player* should be either 'row' of 'col', not '" + player + "'!")

    def get_row_actions(self) -> List[Action]:
        """Return the actions of the row player.
        """
        return self.row_actions

    def get_col_actions(self) -> List[Action]:
        """Return the actions of the column player.
        """
        return self.col_actions

    def get_payoffs(self, row_action: int, col_action: int) -> Tuple[float, float]:
        """Return the payoffs of both players given their respective actions.
        """
        value = 0.0
        try:
            value = self.payoff_matrix[row_action][col_action]
        except:
            print(self.payoff_matrix)
            print(str(row_action), str(col_action))

        return value

    def get_max_payoff(self, player: int) -> float:
        """Return the maximum possible payoff for the specified player.
        0 = row player
        1 = column player
        """
        max_payoff = 0
        for row in self.payoff_matrix:
            for payoffs in row:
                if payoffs[player] > max_payoff:
                    max_payoff = payoffs[player]

        return max_payoff



class FixedMatrixSuite(MatrixSuite):
    """Predetermined suite of matrices, don't use with more than 9 restarts, because it will run out of matrices.

    Class attributes:
        *matrices*: A dictionary of the matrices and their attributes.

        Structured like this,

        key: index, to be matched with *k*

        value: List containing the number of row actions,
        the number of column actions and
        the payoff matrix in that order.

        *k*: Number of the matrix that is currently active.
    """
    matrices: Dict[int, Tuple[int, int, List[List[Tuple[Payoff, Payoff]]]]]
    k: int

    def __init__(self) -> None:
        """Initialize the suite and 'generate' the first payoff matrix."""
        self.name = "Fixed Matrix Suite"
        self.k = 0

        self.matrices = {
            1: (2, 2, [[(2, 2), (6, 0)], [(0, 6), (4, 4)]]),
            2: (2, 2, [[(9, 2), (2, 8)], [(8, 0), (1, 7)]]),
            3: (2, 2, [[(1, 8), (1, 1)], [(2, 1), (2, 9)]]),
            4: (3, 3, [[(1, 8), (9, 0), (6, 3)], [(9, 0), (0, 9), (0, 9)], [(9, 0), (2, 7), (9, 0)]]),
            5: (3, 3, [[(3, 3), (4, 4), (9, 9)], [(2, 2), (0, 0), (6, 6)], [(1, 1), (5, 5), (8, 8)]]),
            6: (3, 3, [[(9, 1), (0, 2), (10, 1)], [(8, 2), (8, 2), (8, 0)], [(0, 1), (1, 1), (1, 9)]]),
            7: (3, 3, [[(2, 8), (1, 8), (7, 1)], [(7, 0), (1, 7), (8, 2)], [(7, 2), (7, 2), (8, 0)]]),
            8: (3, 3, [[(2, 2), (4, 1), (6, 0)], [(1, 4), (3, 3), (5, 2)], [(0, 6), (2, 5), (4, 4)]]),
            9: (4, 4,
                [[(5, 9), (0, 10), (9, 6), (3, 2)], [(7, 7), (1, 1), (4, 1), (1, 7)], [(3, 0), (4, 0), (9, 3), (5, 9)],
                 [(2, 1), (2, 7), (0, 10), (0, 9)]]),
            10: (4, 4, [[(10, 0), (4, 6), (5, 5), (8, 2)], [(8, 2), (5, 5), (6, 4), (10, 0)],
                        [(5, 5), (8, 2), (0, 10), (8, 2)], [(1, 9), (4, 6), (7, 3), (6, 4)]])
        }

        self.generate_new_payoff_matrix()

    def __repr__(self) -> str:
        """Add some extra information to the print of this class."""
        out = self.name + ": Matrix " + self.k.__repr__() + "\n"
        out += super().__repr__()  # Add the representation of the superclass ( GameSuite.__repr__() ).
        return out

    def generate_new_payoff_matrix(self) -> None:
        """Not so much generate as just getting the next matrix out of the dictionary."""
        self.k += 1
        try:
            v = self.matrices[self.k]
        except KeyError:
            raise Exception("Key is not in matrix dictionary, you probably did too many restarts.")
        self.row_actions = list(range(v[0]))
        self.col_actions = list(range(v[1]))
        self.payoff_matrix = v[2]


# Add the other game suites below

class RandomIntMatrixSuite(MatrixSuite):
    """
    Generates a matrix of N rows per M columns with random integers payoffs.
    Class attributes:
        *matrix_suite: The MatrixSuite that the game is played on
    """

    def __init__(self) -> None:
        self.name = "Random Int Matrix Suite"

        self.generate_new_payoff_matrix()

    def __repr__(self) -> str:
        """Add some extra information to the print of this class."""
        out = self.name + ": Matrix randomly generated (INT Payoffs) \n"
        out += super().__repr__()  # Add the representation of the superclass ( GameSuite.__repr__() ).
        return out

    def generate_new_payoff_matrix(self) -> None:
        n_row = random.randint(2, 5)
        n_col = random.randint(2, 5)

        matrix = []
        payoffs = Tuple[Payoff, Payoff]

        for row in range(0, n_row):
            matrix_row = []
            for col in range(0, n_col):
                payoffs = (random.randint(0, 3), random.randint(0, 3))
                matrix_row.append(payoffs)
            matrix.append(matrix_row)

        self.row_actions = list(range(n_row))
        self.col_actions = list(range(n_col))
        self.payoff_matrix = matrix


class RandomFloatMatrixSuite(MatrixSuite):
    """
    Generates a matrix of N rows per M columns with random float payoffs.
    Parameters:
        *matrix_suite: The MatrixSuite that the game is played on
    """

    def __init__(self) -> None:
        self.name = "Random Float Matrix Suite"

        self.generate_new_payoff_matrix()

    def __repr__(self) -> str:
        """Add some extra information to the print of this class."""
        out = self.name + ": Matrix randomly generated (Float Payoffs) \n"
        out += super().__repr__()  # Add the representation of the superclass ( GameSuite.__repr__() ).
        return out

    def generate_new_payoff_matrix(self) -> None:
        n_row = random.randint(2, 5)
        n_col = random.randint(2, 5)

        matrix = []
        payoffs = Tuple[Payoff, Payoff]

        for row in range(0, n_row):
            matrix_row = []
            for col in range(0, n_col):
                payoffs = (round(random.uniform(0.0, 3.0), 3), round(random.uniform(0.0, 3.0), 3))
                matrix_row.append(payoffs)
            matrix.append(matrix_row)

        self.row_actions = list(range(n_row))
        self.col_actions = list(range(n_col))
        self.payoff_matrix = matrix
