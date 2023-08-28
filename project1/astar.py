from pacman_module.game import Agent, Directions
from pacman_module.util import PriorityQueueWithFunction, manhattanDistance


def key(state):
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key tuple.
    """

    return (
        state.getPacmanPosition(),
        state.getFood()
    )


def f(item):
    """Given a Pacman game state, path and path cost returns the estimated cost
    of cheapest solution through the state.
    g is the path cost given as argument.
    h is the manhattan distance from the state to the furthest food

    Arguments:
        item: a tuple with a state, path and path cost

    Returns:
        The estimated cost of cheapest solution.
    """

    g = item[2]
    h = 0
    xy1 = item[0].getPacmanPosition()
    xy2 = [0, 0]

    result = set()
    foodMatrix = item[0].getFood()
    for col in range(foodMatrix.width):
        for row in range(foodMatrix.height):
            if foodMatrix[col][row]:
                xy2[0] = col
                xy2[1] = row
                result.add(manhattanDistance(xy1, xy2))

    if result:
        h = max(result)

    return g + h


class PacmanAgent(Agent):
    """Pacman agent based on A*."""

    def __init__(self):
        super().__init__()

        self.moves = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Return:
            A legal move as defined in `game.Directions`.
        """

        if self.moves is None:
            self.moves = self.astar(state)

        if self.moves:
            return self.moves.pop(0)
        else:
            return Directions.STOP

    def astar(self, state):
        """Given a Pacman game state, returns a list of legal moves to solve
        the search layout.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A list of legal moves.
        """

        path = []

        if state.isWin():
            return path

        cost = state.getScore()
        fringe = PriorityQueueWithFunction(f)
        fringe.push((state, path, cost))
        closed = set()

        while True:
            if fringe.isEmpty():
                return []

            _, (current, path, cost) = fringe.pop()

            if current.isWin():
                return path

            current_key = key(current)

            if current_key in closed:
                continue
            else:
                closed.add(current_key)

            capsules = current.getCapsules()

            for successor, action in current.generatePacmanSuccessors():
                successorCost = cost + 1
                if successor.getCapsules() != capsules:
                    successorCost += 5
                fringe.push((successor, path + [action], successorCost))

        return path
