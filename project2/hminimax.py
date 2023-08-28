from pacman_module.game import Agent, Directions
import numpy as np

MAXDEPTH = 5


def key(state):
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key tuple.
    """

    return (
        state.getPacmanPosition(),
        state.getFood(),
        state.getGhostPosition(1),
        state.getGhostDirection(1)
    )


def cycleDict(state, closedDict):
    """Adds the state in the closedDict if it is not already in it.

    Arguments:
        state: a game state
        closedDict: a dictionary that stores the visited states and the number
                    of time visited

    Returns:
        True if the state is in the closed, false otherwise.
    """

    state_key = key(state)

    if state_key in closedDict:
        value = closedDict.get(state_key)
        closedDict[state_key] = value + 1
        return True
    else:
        closedDict[state_key] = 1

    return False


def playerFct(current):
    """Return the next player to play

    Arguments:
        current: value of the previous player + 1

    Returns:
        Return 0 if Pacman is the next player and 1 if it is the ghost.
    """

    return current % 2


def eval(state, closed):
    """Calculates an estimate of the expected utility of the game at the state

    Arguments:
        state: a game state
        closed: a set that stores the visited states

    Returns:
        An estimate of the expected utility of the game
    """

    h = 0
    g = state.getScore()

    for i in closed.values():
        if i > 1:
            g = g - 10**i

    xy1 = state.getPacmanPosition()
    foodMatrixList = state.getFood().asList()

    if not foodMatrixList:
        return g

    smallestBox = [foodMatrixList[0][0], foodMatrixList[0][1],
                   foodMatrixList[0][0], foodMatrixList[0][1]]

    for (i, j) in foodMatrixList:
        if i < smallestBox[0]:
            smallestBox[0] = i
        elif i > smallestBox[2]:
            smallestBox[2] = i
        if j > smallestBox[1]:
            smallestBox[1] = j
        elif j < smallestBox[3]:
            smallestBox[3] = j

    deltaXLeft = xy1[0] - smallestBox[0]
    deltaXRight = xy1[0] - smallestBox[2]
    deltaYUp = xy1[1] - smallestBox[1]
    deltaYDown = xy1[1] - smallestBox[3]
    height = abs(smallestBox[2] - smallestBox[0])
    width = abs(smallestBox[3] - smallestBox[1])

    if deltaXLeft == deltaXRight:
        if deltaXLeft >= 0:
            h += deltaXLeft
        else:
            h += - deltaXLeft
    elif deltaXLeft < 0:
        h += -deltaXLeft + width
    elif deltaXRight > 0:
        h += deltaXRight + width
    else:
        if deltaXLeft < - deltaXRight:
            h += deltaXLeft + width
        else:
            h += - deltaXRight + width

    if deltaYUp == deltaYDown:
        if deltaYUp >= 0:
            h += deltaYUp
        else:
            h += - deltaYUp
    elif deltaYUp > 0:
        h += deltaYUp + height
    elif deltaYDown < 0:
        h += - deltaYDown + height
    else:
        if - deltaYUp < deltaYDown:
            h += - deltaYUp + height
        else:
            h += deltaYDown + height

    return g - h


def cutoffTest(state, depth):
    """Tells when to stop expanding a state

    Arguments:
        state: a game state
        depth: the amont of actions the ghost and pacman have done up to this
               iteration

    Returns:
        True if in the state Pacman lost or won and if the depth is higher than
        a fixed value, false otherwise.
    """

    if state.isWin() or state.isLose() or depth > MAXDEPTH:
        return True

    return False


def MaxValue(state, player, closed, depth):
    """Returns the hminimax value for Pacman, the MAX player

    Arguments:
        state: a game state
        player: the current player
        closed: a set that stores the visited states
        depth: the amont of actions the ghost and pacman have done up to this
               iteration

    Returns:
        Returns the hminimax value for Pacman
    """

    v = -np.inf

    cycleDict(state, closed)

    maximal = []

    for successor, _ in state.generatePacmanSuccessors():
        nextPlayer = playerFct(player + 1)
        closedCopy = closed.copy()
        maximal = hminimax(successor, nextPlayer, closedCopy, depth)

        if maximal > v:
            v = maximal

    return v


def MinValue(state, player, closed, depth):
    """Returns the hminimax value for the ghost, the MIN player

    Arguments:
        state: a game state
        player: the current player
        closed: a set that stores the visited states
        depth: the amont of actions the ghost and pacman have done up to this
               iteration

    Returns:
        Returns the minimax value for the ghost
    """

    v = np.inf

    minimal = []

    for successor, _ in state.generateGhostSuccessors(1):
        nextPlayer = playerFct(player + 1)
        closedCopy = closed.copy()
        minimal = hminimax(successor, nextPlayer, closedCopy, depth)

        if minimal < v:
            v = minimal

    return v


def hminimax(state, player, closed, depth):
    """Call MaxValue if the player is Pacman and MinValue if the player is the
       ghost. If the game reaches a cutoff state it returns eval

    Arguments:
        state: a game state
        player: the current player
        closed: a set that stores the visited states
        depth: the amont of actions the ghost and pacman have done up to this
               iteration

    Returns:
        eval if the cutoff test is true, MaxValue if it is the player's turn
        and MinValue if it is the ghost's turn
    """

    if cutoffTest(state, depth):
        return eval(state, closed)

    if player == 0:
        return MaxValue(state, player, closed, depth + 1)
    else:
        return MinValue(state, player, closed, depth + 1)


class PacmanAgent(Agent):
    """Pacman agent based on hminimax."""

    def __init__(self):
        super().__init__()

        self.closedDict = dict()

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Return:
            A legal move as defined in `game.Directions`.
        """

        v = -np.inf
        player = 0
        depth = 0
        tmpAction = Directions.STOP

        cycleDict(state, self.closedDict)

        for successor, action in state.generatePacmanSuccessors():
            nextPlayer = playerFct(player + 1)
            closedCopy = self.closedDict.copy()
            maximal = hminimax(successor, nextPlayer, closedCopy, depth)

            if maximal > v:
                v = maximal
                tmpAction = action

        return tmpAction
