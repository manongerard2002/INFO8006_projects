from pacman_module.game import Agent, Directions
import numpy as np


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


def cycle(state, closed):
    """Adds the state in the closed if it is not already in it.

    Arguments:
        state: a game state
        closed: a set that stores the visited states

    Returns:
        True if the state is in the closed, false otherwise.
    """

    state_key = key(state)

    if state_key in closed:
        return True
    else:
        closed.add(state_key)

    return False


def playerFct(current):
    """Return the next player to play

    Arguments:
        current: value of the previous player + 1

    Returns:
        Return 0 if Pacman is the next player and 1 if it is the ghost.
    """

    return current % 2


def utility(state):
    """Return the score of the state.

    Arguments:
        state: a game state

    Returns:
        Return the score of the state.
    """

    return state.getScore()


def terminalTest(state):
    """Tells wheter a state is terminal or not

    Arguments:
        state: a game state

    Returns:
        True if in the state Pacman lost or won, false otherwise.
    """

    if state.isWin() or state.isLose():
        return True

    return False


def MaxValue(state, player, closed):
    """Returns the minimax value for Pacman, the MAX player

    Arguments:
        state: a game state
        player: the current player
        closed: a set that stores the visited states

    Returns:
        Returns the minimax value for Pacman
    """
    v = -np.inf

    if cycle(state, closed):
        return v

    maximal = []

    for successor, _ in state.generatePacmanSuccessors():
        nextPlayer = playerFct(player + 1)
        closedCopy = closed.copy()
        maximal = minimax(successor, nextPlayer, closedCopy)

        if maximal > v:
            v = maximal

    return v


def MinValue(state, player, closed):
    """Returns the minimax value for the ghost, the MIN player

    Arguments:
        state: a game state
        player: the current player
        closed: a set that stores the visited states

    Returns:
        Returns the minimax value for the ghost
    """
    v = np.inf

    minimal = []

    for successor, _ in state.generateGhostSuccessors(1):
        nextPlayer = playerFct(player + 1)
        closedCopy = closed.copy()
        minimal = minimax(successor, nextPlayer, closedCopy)

        if minimal < v:
            v = minimal

    return v


def minimax(state, player, closed):
    """Call MaxValue if the player is Pacman and MinValue if the player is the
       ghost. If the game reaches a terminal state it return the utility

    Arguments:
        state: a game state
        player: the current player
        closed: a set that stores the visited states

    Returns:
        the utility if the state is terminal, MaxValue if it is the player's
        turn and MinValue if it is the ghost's turn
    """

    if terminalTest(state):
        return utility(state)

    if player == 0:
        return MaxValue(state, player, closed)
    else:
        return MinValue(state, player, closed)


class PacmanAgent(Agent):
    """Pacman agent based on Minimax."""

    def __init__(self):
        super().__init__()

        self.closed = set()

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Return:
            A legal move as defined in `game.Directions`.
        """

        v = -np.inf
        player = 0
        tmpAction = Directions.STOP

        if cycle(state, self.closed):
            return tmpAction

        for successor, action in state.generatePacmanSuccessors():
            nextPlayer = playerFct(player + 1)
            maximal = minimax(successor, nextPlayer, self.closed)

            if maximal > v:
                v = maximal
                tmpAction = action

        return tmpAction
