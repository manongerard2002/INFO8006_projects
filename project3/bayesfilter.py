import numpy as np
from pacman_module.game import Agent, Directions, manhattanDistance
from pacman_module.util import PriorityQueue, manhattanDistance
from pacman_module import util


def f(position, cost, pos_max_beliefs):
    """
    Given Pacman position, the cost and estimates position of the ghost
    returns the estimated cost of cheapest solution.
    g is the path cost given as argument.
    h is the manhattan distance from the pacman to the ghost

    Arguments:
        position: the position of Pacman.
        cost: the cost of the game.
        pos_max_beliefs: the best estimated position of the ghost.

    Returns:
        The estimated cost of cheapest solution.
    """

    g = cost
    h = manhattanDistance(pos_max_beliefs, position)

    return g + h


def isWin(positionPacman, pos_max_beliefs):
    """Returns whether the pacman arrived at the ghost estimated position.

    Arguments:
        position: the position of Pacman.
        pos_max_beliefs: the best estimated position of the ghost.

    Returns:
        True if pacman and the ghost are at the same position, false otherwise.
    """
    if positionPacman == pos_max_beliefs:
        return True

    return False


def coefBinomial(n, m):
    """Compute the binomial coefficient nCm : "n choose m"

    Arguments:
        n: first parameter of the binomial coefficient, the amount we have
        m: second parameter of the binomial coefficient, the amout we choose

    Returns:
        The binomial coeffiecient nCm, or 0 if the parameters are wrong
    """

    if n >= 0 and m >= 0 and n >= m:
        num = 1
        m_fact = 1
        n_m_fact = 1
        for i in range(1, n + 1, 1):
            num *= i
            if i <= m:
                m_fact *= i

            if i <= n - m:
                n_m_fact *= i

        return num / (m_fact * n_m_fact)
    return 0


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        self.ghost = ghost

        self.fear = 1.0  # by default afraid
        if self.ghost == 'fearless':
            self.fear = 0.0
        if self.ghost == 'terrified':
            self.fear = 3.0

    def transition_matrix(self, walls, position):
        """Builds the transitiion matrix

            T_t = P(X_t | X_{t-1})

        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, k, l)
            of T_t is the probability P(X_t = (i, j) | X_{t-1} = (k, l)) for
            the ghost to move from (k, l) to (i, j).
        """

        W = walls.width
        H = walls.height
        T_t = np.zeros((W, H, W, H))
        successor_position = np.zeros((W, H), dtype=object)

        for k in range(W):
            for L in range(H):
                if walls[k][L] is False:
                    actions = set()
                    if k - 1 >= 0 and walls[k - 1][L] is False:
                        actions.add((k - 1, L))
                    if k + 1 < W and walls[k + 1][L] is False:
                        actions.add((k + 1, L))
                    if L - 1 >= 0 and walls[k][L - 1] is False:
                        actions.add((k, L - 1))
                    if L + 1 < H and walls[k][L + 1] is False:
                        actions.add((k, L + 1))

                    successor_position[k][L] = actions

                # to avoid error of no len(action)
                else:
                    successor_position[k][L] = set()

        for k in range(W):
            for L in range(H):
                distance = manhattanDistance([k, L], position)
                actions = successor_position[k][L]
                number_actions = len(actions)
                dist = np.zeros(number_actions)
                succ_pos = np.zeros(number_actions, dtype=tuple)

                total = 0
                i = 0
                for a in actions:
                    succ_pos[i] = a
                    succ_distance = manhattanDistance(succ_pos[i], position)

                    dist[i] = 2**self.fear if succ_distance >= distance else 1
                    total += dist[i]
                    i += 1

                for i in range(number_actions):
                    T_t[succ_pos[i][0], succ_pos[i][1]][k][L] = dist[i] / total

        return T_t

    def observation_matrix(self, walls, evidence, position):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """

        W = walls.width
        H = walls.height
        O_t = np.zeros((W, H))

        n = 4
        p = 0.5
        for i in range(W):
            for j in range(H):
                if walls[i][j] is False:
                    X_t = [i, j]
                    z = evidence - manhattanDistance(position, X_t) + n*p
                    O_t[i][j] = coefBinomial(n, z) * p**z * (1-p)**(n-z)

        return O_t

    def update(self, walls, belief, evidence, position):
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1})

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """

        T = self.transition_matrix(walls, position)
        O_t = self.observation_matrix(walls, evidence, position)
        W = walls.width
        H = walls.height
        b_t = np.zeros((W, H))
        inverseZ = 0

        for i in range(W):
            for j in range(H):
                sum = 0
                for k in range(W):
                    for L in range(H):
                        sum += T[i][j][k][L] * belief[k][L]
                b_t[i][j] = O_t[i, j] * sum
                inverseZ += b_t[i][j]

        for i in range(W):
            for j in range(H):
                b_t[i][j] /= inverseZ

        return b_t

    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """

        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(walls, beliefs[i], evidences[i],
                                             position)

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()

        self.score = -1  # because ever the first time in get_action we do +1
        self.number_eaten = 0

    def _get_action(self, walls, beliefs, eaten, position):
        """
        Arguments:
            walls: The W x H grid of walls.
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        self.score += 1

        ghost_eaten = 0
        for i in range(beliefs.shape[0]):
            if eaten[i] is True:
                ghost_eaten += 1

        if self.number_eaten != ghost_eaten:
            self.score -= 200*(ghost_eaten - self.number_eaten)
            self.number_eaten = ghost_eaten

        W = walls.width
        H = walls.height
        shape = beliefs[0].shape
        idx = shape[0] * shape[1]

        ghost_not_eaten = []

        for i in range(beliefs.shape[0]):
            if eaten[i] is False:
                ghost_not_eaten.append(i)

        pos_max_beliefs_ = np.zeros(len(ghost_not_eaten), dtype=object)
        astar_path = np.zeros(len(ghost_not_eaten), dtype=object)
        astar_cost = np.zeros(len(ghost_not_eaten))

        j = 0
        for i in ghost_not_eaten:
            max_beliefs_ghost = np.unravel_index(np.ndarray.argmax(beliefs[i]),
                                                 shape)
            tmp = set()

            if (max_beliefs_ghost[0] == position[0]
                    and max_beliefs_ghost[1] == position[1]):

                k = (idx - 1) - 1  # to have the second argmax
                tmp.add(np.unravel_index(np.argpartition(beliefs[i], k,
                                         axis=None)[k], shape))
            else:
                tmp.add((max_beliefs_ghost[0], max_beliefs_ghost[1]))

            pos_max_beliefs_[j] = tmp
            j += 1

        for i in range(len(ghost_not_eaten)):
            tmp = self.astar(walls, position, tuple(pos_max_beliefs_[i])[0])
            astar_path[i] = tmp[0]
            astar_cost[i] = tmp[1]

        return astar_path[np.ndarray.argmin(astar_cost)]

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )

    def astar(self, walls, position, pos_max_beliefs):
        """Returns a legal moves to solve the search layout.

        Arguments:
            walls: The W x H grid of walls.
            position: the position of Pacman.
            pos_max_beliefs: the best estimated position of the ghost.

        Returns:
            A legal moves.
        """

        path = []
        cost = self.score

        if isWin(position, pos_max_beliefs):
            return (path, cost)

        fringe = PriorityQueue()
        fringe.push((path, position, cost), f(position, cost, pos_max_beliefs))
        closed = set()

        W = walls.width
        H = walls.height

        while True:
            if fringe.isEmpty():
                return ([], cost)

            _, (path, positionPacman, cost) = fringe.pop()

            if isWin(positionPacman, pos_max_beliefs):
                return (path[0], cost)

            if positionPacman in closed:
                continue
            else:
                closed.add(positionPacman)

            k = positionPacman[0]
            L = positionPacman[1]

            successor_position = set()
            if k - 1 >= 0 and walls[k - 1][L] is False:
                successor_position.add((k - 1, L))
            if k + 1 < W and walls[k + 1][L] is False:
                successor_position.add((k + 1, L))
            if L - 1 >= 0 and walls[k][L - 1] is False:
                successor_position.add((k, L - 1))
            if L + 1 < H and walls[k][L + 1] is False:
                successor_position.add((k, L + 1))

            for positions in successor_position:
                successorCost = cost + 1
                newPosition = positions

                deltaX = newPosition[0] - position[0]
                deltaY = newPosition[1] - position[1]

                if deltaY > 0:
                    action = Directions.NORTH
                elif deltaY < 0:
                    action = Directions.SOUTH
                elif deltaX < 0:
                    action = Directions.WEST
                elif deltaX > 0:
                    action = Directions.EAST

                fringe.push((path + [action], newPosition, successorCost),
                            f(newPosition, successorCost, pos_max_beliefs))

        return (path[0], cost)
