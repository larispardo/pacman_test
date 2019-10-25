from game import Agent
from game import Directions
import numpy as np
from enum import Enum
import copy
import util
from game import Actions

class CompAgent(Agent):
    def __init__(self):
        self.leaf = ["SEL",
             ["SEQ", "Valid.North", "Danger.North", "GoNot.North"],
             ["SEQ", "Valid.East", "Danger.East", "GoNot.East"],
             ["SEQ", "Valid.South", "Danger.South", "GoNot.South"],
             ["SEQ", "Valid.West", "Danger.West", "GoNot.West"],
             ["SEQ","GhostsCorners","CheckCorners"],
             ["SEL", "aStar.none", "breathfirst.none"],
                       "Go.Random"]# Just to be able to run
        self.tree = parse_node(self.leaf, None)

    def getAction(self, state):
        state.getFood()
        action = self.tree(state)
        if action not in state.getLegalPacmanActions():
            action = 'Stop'
        return action

opposites = {
    "North" : "South",
    "South": "North",
    "West": "East",
    "East": "West",
}

class NodeState(Enum):
    RUNNING = 1
    SUCCESS = 2
    FAILED = 3

class Sequence:
    """ Continues until one failure is found."""
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.state = NodeState.RUNNING

    def add_child(self, child):
        self.children.append(child)
        return child

    def __call__(self, state):
        self.state = NodeState.RUNNING
        for node in self.children:
            action = node(state)
            if node.state == NodeState.FAILED:
                #print("Sequence", node(state))
                self.state = NodeState.FAILED
                return action
        self.state = NodeState.SUCCESS
        return action


class Selector:
    """ Continues until one success is found."""
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.state = NodeState.RUNNING

    def add_child(self, child):
        self.children.append(child)
        return child

    def __call__(self, state):
        self.state = NodeState.RUNNING
        for node in self.children:
            action = node(state)
            if node.state == NodeState.SUCCESS:
                #print("I am here",action)
                self.state = NodeState.SUCCESS
                return action
        self.state = NodeState.FAILED



class CheckValid:
    """ Check whether <direction> is a valid action for PacMan
    """
    def __init__(self, direction):
        self.direction = direction
        self.state = NodeState.RUNNING

    def __call__(self, state):
        legalMoves = state.getLegalActions()

        if self.direction in legalMoves:
            self.state = NodeState.SUCCESS
            return self.direction
        self.state = NodeState.FAILED
        return None

class GhostsCorners:
    def __init__(self):
        self.state = NodeState.RUNNING

    def __call__(self, state):

        timer = []
        for index in range(1, len(state.data.agentStates)):
            timer.append(state.data.agentStates[index].scaredTimer)
        if sum(timer) > 10:
            self.state = NodeState.FAILED
            return None
        self.state = NodeState.SUCCESS
        return None

class CheckCorners:

    def __init__(self):
        self.state = NodeState.RUNNING

    def __call__(self, state):
        walls = state.getWalls()
        top, right = walls.height - 2, walls.width - 2
        corners = ((1, 1), (1, top), (right, top), (right, 1))
        capsules = state.getCapsules()
        for corner in corners:
            #print(corner)
            if state.getFood()[corner[0]][corner[1]] or corner in capsules:
                problem = positionSearchP(state, goal=corner)
                vals = aStarSearch(problem, manhattanHeuristic)
                self.state = NodeState.SUCCESS
                return vals[0]
        self.state = NodeState.FAILED
        return None

class CheckDanger:
    """ Check whether there is a ghost in <direction>, or any of the adjacent fields.
    """
    def __init__(self, direction):
        self.direction = direction
        self.state = NodeState.RUNNING

    def __call__(self, state):
        successorGameState = state.generatePacmanSuccessor(self.direction)
        newPos = successorGameState.getPacmanPosition()

        gpos = successorGameState.getGhostPositions()
        #print(state.data.layout)
        gdist = [abs(newPos[0] - b[0])+ abs(newPos[1] - b[1]) for b in gpos]
        for index in range(1, len(state.data.agentStates)):

            if state.data.agentStates[index].scaredTimer < 0.01 and gdist[index-1] < 1.1:
                self.state = NodeState.SUCCESS
                return None
        self.state = NodeState.FAILED
        return self.direction


class breathfirst:
    """ Return <direction> as an action. If <direction> is 'Random' return a random legal action
    """
    def __init__(self):
        self.state = NodeState.RUNNING

    def __call__(self, state):
        problem = FoodSearchProblem(state)
        vals = breadthFirstSearch(problem)
        self.state = NodeState.SUCCESS
        return vals[0]

class aStar:
    """ Return <direction> as an action. If <direction> is 'Random' return a random legal action
    """
    def __init__(self):
        self.state = NodeState.RUNNING

    def __call__(self, state):
        walls = state.getWalls()
        top, height = walls.height - 2, walls.height-2

        capsules = state.getCapsules()
        timer=[]
        for index in range(1, len(state.data.agentStates)):
            timer.append(state.data.agentStates[index].scaredTimer)

        if len(capsules) == 0 and not sum(timer) > 10:
            self.state = NodeState.FAILED
            return None
        elif sum(timer) > 10:
            gpos = state.getGhostPositions()
            x =int(gpos[0][0])
            y = int(gpos[0][1])
            tmp_tuple = (x,y)
            newpos=[tmp_tuple]
            x = int(gpos[1][0])
            y = int(gpos[1][1])
            tmp_tuple = (x, y)
            newpos.append(tmp_tuple)
            vals = []
            vals_len = []
            for index in range(1, len(state.data.agentStates)):
                if state.data.agentStates[index].scaredTimer > 10:
                    problem = positionSearchP(state, goal=newpos[index - 1])
                    tmp_val = breadthFirstSearch(problem)
                    vals.append(tmp_val)
                    vals_len.append(len(tmp_val))
                    self.state = NodeState.SUCCESS
            if len(vals) > 0:
                return vals[np.argmin(vals_len)][0]
        else:
            problem =positionSearchP(state, goal=capsules[0])
            vals = aStarSearch(problem,manhattanHeuristic)
            self.state = NodeState.SUCCESS
            return vals[0]

        '''
        '''

class ActionGo:
    """ Return <direction> as an action. If <direction> is 'Random' return a random legal action
    """
    def __init__(self, direction="Random"):
        self.state = NodeState.RUNNING
        self.direction = direction

    def __call__(self, state):
        if self.direction == "Random":
            return None
        else:
            self.state = NodeState.SUCCESS
            return self.direction


class ActionGoNot:
    """ Go in a random direction that isn't <direction>
    """
    def __init__(self, direction):
        self.state = NodeState.RUNNING
        self.direction = direction

    def __call__(self, state):
        
        self.state = NodeState.SUCCESS
        newPos = state.getPacmanPosition()

        gpos = state.getGhostPositions()
        x=[]
        y=[]
        walls = state.getWalls()
        top = walls.height - 2
        for ghost in gpos:
            x.append(ghost[0])
            y.append(ghost[1])
        '''
            This will only work for two ghosts, the idea is that both are in the 
            same x or y position, then they have a higher chance of cornering pacman
        '''
        legalMoves = state.getLegalActions()
        # This to make sure it does not go into the square of death (where the ghosts spawn).
        if opposites[self.direction] in legalMoves and newPos[1]>top/2:
            return opposites[self.direction]
        checklegal = copy.copy(legalMoves)
        checklegal = [i for i in checklegal if i != self.direction]
        checklegal = [i for i in checklegal if i != "Stop"]
        if len(set(x)) == 1 and x[0] == newPos[0]:
            checklegal = [i for i in checklegal if i != "North"]
            checklegal = [i for i in checklegal if i != "South"]
        if len(set(y)) == 1 and y[0] == newPos[1]:
            checklegal = [i for i in checklegal if i != "West"]
            checklegal = [i for i in checklegal if i != "East"]
        if len(checklegal)>0:
            #print('Choosing chekc',checklegal)
            move = checklegal[np.random.randint(len(checklegal))]
            return move
        #print("legal move", legalMoves)
        legalMoves = [i for i in legalMoves if i != self.direction]
        legalMoves = [i for i in legalMoves if i != "Stop"]
        if len(legalMoves)>0:
            move = legalMoves[np.random.randint(len(legalMoves))]
            #print('choosing legal', move)
            return move
        return "Stop"


class DecoratorInvert:
    def __call__(self, arg):
        return not arg





def parse_node(genome, parent=None):
    if isinstance(genome[0], list):
        parse_node(genome[0], parent)
        parse_node(genome[1:], parent)

    elif genome[0] is "SEQ":
        if parent is not None:
            node = parent.add_child(Sequence(parent))
        else:
            node = Sequence(parent)
            parent = node
        parse_node(genome[1:], node)

    elif genome[0] is "SEL":
        if parent is not None:
            node = parent.add_child(Selector(parent))
        else:
            node = Selector(parent)
            parent = node
        parse_node(genome[1:], node)

    elif genome[0].startswith("Valid"):
        arg = genome[0].split('.')[-1]
        parent.add_child(CheckValid(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0].startswith("Danger"):
        arg = genome[0].split('.')[-1]
        parent.add_child(CheckDanger(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0].startswith("GoNot"):
        arg = genome[0].split('.')[-1]
        parent.add_child(ActionGoNot(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0].startswith("Go"):
        arg = genome[0].split('.')[-1]
        parent.add_child(ActionGo(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0] is ("Invert"):
        arg = genome[0].split('.')[-1]
        parent.add_child(DecoratorInvert(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)
    elif genome[0].startswith("aStar"):
        parent.add_child(aStar())
        if len(genome) > 1:
            parse_node(genome[1:], parent)
    elif genome[0].startswith("GhostsCorners"):
        parent.add_child(GhostsCorners())
        if len(genome) > 1:
            parse_node(genome[1:], parent)
    elif genome[0].startswith("CheckCorners"):
        parent.add_child(CheckCorners())
        if len(genome) > 1:
            parse_node(genome[1:], parent)
    elif genome[0].startswith("breathfirst"):
        parent.add_child(breathfirst())
        if len(genome) > 1:
            parse_node(genome[1:], parent)
    else:
        print("Unrecognized in ")
        raise Exception

    return parent


def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a FoodSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    frontier = util.Queue()
    visited = []
    startNode = (problem.getStartState(), None, [])
    frontier.push(startNode)
    while not frontier.isEmpty():
        curr = frontier.pop()
        currLoc = curr[0]
        currDir = curr[1]
        currPath = curr[2]
        if (currLoc not in visited):
            visited.append(currLoc)
            if (problem.isGoalState(currLoc)):
                return currPath
            successors = problem.getSuccessors(currLoc)
            successorsList = list(successors)
            for i in successorsList:
                if i[0] not in visited:
                    frontier.push((i[0], i[1], currPath + [i[1]]))


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    startState = problem.getStartState()
    visited = set()
    fringe = util.PriorityQueue()
    fringe.push((startState, ()), heuristic(startState, problem))
    # print(problem.goal)

    while not fringe.isEmpty():
        currNode = fringe.pop()
        currState = currNode[0]
        currPlan = currNode[1]
        if problem.isGoalState(currState):
            return list(currPlan)
        if not currState in visited:
            visited.add(currState)
            paths = problem.getSuccessors(currState)
            for path in paths:
                newPlan = list(currPlan)
                newPlan.append(path[1])
                nextNode = (path[0], tuple(newPlan))
                if not path[0] in visited:
                    fringe.push(nextNode, heuristic(path[0], problem)
                                + problem.getCostOfActions(newPlan))


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


class FoodSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.currentstate = gameState
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        # if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
        # print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        food = self.currentstate.getFood()
        isGoal = food[state[0]][state[1]]

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display):  # @UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist)  # @UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost


class positionSearchP(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn=lambda x: x[1] / 1000 if x[1] > 7 else 1, goal=(1, 1), start=None, warn=True,
                 visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.currentstate = gameState
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        # if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
        # print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):

        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display):  # @UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist)  # @UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0

        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)

            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost