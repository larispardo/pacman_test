from game import Agent
from game import Directions
import numpy as np
from enum import Enum
from RollingHorizonE import RHE
import util
from game import Actions

class CompAgent(Agent):
    def __init__(self):
        '''
        This agent is an agent that uses the a behaviour tree to play pacman
        It relies heavily on the astar algorithm tat checks for ghosts to work.
        '''
        self.encode = ["SEL",
                       "RHE.none",
                       "aStar.none"]

        self.tree = parse_node(self.encode, None)

    def getAction(self, state):
        state.getFood()
        # ag = ClosestDotSearchAgent()
        action = self.tree(state)
        if action not in state.getLegalPacmanActions():
            # print "Illegal action!!"
            action = 'Stop'
        return action


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
                # print("Sequence", node(state))
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
                # print("I am here",action)
                self.state = NodeState.SUCCESS
                return action
        self.state = NodeState.FAILED


class aStar:
    """ Return <direction> as an action. If <direction> is 'Random' return a random legal action
    """

    def __init__(self):
        self.state = NodeState.RUNNING

    def __call__(self, state):
        self.state = NodeState.SUCCESS
        problem = FoodSearchProblem(state, visualize=False)
        pos = breadthFirstSearchPath(problem)
        problem = positionSearchP(state, goal=pos, visualize=False)
        vals = aStarSearchAvoid(problem, state, AvoidGhosts)
        return vals[0]

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

    elif genome[0].startswith("aStar"):
        parent.add_child(aStar())
        if len(genome) > 1:
            parse_node(genome[1:], parent)
    elif genome[0].startswith("RHE"):
        parent.add_child(RHE())
        if len(genome) > 1:
            parse_node(genome[1:], parent)
    else:
        print("Unrecognized in ")
        raise Exception

    return parent


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

class GhostSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    In this case we want to find the closest ghosts to pacman using the breath first algorithm
    """

    def __init__(self, gameState, costFn = lambda x: 1, start=None, warn=True, visualize=True):
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
        self.costFn = costFn
        self.visualize = visualize
        #if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            #print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        '''
        instead of using an specific goal we want to find the closest ghost to pacman.
        '''
        GP = self.currentstate.getGhostPositions()
        pos = []
        # might be good for eating the closest one
        for position in GP:
            x = round(position[0]+.1)
            y = round(position[1]+.1)
            pos.append((x,y))
        isGoal = state in GP
        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

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
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
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
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
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


class FoodSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    this problem is used when searching for food.
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


def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def aStarSearch(problem, heuristic=manhattanHeuristic):
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




def breadthFirstSearchPath(problem):
    """
    Search the shallowest nodes in the search tree first.
    instead of giving the list of actions it returns the position of the closest one.
    """
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
                return currLoc
            successors = problem.getSuccessors(currLoc)
            successorsList = list(successors)
            for i in successorsList:
                if i[0] not in visited:
                    frontier.push((i[0], i[1], currPath + [i[1]]))


def AvoidGhosts(state, plan):
    ## Maybe the length of the plan can be used to not calculate the heuristic after some steps
    ## this will make this faster.
    closestAccepted = 6
    nextState = state.generateSuccessor(0, plan)
    xy2 = state.getGhostPositions()
    heuristic = 0
    '''
    If two ghosts intercept Pac in a non corner, then pacman moves as a "next move"
    closer to either one of them without regarding if there is an exist, 
    then the closest path becomes the closest one, not the safest!
    '''

    for i in range(len(xy2)):
        pos = xy2[i]
        timer = state.data.agentStates[i + 1].scaredTimer
        if timer > 10:
            continue
        ghProblem = GhostSearchProblem(state, visualize=False)
        closestGh = breadthFirstSearch(ghProblem)
        if closestGh is not None:
            if (len(closestGh) > closestAccepted):
                return heuristic
        #Think best option
        problem = positionSearchP(state, goal=pos, visualize=False)
        problem2 = positionSearchP(nextState, goal=pos, visualize=False)
        vals = aStarSearch(problem,manhattanHeuristic)
        vals2 = aStarSearch(problem2, manhattanHeuristic)
        if vals is not None and vals2 is not None:
            if len(vals)<closestAccepted and len(vals2)<=len(vals):
                heuristic += 1000000*closestAccepted-(len(vals2))
    return heuristic

def aStarSearchAvoid(problem, state, heuristic=AvoidGhosts):
    """
    Search the node that has the lowest combined cost and heuristic first.
    The heuristic given to this is always
    """
    startState = problem.getStartState()
    visited = set()
    fringe = util.PriorityQueue()
    fringe.push((startState, ()), heuristic( state, "Stop"))
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
                    fringe.push(nextNode, heuristic(state, newPlan[0])
                                + problem.getCostOfActions(newPlan))
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
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