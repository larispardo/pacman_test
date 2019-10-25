from game import Directions
from game import Agent
import numpy as np
from enum import Enum
import copy
import util
from game import Actions

class CompAgent(Agent):
    """ Return <direction> as an action. If <direction> is 'Random' return a random legal action
    """
    def getAction(self, state):
        problem = PositionSearchProblem(state, visualize=False)
        pos = breadthFirstSearchPath(problem)
        problem = positionSearchP(state, goal=pos, visualize=False)
        vals = aStarSearchAvoid(problem, state, AvoidGhosts)
        return vals[0]

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

class positionSearchP(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: x[1]/1000 if x[1]>7 else 1, goal=(1,1), start=None, warn=True, visualize=True):
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
        #if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            #print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):

        isGoal = state == self.goal

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

class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
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
        #if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            #print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

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
    fringe.push((startState, ()), heuristic(startState,problem))
    #print(problem.goal)

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
                    fringe.push(nextNode, heuristic(path[0],problem)
                            + problem.getCostOfActions(newPlan))

def aStarSearchAvoid(problem, state, heuristic=manhattanHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    startState = problem.getStartState()
    visited = set()
    fringe = util.PriorityQueue()
    fringe.push((startState, ()), heuristic(startState,state, "Stop"))
    #print(problem.goal)

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
                    fringe.push(nextNode, heuristic(path[0],state, newPlan[0])
                            + problem.getCostOfActions(newPlan))


def breadthFirstSearchPath(problem):
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
                return currLoc
            successors = problem.getSuccessors(currLoc)
            successorsList = list(successors)
            for i in successorsList:
                if i[0] not in visited:
                    frontier.push((i[0], i[1], currPath + [i[1]]))

def AvoidGhosts(position, state, plan):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    nextState = state.generateSuccessor(0, plan)
    xy2 = state.getGhostPositions()
    heuristic = 0
    for i in range(len(xy2)):
        pos = xy2[i]
        problem = positionSearchP(state, goal=pos, visualize=False)
        problem2 = positionSearchP(nextState, goal=pos, visualize=False)
        vals = breadthFirstSearch(problem)
        vals2 = breadthFirstSearch(problem2)
        timer = state.data.agentStates[i + 1].scaredTimer
        if vals is not None and vals2 is not None and timer < 10:
            if len(vals) < 15 and len(vals2) < len(vals):
                heuristic += 1000 * len(vals2)
    return heuristic

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
