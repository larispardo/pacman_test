import numpy as np
from enum import Enum
import searchAgents as sa
import search
import copy


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
class AllGhostsScared:
    def __init__(self):
        self.state = NodeState.RUNNING

    def __call__(self, state):

        timer = []
        for index in range(1, len(state.data.agentStates)):
            timer.append(state.data.agentStates[index].scaredTimer)

        if all(i > 10 for i in timer):
            self.state = NodeState.SUCCESS
            return None
        self.state = NodeState.FAILED
        return None

class EatGhosts:
    def __init__(self):
        self.state = NodeState.RUNNING

    def __call__(self, state):
        gpos = state.getGhostPositions()
        offsetx = 3
        offsety = 2
        walls = state.getWalls()
        top, right = walls.height - 2, walls.width - 2
        timer = []
        for index in range(1, len(state.data.agentStates)):
            timer.append(state.data.agentStates[index].scaredTimer)
        if sum(timer) > 10/len(gpos):
            vals = []
            vals_len = []
            for index in range(len(gpos)):
                if timer[index] > 10:
                    tmppos = gpos[index]
                    x = int(tmppos[0])
                    y = int(tmppos[1])
                    newpos=(x,y)
                    if (x < right/2+offsetx and x > right/2-offsetx) and (y < top/2+offsety and y > top/2-offsety):
                        continue
                    problem = sa.positionSearchP(state, goal=newpos, visualize=False)
                    tmp_val = search.aStarSearchAvoid(problem,state,sa.AvoidGhosts)
                    vals.append(tmp_val)
                    vals_len.append(len(tmp_val))
                    self.state = NodeState.SUCCESS
            if len(vals) > 0:
                return vals[np.argmin(vals_len)][0]
        self.state = NodeState.FAILED
        return None

class CheckCorners:

    def __init__(self):
        self.state = NodeState.RUNNING
        self.firstEaten = False

    def __call__(self, state):
        capsules = state.getCapsules()
        minammountCapsules = 2
        ## Set deppending on the level!
        if (len(capsules) < minammountCapsules):
            self.firstEaten = True
        else:
            self.firstEaten = False
        if not self.firstEaten:
            vals = []
            for capsule in capsules:
                problem = sa.positionSearchP(state, goal=capsule, visualize=False)
                vals.append(search.aStarSearchAvoid(problem,state, sa.AvoidGhosts))
            path = sorted(vals, key=len)[0]
            self.state = NodeState.SUCCESS
            return path[0]
        else:
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
            '''
            problem = sa.positionSearchP(state, goal=gpos[index-1], visualize=False)
            vals = search.aStarSearch(problem)
            if state.data.agentStates[index].scaredTimer < 0.01 and len(vals) < 5:
            '''
            if state.data.agentStates[index].scaredTimer < 0.01 and gdist[index-1] < 1.1:
                self.state = NodeState.SUCCESS
                return None
        self.state = NodeState.FAILED
        return self.direction

        '''
        if all(i > 1 for i in gdist):
            self.state = NodeState.FAILED
            return self.direction
        '''

class breathfirst:
    """ Return <direction> as an action. If <direction> is 'Random' return a random legal action
    """
    def __init__(self):
        self.state = NodeState.RUNNING

    def __call__(self, state):
        problem = sa.PositionSearchProblem(state)
        vals = search.breadthFirstSearch(problem)
        self.state = NodeState.SUCCESS
        return vals[0]

class aStar:
    """ Return <direction> as an action. If <direction> is 'Random' return a random legal action
    """
    def __init__(self):
        self.state = NodeState.RUNNING

    def __call__(self, state):
        self.state = NodeState.SUCCESS
        problem = sa.PositionSearchProblem(state, visualize=False)
        pos = search.breadthFirstSearchPath(problem)
        problem = sa.positionSearchP(state, goal=pos, visualize=False)
        vals = search.aStarSearchAvoid(problem, state, sa.AvoidGhosts)
        return vals[0]

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
        """ YOUR CODE HERE!"""

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
    elif genome[0].startswith("EatGhosts"):
        parent.add_child(EatGhosts())
        if len(genome) > 1:
            parse_node(genome[1:], parent)
    elif genome[0].startswith("AllGhostsScared"):
        parent.add_child(AllGhostsScared())
        if len(genome) > 1:
            parse_node(genome[1:], parent)
    else:
        print("Unrecognized in ")
        raise Exception

    return parent




