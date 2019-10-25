import numpy as np
from enum import Enum
import searchAgents as sa
import search
import copy

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


class CheckCapsules:

    def __init__(self):
        self.state = NodeState.RUNNING
        self.firstEaten = False

    def __call__(self, state):
        capsules = state.getCapsules()
        ## Set deppending on the level!
        if(len(capsules)!=2):
            self.firstEaten = True
        else:
            self.firstEaten = False
        if not self.firstEaten:
            vals =[]
            for capsule in capsules:
                problem = sa.positionSearchP(state, goal=capsule)
                vals.append(search.aStarSearch(problem, sa.manhattanHeuristic))
            path = sorted(vals, key=len)[0]
            self.state = NodeState.SUCCESS
            return path[0]
        else:
            self.state = NodeState.FAILED
            return None


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

    elif genome[0].startswith("CheckCapsules"):
        arg = genome[0].split('.')[-1]
        parent.add_child(CheckCapsules())
        if len(genome) > 1:
            parse_node(genome[1:], parent)
    elif genome[0].startswith("aStar"):
        parent.add_child(aStar())
        if len(genome) > 1:
            parse_node(genome[1:], parent)
    else:
        print("Unrecognized in ")
        raise Exception

    return parent




