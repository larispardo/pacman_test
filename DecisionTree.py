from enum import Enum


class NodeState(Enum):
    READY = 1
    VISITING = 2
    SUCCESS = 3
    FAILED = 4
    RUNNING = 5


class Node:
    def __init__(self, action):
        self.state = NodeState.READY
        self.action = action

    def execute(self, state):
        self.state = NodeState.RUNNING
        result = self.action(state)
        if result:
            self.state = NodeState.SUCCESS
        else:
            self.state = NodeState.FAILED


class Sequence(Node):
    def __init__(self, children):
        self.state = NodeState.READY
        self.children = children

    def execute(self, state):
        self.state = NodeState.RUNNING
        for node in self.children:
            node.execute(state)
            if node.state == NodeState.FAILED:
                self.state = NodeState.FAILED
                return
        self.state = NodeState.SUCCESS


class Selector(Node):
    def __init__(self, children):
        self.state = NodeState.READY
        self.children = children

    def execute(self, state):
        self.state = NodeState.RUNNING
        for node in self.children:
            node.execute(state)
            if node.state == NodeState.SUCCESS:
                print("I am here")
                self.state = NodeState.SUCCESS
                return
        self.state = NodeState.FAILED
