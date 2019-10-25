import searchAgents as sa
from game import Agent
import numpy as np
from enum import Enum
import copy
import util
from game import Actions
from game import Directions
import search
import GA_util

class CompAgent(Agent):
    "An agent that does RHE."

    def __init__(self):
        self.length = 10
        self.actions = [Directions.WEST, Directions.EAST, Directions.SOUTH, Directions.NORTH]
        self.generations = 60
        self.plan = self.RandomPlan()
        self.mutationRate = 0.6
        #self.ghostSimulator = DirectionalGhost()



    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        fitness = self.getFitness(self.plan, state)
        legalMoves = state.getLegalPacmanActions()
        for i in range(self.generations):
            offspring = self.Mutate(self.plan)
            offspringFitness = self.getFitness(offspring,state)
            if(offspringFitness > fitness):
                self.plan = offspring
                fitness = offspringFitness
        action = self.plan.pop(0)
        self.plan.append(self.RandomAction())
        while(action not in legalMoves):
            action = self.plan.pop(0)
            self.plan.append(self.RandomAction())
        return action

        # Filter impossible moves

    def RandomPlan(self):
        return list(np.random.choice(self.actions,self.length,True))
    def RandomAction(self):
        return np.random.choice(self.actions)

    def getFitness(self,plan,state):
        nextState = state
        for action in plan:
            if action  in  nextState.getLegalPacmanActions():
                nextState = nextState.generateSuccessor(0,action)
                ghosts = nextState.getGhostPositions()
                index = 1
                if nextState.isWin():
                    return nextState.getScore()

                if nextState.isLose():
                    return -np.inf

                pacmanPosition = nextState.getPacmanPosition()
                for ghost in ghosts:
                    nextState = self.simulateGhost(ghost, pacmanPosition, index, nextState)
                    if nextState.isWin():
                        return nextState.getScore()

                    if nextState.isLose():
                        return -np.inf

        return nextState.getScore()

    def simulateGhost(self,position, pacmanPosition, index, state):

        legalActions = state.getLegalActions(index)

        bestDistance = np.inf
        bestAction = None

        for action in legalActions:
            actionVector = Actions.directionToVector(action, 1)
            newPosition = (position[0] + actionVector[0], position[1] + actionVector[1])

            distance = util.manhattanDistance(newPosition, pacmanPosition)
            if distance < bestDistance:
                bestDistance = distance
                bestAction = action

        return state.generateSuccessor(index, bestAction)


    def Mutate(self, plan):
        offspring = copy.copy(plan)
        for i in range(len(plan)):
            if(np.random.random_sample()<self.mutationRate):
                offspring[i]=self.RandomAction()
        return offspring



