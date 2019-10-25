import searchAgents as sa
from game import Agent
import numpy as np
import CompetitionPacman
from enum import Enum
import copy
import util
from game import Actions
from game import Directions
import search
import GA_util

class RHE:
    "An agent that does RHE."

    def __init__(self):
        # Adding some momentum can be good
        self.state=CompetitionPacman.NodeState.RUNNING
        self.length = 15
        self.actions = [Directions.WEST, Directions.EAST, Directions.SOUTH, Directions.NORTH]
        self.generations = 200
        self.plan = self.RandomPlan()
        self.mutationRate = 0.7
        self.discount = 0.85
        self.callCount = 0



    def __call__(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        self.callCount += 1
        initialScore = state.getScore()
        fitness = self.getFitness(self.plan, state)
        legalMoves = state.getLegalPacmanActions()
        for i in range(self.generations):
            offspring = self.Mutate(self.plan, state)
            offspringFitness = self.getFitness(offspring,state)

            if(offspringFitness > fitness):
                self.plan = offspring
                fitness = offspringFitness
        if fitness < initialScore:
            self.state = CompetitionPacman.NodeState.FAILED
            return None
        action = self.plan.pop(0)
        self.plan.append(self.RandomAction())
        while(action not in legalMoves):
            action = self.plan.pop(0)
            self.plan.append(self.RandomAction())
        self.state = CompetitionPacman.NodeState.SUCCESS
        return action

        # Filter impossible moves



    def getFitness(self,plan,state):
        nextState = state
        fitness = state.getScore()
        prevScore = fitness
        for actionIndex in range(len(plan)):
            action = plan[actionIndex]
            if action in  nextState.getLegalPacmanActions():
                nextState = nextState.generateSuccessor(0,action)

                if nextState.isWin():
                    return fitness + self.getdiscountedChange(nextState, prevScore, actionIndex) + 200

                if nextState.isLose():
                    return fitness - 400 * (self.discount ** actionIndex)

                pacmanPosition = nextState.getPacmanPosition()
                ghosts = nextState.getGhostPositions()
                index = 1
                for ghost in ghosts:
                    if(nextState.data.agentStates[index].scaredTimer):
                        index += 1
                        continue
                    nextState = self.simulateGhost(ghost, pacmanPosition, index, nextState)
                    index +=1

                    if nextState.isLose():
                        return fitness - 400*(self.discount**actionIndex)

                fitness += self.getdiscountedChange(nextState, prevScore, actionIndex)
                prevScore = nextState.getScore()
            else:
                fitness -= 10*(len(plan)-actionIndex)
        return fitness


    def getdiscountedChange(self, state, prevScore, index):
        newScore = state.getScore()
        delta = newScore - prevScore
        return delta * (self.discount ** index)

    def simulateGhost(self,position, pacmanPosition, index, state):

        legalActions = state.getLegalActions(index)

        bestDistance = np.inf
        bestAction = None
        if len(legalActions)==1:
            return state.generateSuccessor(index, legalActions[0])

        for action in legalActions:
            actionVector = Actions.directionToVector(action, 1)
            newPosition = (position[0] + actionVector[0], position[1] + actionVector[1])

            distance = util.manhattanDistance(newPosition, pacmanPosition)
            if distance < bestDistance:
                bestDistance = distance
                bestAction = action

        return state.generateSuccessor(index, bestAction)


    def Mutate(self, plan, state):
        offspring = copy.copy(plan)
        nextState = state
        for i in range(len(plan)):
            if(nextState.isWin() or nextState.isLose()):
                return offspring
            action = offspring[i]
            legalMoves = nextState.getLegalPacmanActions()
            legalMoves.remove('Stop')
            if(action not in legalMoves):
                offspring[i]=self.RandomLegalAction(legalMoves)
                action = offspring[i]
            elif(np.random.random_sample()<self.mutationRate):
                offspring[i] = self.RandomLegalAction(legalMoves)
                action = offspring[i]
            nextState= nextState.generateSuccessor(0,action)

        return offspring

    def RandomPlan(self):
        return list(np.random.choice(self.actions,self.length,True))
    def RandomAction(self):
        return np.random.choice(self.actions)
    def RandomLegalAction(self, legalActions):
        return np.random.choice(legalActions)