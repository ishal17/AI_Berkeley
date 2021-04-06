# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()
        foodList = newFood.asList()

        if len(foodList) == 0:
            foodBonus = 1
        else:
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            foodBonus = 1/(min(foodDistances) + 1)
               
        return score + foodBonus

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        action, value = self.minimax(gameState, self.depth * gameState.getNumAgents(), 0)
        return action


    def minimax(self, state, depth, idx):
        if depth == 0 or state.isWin() or state.isLose():
            return None, self.evaluationFunction(state)

        legalMoves = state.getLegalActions(idx)
        if idx == 0: # pacman
            maxScore = float('-inf')
            bestAction = None
            for action in legalMoves:
                successor = state.generateSuccessor(idx, action)
                _, successorScore = self.minimax(successor, depth-1, idx+1)
                if maxScore < successorScore:
                    maxScore = successorScore
                    bestAction = action
            return bestAction, maxScore
        else: # smart ghosts
            minScore = float('inf')
            worstAction = None
            for action in legalMoves:
                successor = state.generateSuccessor(idx, action)
                _, successorScore = self.minimax(successor, depth-1, (idx+1) % state.getNumAgents())
                if minScore > successorScore:
                    minScore = successorScore
                    worstAction = action
            return worstAction, minScore

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        action, value = self.alphabeta(gameState, self.depth * gameState.getNumAgents(), float('-inf'), float('inf'), 0)
        return action


    def alphabeta(self, state, depth, alpha, beta, idx):
        if depth == 0 or state.isWin() or state.isLose():
            return None, self.evaluationFunction(state)

        legalMoves = state.getLegalActions(idx)
        if idx == 0: # pacman
            maxScore = float('-inf')
            bestAction = None
            for action in legalMoves:
                successor = state.generateSuccessor(idx, action)
                _, successorScore = self.alphabeta(successor, depth-1, alpha, beta, idx+1)                
                if maxScore < successorScore:
                    maxScore = successorScore
                    bestAction = action
                alpha = max(alpha, maxScore)
                if alpha > beta:
                    break
            return bestAction, maxScore
        else: # smart ghosts
            minScore = float('inf')
            worstAction = None
            for action in legalMoves:
                successor = state.generateSuccessor(idx, action)
                _, successorScore = self.alphabeta(successor, depth-1, alpha, beta, (idx+1) % state.getNumAgents())
                if minScore > successorScore:
                    minScore = successorScore
                    worstAction = action
                beta = min(beta, minScore)
                if beta < alpha:
                    break
            return worstAction, minScore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        action, value = self.expectimax(gameState, self.depth * gameState.getNumAgents(), 0)
        return action


    def expectimax(self, state, depth, idx):
        if depth == 0 or state.isWin() or state.isLose():
            return None, self.evaluationFunction(state)

        legalMoves = state.getLegalActions(idx)
        if idx == 0: # pacman
            maxScore = float('-inf')
            bestAction = None
            for action in legalMoves:
                successor = state.generateSuccessor(idx, action)
                _, successorScore = self.expectimax(successor, depth-1, idx+1)
                if maxScore < successorScore:
                    maxScore = successorScore
                    bestAction = action
            return bestAction, maxScore
        else: # random ghosts
            totalScore = 0
            for action in legalMoves:
                successor = state.generateSuccessor(idx, action)
                _, successorScore = self.expectimax(successor, depth-1, (idx+1) % state.getNumAgents())
                totalScore += successorScore
            return None, totalScore / len(legalMoves)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: bonus combines 3 components:
                    1.  foodBonus - function of distance to nearest food
                                    varies from 0 to 1 (bigger when food is closer)
                                    coefficient: 1
                    2.  ghostBonus - function of distance to nearest ghost
                                     varies from 0 to 1 (bigger when ghost is far away)
                                     coefficient: -1
                    3.  powerBonus - activates when pacman eats power pill and pushes pacman to eat nearest ghost
                                     product of distance to nearest ghost and scared time left
                                     coefficient: 10
    """
    me = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostList = currentGameState.getGhostPositions()

    foodDistances = [manhattanDistance(me, food) for food in foodList]
    ghostDistances = [manhattanDistance(me, ghost) for ghost in ghostList]
    scaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]

    foodBonus = 1
    if len(foodList) > 0:
        foodBonus = 1/(1 + min(foodDistances))

    ghostBonus = 1
    if len(ghostList) > 0:
        ghostBonus = 1/(1 + min(ghostDistances))

    powerBonus = ghostBonus*min(scaredTimes)
    bonus = foodBonus - ghostBonus + powerBonus*10
    
    return currentGameState.getScore() + bonus

# Abbreviation
better = betterEvaluationFunction
