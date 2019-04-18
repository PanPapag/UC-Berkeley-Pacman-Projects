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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

        "*** YOUR CODE HERE ***"
        newFoodList = newFood.asList()
        remainingFood = len(newFoodList)
        stepsGhostIsScared = sum(newScaredTimes)
        ghostDist = 0

        # Calculate distance from closest ghost
        closestGhost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
        if not stepsGhostIsScared:
            if closestGhost:
                ghostDist = -10 / closestGhost
            else:
                ghostDist = -1000
        else:
            for steps in range(stepsGhostIsScared):
                ghostDist += 10 / closestGhost
        # Calculate distance from closest food
        if newFoodList:
            closestFood = min([manhattanDistance(newPos, foodPos) for foodPos in newFoodList])
        else:
            closestFood = 0

        return ghostDist - closestFood - (100 * remainingFood)

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

    def isTerminalNode(self, gameState):
        if gameState.isWin() or gameState.isLose():
            return True
        else:
            return False

    def isMaximizingPlayer(self, agentIndex):
        if agentIndex == 0:
            return True
        elif agentIndex >= 1:
            return False

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def miniMax(self, gameState, depth, agentIndex):
        # when all agents' possible outcomes in this level have been examined go one level deeper
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            depth -= 1

        if depth == 0 or self.isTerminalNode(gameState):
            return self.evaluationFunction(gameState)

        if self.isMaximizingPlayer(agentIndex):
            return self.maxValue(gameState, depth, agentIndex)

        else:
            return self.minValue(gameState, depth, agentIndex)

    def maxValue(self, gameState, depth, agentIndex):
        maximum = ("unknown", float("-inf"))

        if not gameState.getLegalActions(agentIndex):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(agentIndex):
            if action == Directions.STOP:
                continue

            retVal = self.miniMax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)

            maxAction, maxVal = maximum
            # check if retVal has been returned by not legal move condition
            if type(retVal) is not tuple:
                newVal = retVal
            else:
                _ ,newVal = retVal

            if newVal > maxVal:
                maximum = (action, newVal)

        return maximum

    def minValue(self, gameState, depth, agentIndex):
        minimum = ("unknown", float("inf"))

        if not gameState.getLegalActions(agentIndex):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(agentIndex):
            if action == Directions.STOP:
                continue

            retVal = self.miniMax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)

            minAction, minVal = minimum
            # check if retVal has been returned by not legal move condition
            if type(retVal) is not tuple:
                newVal = retVal
            else:
                _ ,newVal = retVal

            if newVal < minVal:
                minimum = (action, newVal)


        return minimum

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
        """
        "*** YOUR CODE HERE ***"
        bestAction, _ = self.miniMax(gameState, self.depth, 0)
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphaBeta(self, gameState, depth, alpha, beta, agentIndex):
        # when all agents' possible outcomes in this level have been examined go one level deeper
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            depth -= 1

        if depth == 0 or self.isTerminalNode(gameState):
            return self.evaluationFunction(gameState)

        if self.isMaximizingPlayer(agentIndex):
            return self.maxValue(gameState, depth, alpha, beta, agentIndex)

        else:
            return self.minValue(gameState, depth, alpha, beta, agentIndex)

    def maxValue(self, gameState, depth, alpha, beta, agentIndex):
        maximum = ("unknown", float("-inf"))

        if not gameState.getLegalActions(agentIndex):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(agentIndex):
            if action == Directions.STOP:
                continue

            retVal = self.alphaBeta(gameState.generateSuccessor(agentIndex, action), depth, alpha, beta, agentIndex + 1)

            maxAction, maxVal = maximum
            # check if retVal has been returned by not legal move condition
            if type(retVal) is not tuple:
                newVal = retVal
            else:
                _ ,newVal = retVal

            if newVal > maxVal:
                maximum = (action, newVal)

            if maximum[1] > beta:
                return maximum
            alpha = max(alpha, maximum[1])

        return maximum

    def minValue(self, gameState, depth, alpha, beta, agentIndex):
        minimum = ("unknown", float("inf"))

        if not gameState.getLegalActions(agentIndex):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(agentIndex):
            if action == Directions.STOP:
                continue

            retVal = self.alphaBeta(gameState.generateSuccessor(agentIndex, action), depth, alpha, beta, agentIndex + 1)

            minAction, minVal = minimum
            # check if retVal has been returned by not legal move condition
            if type(retVal) is not tuple:
                newVal = retVal
            else:
                _ ,newVal = retVal

            if newVal < minVal:
                minimum = (action, newVal)

            if minimum[1] < alpha:
                return minimum
            beta = min(beta, minimum[1])

        return minimum

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestAction, _ = self.alphaBeta(gameState, self.depth, float("-inf"), float("inf"), 0)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimaxMax(self, gameState, depth, agentIndex):
        # when all agents' possible outcomes in this level have been examined go one level deeper
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            depth -= 1

        if depth == 0 or self.isTerminalNode(gameState):
            return self.evaluationFunction(gameState)

        if self.isMaximizingPlayer(agentIndex):
            return self.maxValue(gameState, depth, agentIndex)

        else:
            return self.expValue(gameState, depth, agentIndex)

    def maxValue(self, gameState, depth, agentIndex):
        maximum = ("unknown", float("-inf"))

        legalActions = gameState.getLegalActions(agentIndex)

        if not gameState.getLegalActions(agentIndex):
            return self.evaluationFunction(gameState)

        for action in legalActions:
            if action == Directions.STOP:
                continue

            retVal = self.expectiMax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)

            maxAction, maxVal = maximum
            # check if retVal has been returned by not legal move condition
            if type(retVal) is not tuple:
                newVal = retVal
            else:
                _ ,newVal = retVal

            if newVal > maxVal:
                maximum = (action, newVal)

        return maximum

    def expValue(self, gameState, depth, agentIndex):
        exp = ("unknown", float("inf"))

        legalActions = gameState.getLegalActions(agentIndex)
        actionPropability = 1.0 / len(legalActions)
        avgScore = 0

        if not gameState.getLegalActions(agentIndex):
            return self.evaluationFunction(gameState)

        for action in legalActions:
            if action == Directions.STOP:
                continue

            retVal = self.expectiMax(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
            avgScore += actionPropability * retVal[1]

            expAction, expVal = exp
            # check if retVal has been returned by not legal move condition
            if type(retVal) is not tuple:
                newVal = retVal
            else:
                _ ,newVal = retVal

            if newVal < minVal:
                minimum = (action, newVal)


        return minimum

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
