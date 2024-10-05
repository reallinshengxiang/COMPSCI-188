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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        score = successorGameState.getScore()
        for ghostState in newGhostStates:
            ghost_position = ghostState.getPosition()
            ghost_distance = manhattanDistance(newPos, ghost_position)
            if ghostState.scaredTimer > 0:
                score = score + 200/(ghost_distance + 1)
            else:
                if ghost_distance <= 1:
                    score = float("-inf")
        foodList = newFood.asList()
        currentCapsules = currentGameState.getCapsules()
        if currentCapsules:
            minCapsuleDistance = min([manhattanDistance(newPos, capsule) for capsule in currentCapsules])
            score = score + 500 / (minCapsuleDistance + 1)
        if foodList:
            minFoodDistance = min([manhattanDistance(newPos, food) for food in foodList])
            score = score + 10 / (minFoodDistance + 1)
        if currentGameState.getNumFood() > successorGameState.getNumFood():
            score = score + 100
        return score
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        "*** YOUR CODE HERE ***"

        def minimax(agentIndex, depth, gameState):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            LegalActions = gameState.getLegalActions(agentIndex)
            if not LegalActions:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                ans = float('-inf')
                for action in LegalActions:
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    ans = max(ans, minimax(1, depth, successorState))
            else:
                ans = float('inf')
                nextAgent = agentIndex + 1
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    depth = depth + 1
                for action in LegalActions:
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    ans = min(ans, minimax(nextAgent, depth, successorState))
            return ans

        bestAction = None
        bestScore = float('-inf')
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, successorState)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphabeta(agentIndex, depth, gameState, alpha, beta):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            LegalActions = gameState.getLegalActions(agentIndex)
            if not LegalActions:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                ans = float('-inf')
                for action in LegalActions:
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    ans = max(ans, alphabeta(1, depth, successorState, alpha, beta))
                    if ans > beta:
                        return ans
                    alpha = max(alpha, ans)
            else:
                ans = float('inf')
                nextAgent = agentIndex + 1
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    depth = depth + 1
                for action in LegalActions:
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    ans = min(ans, alphabeta(nextAgent, depth, successorState, alpha, beta))
                    if ans < alpha:
                        return ans
                    beta = min(beta, ans)
            return ans

        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            score = alphabeta(1, 0, successorState, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(agentIndex, depth, gameState):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            LegalActions = gameState.getLegalActions(agentIndex)
            if not LegalActions:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:
                ans = float('-inf')
                for action in LegalActions:
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    ans = max(ans, expectimax(1, depth, successorState))
                return ans
            else:
                ans = 0
                nextAgent = agentIndex + 1
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    depth = depth + 1
                for action in LegalActions:
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    ans = ans + expectimax(nextAgent, depth, successorState)/len(LegalActions)
            return ans

        bestAction = None
        bestScore = float('-inf')

        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            score = expectimax(1, 0, successorState)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    nowPos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    nowFood = currentGameState.getFood()
    foodList = nowFood.asList()
    nowGhostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in nowGhostStates]
    closestFoodDist = min((manhattanDistance(nowPos, foodPos) for foodPos in foodList), default=0)
    closestGhostDist = float("inf")
    for ghost in nowGhostStates:
        ghostPosition = ghost.getPosition()
        #print(ghost.scaredTimer)
        if ghost.scaredTimer==0:
            closestGhostDist = min(closestGhostDist, manhattanDistance(nowPos, ghostPosition))
    #minscaredtime = float("inf")
    #for ghost in nowGhostStates:
    #    minscaredtime = min(ghost.scaredTimer,minscaredtime)
    remainingFoodCount = currentGameState.getFood().count()
    scaredBonus = sum(scaredTimes) > 0
    score = score - 2 * closestFoodDist
    score = score +  5 / (closestGhostDist + 1e-6)
    score = score -  20 * remainingFoodCount
    score += 200 if scaredBonus else 0
    if remainingFoodCount == 0:
        score += 1000
    elif closestFoodDist == 0:
        score += 500
    if closestGhostDist < 3 :
        score = score - 1000
    if closestGhostDist == 0:
        score = float("-inf")
    return score
    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
