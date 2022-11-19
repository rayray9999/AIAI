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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        return childGameState.getScore()

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
    Your minimax agent (par1-1)
    """
    def getAction(self, gameState):
        # Begin your code
        def minimax(agent, depth, gameState):
            # leaf
            if gameState.isWin() or gameState.isLose() or depth >= self.depth:
                return self.evaluationFunction(gameState)

            # max part
            if agent == 0:
                maximum = float("-inf")
                act=None
                for move in gameState.getLegalActions(agent):
                    nextState = gameState.getNextState(agent, move)
                    value = minimax(1, depth, nextState)
                    if value > maximum:
                        maximum = value
                        act = move
                if depth == 0:  # initial call returns action
                    return act
                else:
                    return maximum

            # min part
            else:
                next_agent=agent+1
                if next_agent==gameState.getNumAgents(): 
                    next_agent=0
                    depth+=1
                minimum=float("inf")
                for move in gameState.getLegalActions(agent):
                    nextState = gameState.getNextState(agent, move)
                    value = minimax(next_agent, depth,nextState)
                    minimum = min(minimum, value)
                return minimum

        return minimax(0,0,gameState)
        # End your code

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (part1-2)
    """

    def getAction(self, gameState):

        def expectimax(agent, depth, gameState):
            # leaf
            if gameState.isWin() or gameState.isLose() or depth >= self.depth:
                return self.evaluationFunction(gameState)

            # max part
            if agent == 0:
                maximum = float("-inf")
                act=None
                for move in gameState.getLegalActions(agent):
                    nextState = gameState.getNextState(agent, move)
                    value = expectimax(1, depth, nextState)
                    if value > maximum:
                        maximum = value
                        act = move
                if depth == 0:  
                    return act
                else:
                    return maximum

            # min part
            else:
                next_agent=agent+1
                if next_agent==gameState.getNumAgents(): 
                    next_agent=0
                    depth+=1
                su=0
                for move in gameState.getLegalActions(agent):
                    nextState=gameState.getNextState(agent, move)
                    su+=expectimax(next_agent, depth,nextState)

                return su/len(gameState.getLegalActions(agent))

        return expectimax(0,0,gameState)
        # End your code

def betterEvaluationFunction(currentGameState):
    gameState = currentGameState
    score = gameState.getScore()
    pac_pos = gameState.getPacmanPosition()
    
    K = 5
    Foods = gameState.getFood().asList()
    food_list = []
    for food_pos in Foods:
        tmp = util.manhattanDistance(pac_pos, food_pos)
        food_list.append(tmp)
    food_list.sort()

    if len(food_list) == 0:
        dis_food = 1
    else:
        dis_food = 0
        K = min(len(food_list), K)
        for i in range(K):
            dis_food += food_list[i]
        dis_food /= K

    sum_distance = 0
    ghost_around = 0
    for ghost_pos in gameState.getGhostPositions():
        distance = util.manhattanDistance(pac_pos, ghost_pos)
        sum_distance += distance
        if distance == 1:
            ghost_around += 1

    num_capsules = len(gameState.getCapsules())

    dis_food = max(dis_food, 1)
    sum_distance = max(sum_distance, 1)
    return score + 1/float(dis_food) - 1/float(sum_distance) - (ghost_around + num_capsules)
    # End your code

# Abbreviation
"""
If you complete this part, please replace scoreEvaluationFunction with betterEvaluationFunction ! !
"""
better = scoreEvaluationFunction # betterEvaluationFunction or scoreEvaluationFunction
