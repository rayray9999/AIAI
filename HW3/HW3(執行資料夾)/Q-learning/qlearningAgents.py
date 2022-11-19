# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

"""
part 2-2 & part 2-3
"""

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # Begin your code
        self.q_values=util.Counter() #set initial Q value
        # End your code


    def getQValue(self, state, action):
        # Begin your code
        return self.q_values[(state, action)] #return what it ask
        # End your code


    def computeValueFromQValues(self, state):
        LegalActions = self.getLegalActions(state)
        temp = util.Counter()
        if len(LegalActions)==0:
            return 0.0
        for action in LegalActions:
            temp[action] = self.getQValue(state, action)
        return temp[temp.argMax()]
        # End your code

    def computeActionFromQValues(self, state):
        actions = self.getLegalActions(state)
        max_val = float('-inf')
        best_action = None
        for action in actions:
          q_value = self.getQValue(state, action)
          if max_val < q_value:
            max_val = q_value
            best_action = action
        return best_action
        # End your code

    def getAction(self, state):
        legalActions = self.getLegalActions(state)

        "*** YOUR CODE HERE ***"
        # Begin your code
        explore = util.flipCoin(self.epsilon)
        if explore:
            return random.choice(legalActions)
        else:
            return self.getPolicy(state)
        # End your code
        

    def update(self, state, action, nextState, reward):
        # Begin your code
        a = self.alpha
        r = reward
        g = self.discount
        old_q = self.getQValue(state, action)
        if nextState:
            self.q_values[(state, action)] = (1 - a) * old_q + a * (r + g * self.getValue(nextState))
        else:
            self.q_values[(state, action)] = (1 - a) * old_q + a * r
        # End your code

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


"""
part 2-4
"""

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        features = self.featExtractor.getFeatures(state, action)
        Q_val = 0
        for i in features:
            Q_val += features[i] * self.weights[i]
        return Q_val
        # End your code

    def update(self, state, action, nextState, reward):
        cor = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for i in features:
            self.weights[i] = self.weights[i] + self.alpha * cor * features[i]

        # End your code


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
