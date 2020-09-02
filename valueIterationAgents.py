import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
            counter = util.Counter()
            for state in states:
                maxVal = float("-inf")
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    tempVal = self.computeQValueFromValues(state, action)
                    if tempVal > maxVal:
                        maxVal = tempVal
                    counter[state] = maxVal
            self.values = counter


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        pairs = self.mdp.getTransitionStatesAndProbs(state, action)
        qVal = 0
        # for next_state, prob in action_prob_pairs:
        for pair in pairs:
            next_state = pair[0]
            prob = pair[1]
            reward = self.mdp.getReward(state, action, next_state)
            qVal += prob * (reward + self.discount * self.getValue(next_state))
        return qVal
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction = None
        # initialize the action, and None will be returned if no legal actions
        maxVal = float("-inf")
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            qVal = self.computeQValueFromValues(state, action)
            if qVal > maxVal:
                maxVal = qVal
                bestAction = action
        return bestAction
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(self.iterations):
            # counter = util.counter
            state = states[i % len(states)]
            if state != 'TERMINAL_STATE':
                actions = self.mdp.getPossibleActions(state)
                maxVal = float("-inf")
                for action in actions:
                    qVal = self.computeQValueFromValues(state, action)
                    if qVal > maxVal:
                        maxVal = qVal
                self.values[state] = maxVal



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        predecessors = [set() for i in states]
        priorityQueue = util.PriorityQueue()
        for state in states:
            if state != 'TERMINAL_STATE':
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    nextStateList = []
                    for pair in self.mdp.getTransitionStatesAndProbs(state, action):
                        nextStateList.append(pair[0])
                    for nextState in nextStateList:
                        predecessors[states.index(nextState)].add(state)
        for state in states:
            if state != 'TERMINAL_STATE':
                actions = self.mdp.getPossibleActions(state)
                maxVal = float("-inf")
                for action in actions:
                    qVal = self.computeQValueFromValues(state, action)
                    if qVal > maxVal:
                        maxVal = qVal
                diffVal = abs(self.values[state] - maxVal)
                priorityQueue.update(state, -diffVal)
        for i in range(self.iterations):
            if priorityQueue.isEmpty():
                break
            state = priorityQueue.pop()
            if state != 'TERMINAL_STATE':
                actions = self.mdp.getPossibleActions(state)
                maxVal = float("-inf")
                for action in actions:
                    qVal = self.computeQValueFromValues(state, action)
                    if qVal > maxVal:
                        maxVal = qVal
                self.values[state] = maxVal
            for predecessor in predecessors[states.index(state)]:
                if state != 'TERMINAL_STATE':
                    actions = self.mdp.getPossibleActions(predecessor)
                    maxVal = float("-inf")
                    for action in actions:
                        qVal = self.computeQValueFromValues(predecessor, action)
                        if qVal > maxVal:
                            maxVal = qVal
                    diffVal = abs(self.values[predecessor] - maxVal)
                    if diffVal > self.theta:
                        priorityQueue.update(predecessor, -diffVal)


