# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from util import Stack

    path = [] # Every state keeps it's path from the starting state
    visited = set() # Set data structure is the most appropriate to use in this case
    stateAgenda = Stack() # Keeping states into a stack in order to implement DFS

    # Check if initial state is goal states
    if problem.isGoalState(problem.getStartState()):
        return path
    # otherwise push initial state into the stack
    stateAgenda.push((problem.getStartState(), path))

    while(True):
        # terminate condition
        if stateAgenda.isEmpty():
            return []
        # extract info for current state and its path
        current_state, path = stateAgenda.pop()
        # add to visites set
        visited.add(current_state)
        # check if we have reached to the goal state
        # if so, return its path
        if problem.isGoalState(current_state):
            return path
        # otherwise get successors
        succStates = problem.getSuccessors(current_state)
        for state in succStates:
            # only if we havent visited this state again
            if state[0] not in visited:
                # construct new path and push state to agenda
                stateAgenda.push((state[0], path + [state[1]]))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    path = [] # Every state keeps it's path from the starting state
    visited = set() # Set data structure is the most appropriate to use in this case
    stateAgenda = Queue() # Keeping states into a queue in order to implement BFS

    # Check if initial state is goal states
    if problem.isGoalState(problem.getStartState()):
        return path
    # otherwise push initial state into the stack
    stateAgenda.push((problem.getStartState(), path))

    while(True):
        # terminate condition
        if stateAgenda.isEmpty():
            return []
        # extract info for current state and its path
        current_state, path = stateAgenda.pop()
        # add to visites set
        visited.add(current_state)
        # check if we have reached to the goal state
        # if so, return its path
        if problem.isGoalState(current_state):
            return path
        # otherwise get successors
        succStates = problem.getSuccessors(current_state)
        for state in succStates:
            # only if we havent visited this state again and this state is not into our stateAgenda
            if state[0] not in visited and state[0] not in (allStates[0] for allStates in stateAgenda.list):
                # construct new path and push state to agenda
                stateAgenda.push((state[0], path + [state[1]]))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    path = [] # Every state keeps it's path from the starting state
    visited = set() # Set data structure is the most appropriate to use in this case
    stateAgenda = PriorityQueue() # Keeping states into a PriorityQueuequeue in order to implement UCS

    # Check if initial state is goal states
    if problem.isGoalState(problem.getStartState()):
        return path
    # otherwise push initial state into the PriorityQueue with path-cost = 0
    stateAgenda.push((problem.getStartState(), path), 0)

    while(True):
        # terminate condition
        if stateAgenda.isEmpty():
            return []
        # extract info for current state and its path
        current_state, path = stateAgenda.pop()
        # check if we have reached to the goal state
        # add to visites set
        visited.add(current_state)
        # if so, return its path
        if problem.isGoalState(current_state):
            return path
        # otherwise get successors
        succStates = problem.getSuccessors(current_state)
        for state in succStates:
            # only if we havent visited this state again and this state is not into our stateAgenda
            if state[0] not in visited and state[0] not in (allStates[2][0] for allStates in stateAgenda.heap):
                # construct new path and push state to agenda
                new_path = path + [state[1]]
                # find new path cost
                state_priority = problem.getCostOfActions(new_path)
                # push new path and its cost into agenda
                stateAgenda.push((state[0], new_path), state_priority)
            # otherwise, if state is in frontier
            elif state[0] not in visited and state[0] in (allStates[2][0] for allStates in stateAgenda.heap):
                for child_state in stateAgenda.heap:
                    if child_state[2][0] == state[0]:
                        old_priority = problem.getCostOfActions(child_state[2][1])
                        break

                new_priority = problem.getCostOfActions(path + [state[1]])
                # and has higher Path-Cost
                if new_priority < old_priority:
                    stateAgenda.update((state[0], path + [state[1]]), new_priority)


def nullHeuristic(state, problem = None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic = nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
