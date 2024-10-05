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
from game import Directions
from typing import List

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




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()
    start = [problem.getStartState(), 0, [problem.getStartState()], []]
    stack.push(start)
    while stack.isEmpty() != 1:
        now_point, now_cost, now_path, now_towards = stack.pop()
        if problem.isGoalState(now_point):
            return now_towards
        successors = problem.getSuccessors(now_point)
        for successor in successors:
            next_point, next_towards, next_cost = successor
            if (next_point not in now_path):
                cost = now_cost + next_cost
                path = now_path + [next_point]
                towards = now_towards + [next_towards]
                stack.push([next_point, cost, path, towards])
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    start = [problem.getStartState(), 0, [problem.getStartState()], []]
    queue.push(start)
    is_done = [start]
    while queue.isEmpty() != 1:
        now_point, now_cost, now_path, now_towards = queue.pop()
        if problem.isGoalState(now_point):
            return now_towards
        successors = problem.getSuccessors(now_point)
        for successor in successors:
            next_point, next_towards, next_cost = successor
            if (next_point not in now_path) and (next_point not in is_done):
                cost = now_cost + next_cost
                path = now_path + [next_point]
                towards = now_towards + [next_towards]
                queue.push([next_point, cost, path, towards])
                is_done.append(next_point)
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    pq = util.PriorityQueue()
    start = (problem.getStartState(), 0, [], [])
    pq.push(start, 0)
    dis = {problem.getStartState(): 0}
    while pq.isEmpty() != 1:
        now_point, now_cost, now_path, now_towards = pq.pop()
        if problem.isGoalState(now_point):
            return now_towards
        for successor in problem.getSuccessors(now_point):
            next_point, next_towards, next_cost = successor
            cost = now_cost + next_cost
            path = now_path + [next_point]
            towards = now_towards + [next_towards]
            if next_point not in dis or dis[next_point] > cost:
                dis[next_point] = cost
                pq.push((next_point, cost, path, towards), cost)
    util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    pq = util.PriorityQueue()
    start = (problem.getStartState(), 0, [], [])
    pq.push(start, heuristic(problem.getStartState(), problem))
    dis = {problem.getStartState(): 0}
    while pq.isEmpty() != 1:
        now_point, now_cost, now_path, now_towards = pq.pop()
        if problem.isGoalState(now_point):
            return now_towards
        for successor in problem.getSuccessors(now_point):
            next_point, next_towards, next_cost = successor
            cost = now_cost + next_cost
            path = now_path + [next_point]
            towards = now_towards + [next_towards]
            if next_point not in dis or dis[next_point] > cost:
                dis[next_point] = cost
                pq.push((next_point, cost, path, towards), cost + heuristic(next_point, problem))
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
