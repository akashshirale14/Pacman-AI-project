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
    from util import Stack
    Fringe=Stack()
    visited=[]
    actions=[]
    newPath=[]
    flag=0
    state=problem.getStartState()
    index=(problem.getStartState(),[])
    Fringe.push(index)
    while not Fringe.isEmpty():
        if problem.isGoalState(state):
            break;
        if state not in visited:
            flag=1
            visited.append(state)
        if flag==1:
            for successor,newActions,stepCost in problem.getSuccessors(state): #getSuccessors returns 3 List -> successor(which gives the new state)
                newPath=actions + [newActions] #newPath stores direction
                index=(successor,newPath)
                Fringe.push(index)
        flag=0        
        state,actions=Fringe.pop()
        newPath=actions        
    return newPath 
    

def breadthFirstSearch(problem): #The difference in DFS and BFS implementation is that BFS uses Queue
    from util import Queue
    Fringe=Queue()
    visited=[]
    actions=[]
    newPath=[]
    flag=0
    state=problem.getStartState()
    index=(problem.getStartState(),[])
    Fringe.push(index)
    while not problem.isGoalState(state):
        if state not in visited:
            flag=1
            visited.append(state)
        if flag==1:
            for successor,newActions,stepCost in problem.getSuccessors(state):
                newPath=actions + [newActions]
                index=(successor,newPath)
                Fringe.push(index)
        flag=0        
        state,actions=Fringe.pop()
        newPath=actions        
    return newPath 
                    

def uniformCostSearch(problem):
    from util import PriorityQueue
    Fringe=PriorityQueue() #UCS uses Priority Queue and has a heuristic value of zero
    visited=[]
    actions=[]
    pathCost=0
    newPath=[]
    Route=PriorityQueue()
    Route.push(actions,0) #Route stores the route 
    state=problem.getStartState()
    index=[state,[],0]
    Fringe.push(state,0)
    while not problem.isGoalState(state):
        if state not in visited:
            flag=1
            visited.append(state)
        if flag==1:    
            for successor,newActions,stepCost in problem.getSuccessors(state):
                newPath=actions + [newActions]
                pathCost=problem.getCostOfActions(newPath)
                index=[successor,newPath,pathCost]
                Fringe.push(index[0],index[2]) # Fridge pushes successor and pathCost
                Route.push(index[1],index[2]) #Route pushes newPath and pathCost
        flag=0        
        state=Fringe.pop()
        actions=Route.pop()        
    return actions 
    
    """Search the node of least total cost first."""
    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueue
    Fringe=PriorityQueue()  #aStar uses Priority Queue and has heuristic value
    visited=[]
    actions=[]
    pathCost=0
    newPath=[]
    Route=PriorityQueue()
    state=problem.getStartState()
    Route.push(actions,heuristic(state,problem))
    index=[state,[],0]
    Fringe.push(state,heuristic(state,problem))
    while not problem.isGoalState(state):
        if state not in visited:
            visited.append(state)
            for successor,newActions,stepCost in problem.getSuccessors(state):
                newPath=actions + [newActions]
                pathCost=problem.getCostOfActions(newPath) + heuristic(successor,problem) #Adds forward cost and backward cost
                index=[successor,newPath,pathCost]
                Fringe.push(index[0],index[2])
                Route.push(index[1],index[2])
        state=Fringe.pop()
        actions=Route.pop()        
    return actions 


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
