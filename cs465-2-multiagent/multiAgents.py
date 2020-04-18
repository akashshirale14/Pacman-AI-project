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
        newFoodList = newFood.asList()
        """Closet distance to food"""
        min_food_distance=999999
        for food in newFoodList:
          current_food_distance=util.manhattanDistance(newPos,food)
          if(min_food_distance >=current_food_distance):
            min_food_distance=current_food_distance
        
        """Total distance to all ghosts from the sucessor position"""
        run_away_pacman=0
        total_ghost_distance=1
        for ghost in successorGameState.getGhostPositions():
          current_ghost_distance=util.manhattanDistance(newPos,ghost)
          if current_ghost_distance <=1:
            run_away_pacman+=1
          total_ghost_distance=total_ghost_distance+current_ghost_distance

        """Final evaluation which gets returned"""
        return successorGameState.getScore()+(1/float(min_food_distance))-3*(1/float(total_ghost_distance))-run_away_pacman
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
            **legal actions are directions.SOUTH,Directions.NORTH,etc....**
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          **at each level the agent takes action from the gamestate
          gameState.getNumAgents():
            Returns the total number of agents in the game
            for ex:returns 5
           ** number of agents means Pacman=0 and ghosts=1,2,3,4.**
        """
        def minimax(agentnum,depth,gameState):
          if gameState.isWin() or gameState.isLose() or self.depth==depth:
            return self.evaluationFunction(gameState)
          if agentnum > 0: #for ghost agents
            min_val=9999999
            nextagent=agentnum+1
            if gameState.getNumAgents()==nextagent:
              nextagent=0           #agentnum is set again to zero as depth is reached
              depth=depth+1
            for action in gameState.getLegalActions(agentnum):
              state1=gameState.generateSuccessor(agentnum,action)
              ans=minimax(nextagent,depth,state1)
              min_val= min(min_val,ans) #finding minimum of all its successors
            return min_val  
          if agentnum == 0:    #for pacman agent
            max_val=-9999999
            for action in gameState.getLegalActions(agentnum):
              state1=gameState.generateSuccessor(agentnum,action)
              ans=minimax(1,depth,state1)
              max_val=max(max_val,ans) #finding the max of all its successors
            return max_val  

        maxi=-999999
        for action in gameState.getLegalActions(0):
          state1=gameState.generateSuccessor(0,action)
          outcome=minimax(1,0,state1) #initial max node calls its successors
          if outcome > maxi:
            maxi=outcome
            final_action=action
        
        return final_action    



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def minimax(agentnum,depth,alpha,beta,gameState):
          if gameState.isWin() or gameState.isLose() or self.depth==depth:
            return self.evaluationFunction(gameState)
          if agentnum > 0:
            dummy_beta=beta
            min_val=9999999
            nextagent=agentnum+1
            if gameState.getNumAgents()==nextagent:
              nextagent=0
              depth=depth+1
            for action in gameState.getLegalActions(agentnum):
              state1=gameState.generateSuccessor(agentnum,action)
              ans=minimax(nextagent,depth,alpha,dummy_beta,state1)
              min_val= min(min_val,ans)
              dummy_beta=min(dummy_beta,ans)
              if alpha > dummy_beta:
                return min_val    
            return min_val  
          if agentnum == 0:
            dummy_alpha=alpha
            max_val=-9999999
            for action in gameState.getLegalActions(agentnum):
              state1=gameState.generateSuccessor(agentnum,action)
              ans=minimax(1,depth,alpha,dummy_beta,state1)
              max_val=max(max_val,ans)
              dummy_alpha=max(dummy_alpha,ans)
              if alpha > dummy_alpha:
                return max_val
            return max_val

        maxi=-999999
        for action in gameState.getLegalActions(0):
          state1=gameState.generateSuccessor(0,action)
          outcome=minimax(1,0,-999999,999999,state1)
          if outcome > maxi:
            maxi=outcome
            final_action=action    
        return final_action
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
        def expectimax(agentnum,depth,gameState):
          if gameState.isWin() or gameState.isLose() or self.depth==depth:
            return self.evaluationFunction(gameState)
          if agentnum > 0:  #for min nodes
            min_val=0.00
            counter=0
            nextagent=agentnum+1
            if gameState.getNumAgents()==nextagent:
              nextagent=0
              depth=depth+1
            for action in gameState.getLegalActions(agentnum):
              state1=gameState.generateSuccessor(agentnum,action)
              ans=expectimax(nextagent,depth,state1)
              counter+=1               #counting the number of states
              min_val= min_val + ans
            min_val=min_val/counter    #calculating the average of all the nodes
            return min_val  
          if agentnum == 0:   #for max nodes
            max_val=-9999999
            for action in gameState.getLegalActions(agentnum):
              state1=gameState.generateSuccessor(agentnum,action)
              ans=expectimax(1,depth,state1)
              max_val=max(max_val,ans)
            return max_val  

        maxi=-999999
        for action in gameState.getLegalActions(0):
          state1=gameState.generateSuccessor(0,action)
          outcome=expectimax(1,0,state1) #the max node calls its successors
          if outcome > maxi:
            maxi=outcome
            final_action=action
        
        return final_action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    proximity_ghost=0
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    near_ghost_distance=1  
    food_ans=1  
    near_capsule_distance=1
    #distance between nearest food and pacman current position
    min_food_distance=999999;
    for food in newFood.asList():
      if food==True:
        distance=util.manhattanDistance(newPos,food)
        if(min_food_distance > distance):
          min_food_distance=distance    
    if(min_food_distance > 0):
      food_ans=min_food_distance
        
    min_ghost_distance=999999;

    #distance to closest ghost and also the scared time of the respective ghost
    for ghostState in newGhostStates:
      distance=util.manhattanDistance(newPos,ghostState.getPosition()) + ghostState.scaredTimer
      if(min_ghost_distance > distance):
        min_ghost_distance=distance
    
    if(min_ghost_distance>0):
      near_ghost_distance=min_ghost_distance;

    #number of capsules remaining in the game at present
    number_of_capsules=len(currentGameState.getCapsules())
    newCapsules=currentGameState.getCapsules()
    #finding distance to the nearest capsule
    min_capsule_distance=999999
    for capsule in newCapsules:
      distance=util.manhattanDistance(newPos,capsule)
      if(min_capsule_distance > distance):
        min_capsule_distance=distance
    if(min_capsule_distance>0):
      near_capsule_distance=min_capsule_distance
    #final evaluation function which gets returned for the current state
    return currentGameState.getScore()+(1/food_ans)-9*(1/near_ghost_distance)-number_of_capsules+20*(1/near_capsule_distance)

# Abbreviation
better = betterEvaluationFunction

