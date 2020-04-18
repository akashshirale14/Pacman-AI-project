pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        "*** YOUR CODE HERE ***"
        mindistance=999999
        for action in legal:
            successorPosition=Actions.getSuccessor(position,action)
            for eachghost in livingGhostPositionDistributions:
                maxProbability=-999999
                for k in eachghost.keys():
                    if(eachghost[k]> maxProbability):
                        maxProbability=eachghost[k]
                        gosPosition=k
            distance=self.distancer.getDistance(successorPosition,gosPosition)
            if(distance<mindistance):
                mindistance=distance
                bestaction=action

        return bestaction





        closestGhostPos = None
        closestDistance = float("inf")
        for ghostDistribution in livingGhostPositionDistributions:
            mostLikelyPosition = ghostDistribution.argMax()
            distance = self.distancer.getDistance(mostLikelyPosition, pacmanPosition)
            if distance < closestDistance:
                closestGhostPos = mostLikelyPosition
                closestDistance = distance

        minAction = None
        minDistance = float("inf")
        for action in legal:
            successorPosition = Actions.getSuccessor(pacmanPosition, action)
            distance = self.distancer.getDistance(successorPosition, closestGhostPos)
            if distance < minDistance:
                minAction = action
                minDistance = distance

        return minAction


nd=0
        temp_particles=[]
        if noisyDistance==None:
            while nd< self.numParticles:
                temp_particles.append(self.getJailPosition())
            self.particles=temp_particles    
        else:
            weights=util.Counter()
            for p in self.particles:
                trueDistance=util.manhattanDistance(pacmanPosition,p)
                weights[p]+=emissionModel[trueDistance]

            if weights.totalCount()==0:
                self.initializeUniformly(gameState) 

            else:
                weights.normalize()
                for i in range(0,self.numParticles):
                    temp_particles.append(util.sample(weights))
                self.particles=temp_particles








        