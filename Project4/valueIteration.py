import util
import random
from plot_graph import plot_nodes
import time
import matplotlib.pyplot as plt
import numpy as np

class ValueIterationAgent():

    def __init__(self, mdp, discount = 0.8, iterations = 100, mode = 'markers+text'):

        self.mdp = mdp #MDP Problem
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() #Values of each state
        self.edges = {} #Dictionary to save state->next_state transitions
        self.mode = mode
        random.seed(0)

        #Initialize the values arbitrarily. Value of terminal state = 0
        for state in self.mdp.get_states():
            if(not self.mdp.isTerminal(state)):
                self.values[tuple(state)] = random.random()*5000
            self.edges[tuple(state)] = set()
            for action in self.mdp.get_actions(state):
                next_state = self.mdp.take_action(state,action,1)
                self.edges[tuple(state)].add(tuple(next_state))

        start = time.time()
        time_vec = [0]
        time_wasted  = 0
        for i in range(self.iterations):
            # print(i)
            change = False
            if(i%5 == 0):
                next_states = {}
            for state in self.mdp.get_states():
                
                possibleActions = self.mdp.get_actions(state) 
                
                valuesForActions = util.Counter()
                if(possibleActions and (not self.mdp.isTerminal(state))):
                    for action in possibleActions:

                        #Get next state, probability for each state and possible action
                        transitionStatesAndProbs = self.mdp.get_transition_states_probs(state, action)
                        valueState = 0
                        for transition in transitionStatesAndProbs:
                            valueState += transition[1] * (self.mdp.get_reward(transition[0]) + self.discount * self.values[tuple(transition[0])])
                            #save state -> next_state transition
                            
                            
                        valuesForActions[action] = valueState
                    old = self.values[tuple(state)]
                    self.values[tuple(state)] = round(valuesForActions[valuesForActions.argMax()],3)
                    if(old != self.values[tuple(state)]):
                        change = True
                if(i%5 == 0):
                    waste_start = time.time()
                    next_state = self.getNextState(state)
                    if(next_state is not None):
                        next_states[tuple(state)] = next_state
                    time_wasted += (time.time() - waste_start) 
                    

            time_vec.append(time.time() - start - time_wasted)
            if(i % 5 == 0):
                if(mode == 'markers+text'):
                    plot_nodes(self.values.copy(), self.edges.copy(), "Value Iteration for small grid: Iteration =" + str(i), next_states.copy(),self.mode)
                else:
                    plot_nodes(self.values.copy(), self.edges.copy(), "Value Iteration for large grid: Iteration =" + str(i), next_states.copy(),self.mode)

            if(change == False):
                end = time.time()
                print("Value iteration converged in " + str(end-start) + " seconds")
                print("Value iteration converged in " + str(i) + " iterations")
                
                next_states = {}
                for state in mdp.get_states():
                    if(self.getNextState(state) is not None):
                        next_states[tuple(state)] = self.getNextState(state) 
                
                state = self.mdp.start_state
                optimal_solution = [state]
                total_reward = self.mdp.get_reward(state)
                while(state != tuple(self.mdp.goal_state)):
                    state = next_states[tuple(state)]
                    optimal_solution.append(state)
                    total_reward += self.mdp.get_reward(list(state))
                print("Optimal path: ", optimal_solution)
                print("Total reward: ",total_reward)

                break
        if(mode == 'markers+text'):
            plot_nodes(self.values, self.edges, "Value Iteration for small grid (Converged)", next_states, self.mode)
        else:
            plot_nodes(self.values, self.edges, "Value Iteration for large grid (Converged)", next_states, self.mode)
        
        

        plt.figure()
        if(mode == 'markers+text'):
            plt.title("Value iteration for small grid: Time vs iterations")
        else:
            plt.title("Value iteration for large grid: Time vs iterations")

        plt.xlabel("Iterations")
        plt.ylabel("Time(s)")
        plt.plot(np.arange(len(time_vec)),time_vec)

    def getValue(self, state):
        return self.values[state]

    def getPolicy(self, state):  
        if self.mdp.isTerminal(state):
            return None

        possibleActions = self.mdp.get_actions(state)

        valuesForActions = util.Counter()
        if(possibleActions):
            for action in possibleActions:
                transitionStatesAndProbs = self.mdp.get_transition_states_probs(state, action)
                valueState = 0
                for transition in transitionStatesAndProbs:
                    valueState += transition[1] * (self.mdp.get_reward(transition[0]) + self.discount * self.values[tuple(transition[0])])
                valuesForActions[action] = valueState

        if valuesForActions.totalCount() == 0:
            return possibleActions[int(random.random() * len(possibleActions))]
        else:
            valueToReturn = valuesForActions.argMax()
            return valueToReturn

    def getNextState(self, state):
        action = self.getPolicy(state)
        if(action is None):
            return None
        return tuple(self.mdp.take_action(state,action,1))



class PolicyIterationAgent():

    def __init__(self, mdp, discount = 0.8, iterations = 100, mode = 'markers+text'):

        self.mdp = mdp #MDP Problem
        self.discount = discount
        self.iterations = iterations
        self.policy = {}
        self.values = util.Counter() #Values of each state
        self.edges = {} #Dictionary to save state->next_state transitions
        self.mode = mode
        random.seed(0)

        #Initialize the values arbitrarily. Value of terminal state = 0
        for state in self.mdp.get_states():
            if(not self.mdp.isTerminal(state)):
                self.values[tuple(state)] = random.random()*5000
            self.policy[tuple(state)] = self.mdp.get_actions(state)[0]
            self.edges[tuple(state)] = set()
            for action in self.mdp.get_actions(state):
                next_state = self.mdp.take_action(state,action,1)
                self.edges[tuple(state)].add(tuple(next_state))

        

        start = time.time()
        time_vec = [0]

        step = 0
        while(True):

            next_states = {} 
            
            #Policy evaluation
            for i in range(self.iterations):
                # print(i)
                change = False
                
                for state in self.mdp.get_states():
                    if(not self.mdp.isTerminal(state)):
                        action = self.policy[tuple(state)]
                        
                        #Get next state, probability for each state and possible action
                        transitionStatesAndProbs = self.mdp.get_transition_states_probs(state, action)
                        valueState = 0
                        for transition in transitionStatesAndProbs:
                            valueState += transition[1] * (self.mdp.get_reward(transition[0]) + self.discount * self.values[tuple(transition[0])])

                        old = self.values[tuple(state)]
                        self.values[tuple(state)] = round(valueState,3)
                        if(old != self.values[tuple(state)]):
                            change = True
                
                if(change == False):
                    break


            #Policy Improvement
            policy_stable = True
            for state in self.mdp.get_states():
                if(not self.mdp.isTerminal(state)):
                    old_action = self.policy[tuple(state)]
                    valuesForActions = util.Counter()
                    
                    for action in self.mdp.get_actions(state):
                        #Get next state, probability for each state and possible action
                        transitionStatesAndProbs = self.mdp.get_transition_states_probs(state, action)
                        valueState = 0
                        for transition in transitionStatesAndProbs:
                            valueState += transition[1] * (self.mdp.get_reward(transition[0]) + self.discount * self.values[tuple(transition[0])])
                            #save state -> next_state transition
                            
                        valuesForActions[action] = valueState
                        


                    self.policy[tuple(state)] = valuesForActions.argMax()
                    
                    next_state = self.mdp.take_action(state,self.policy[tuple(state)],1)
                    if(next_state is not None):
                        next_states[tuple(state)] = tuple(next_state)

                    if(old_action != self.policy[tuple(state)]):
                        policy_stable = False
            
            if(mode == 'markers+text'):
                plot_nodes(self.values.copy(), self.edges.copy(), "Policy Iteration for small grid: Iteration =" + str(step), next_states.copy(), self.mode)
            else:
                plot_nodes(self.values.copy(), self.edges.copy(), "Policy Iteration for large grid: Iteration =" + str(step), next_states.copy(), self.mode)


            time_vec.append(time.time() - start)
            step += 1
            if(policy_stable):
                end = time.time()
                print("Policy iteration converged in " + str(end-start) + " seconds")
                print("Policy iteration converged in " + str(step) + " iterations")
                next_states = {}
                for state in mdp.get_states():
                    if(not self.mdp.isTerminal(state)):
                        next_state = self.mdp.take_action(state,self.policy[tuple(state)],1)
                        if(next_state is not None):
                            next_states[tuple(state)] = tuple(next_state)
                
                state = self.mdp.start_state
                
                optimal_solution = [state]
                total_reward = self.mdp.get_reward(state)
                while(state != tuple(self.mdp.goal_state)):
                    state = next_states[tuple(state)]
                    optimal_solution.append(state)
                    total_reward += self.mdp.get_reward(list(state))
                print("Optimal path: ", optimal_solution)
                print("Total reward: ",total_reward) 

                break
        if(mode == 'markers+text'):
            plot_nodes(self.values.copy(), self.edges.copy(), "Policy Iteration for small grid (Converged)", next_states.copy(), self.mode)
        else:
            plot_nodes(self.values.copy(), self.edges.copy(), "Policy Iteration for large grid (Converged)", next_states.copy(), self.mode)

        
        plt.figure()
        if(mode == 'markers+text'):
            plt.title("Policy iteration for small grid: Time vs iterations")
        else:
            plt.title("Policy iteration for large grid: Time vs iterations")

        plt.xlabel("Iterations")
        plt.ylabel("Time(s)")
        plt.plot(np.arange(len(time_vec)),time_vec)

    def getValue(self, state):
        return self.values[state]



class QLearningAgent():
    def __init__(self, mdp, discount = 0.8, iterations = 10000, epsilon = 0.1, alpha = 0.1, mode = 'markers+text'):

        self.mdp = mdp #MDP Problem
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() #Values of each state
        self.edges = {} #Dictionary to save state->next_state transitions
        self.alpha = alpha
        self.epsilon = epsilon
        self.mode = mode
        random.seed(0)
        self.Qvalues = util.Counter()
        #Initialize the values arbitrarily. Value of terminal state = 0
        for state in self.mdp.get_states():  
            self.edges[tuple(state)] = set()
            for action in self.mdp.get_actions(state):
                if(not self.mdp.isTerminal(state)):
                    self.Qvalues[tuple(state),action] = random.random()*5000
                next_state = self.mdp.take_action(state,action,1)
                self.edges[tuple(state)].add(tuple(next_state))

        num_states = len(self.mdp.get_states())


        state_counter = 0
        start = time.time()
        time_vec = [0]
        for i in range(self.iterations):
            # print(i)
            change = False
            if(i% (self.iterations//10) == 0):
                next_states = {}
            
            state = self.mdp.get_states()[state_counter]
            step = 0
            while(not self.mdp.isTerminal(state) and step < 100):

                action = self.getAction(state)
                next_state = self.mdp.take_action(state,action)
                reward = self.mdp.get_reward(next_state)
                old = self.Qvalues[(tuple(state),action)]
                self.update(state, action, next_state, reward)
                if(old != self.Qvalues[(tuple(state),action)]):
                    change = True
                state = next_state
                # print(state, end = " ")
                
                step += 1
            # print()
            state_counter = (state_counter + 1) % num_states 

            if(i% (self.iterations//10) == 0):
                for state in mdp.get_states():
                    action = self.computeActionFromQValues(state)
                    next_state = self.mdp.take_action(state, action, 1)
                    self.values[tuple(state)] = self.Qvalues[tuple(state),action]
                    if(next_state is not None and not self.mdp.isTerminal(state)):
                        next_states[tuple(state)] = tuple(next_state) 
             
            
                if(mode == 'markers+text'):
                    plot_nodes(self.values.copy(), self.edges.copy(), "Q-Learning for small grid: Iteration =" + str(i), next_states.copy(), self.mode)
                else:
                    plot_nodes(self.values.copy(), self.edges.copy(), "Q-Learning for large grid: Iteration =" + str(i), next_states.copy(), self.mode)
            time_vec.append(time.time() - start)

        
        end = time.time()
        print("Q-Learning converged in " + str(end-start) + " seconds")
        print("Q-Learning converged in " + str(i) + " iterations")
        # if(mode == 'markers+text'):
        #     state = self.mdp.start_state
        #     optimal_solution = [state]
        #     total_reward = self.mdp.get_reward(state)
        #     while(state != tuple(self.mdp.goal_state)):
        #         state = next_states[tuple(state)]
        #         optimal_solution.append(state)
        #         total_reward += self.mdp.get_reward(list(state))
        #     print("Optimal path: ", optimal_solution)
        #     print("Total reward: ",total_reward)

        if(mode == 'markers+text'):
            plot_nodes(self.values.copy(), self.edges.copy(), "Q-Learning for small grid (Converged)", next_states.copy(), self.mode)
        else:
            plot_nodes(self.values.copy(), self.edges.copy(), "Q-Learning for large grid (Converged)", next_states.copy(), self.mode)
        
        plt.figure()
        if(mode == 'markers+text'):
            plt.title("Q-Learning for small grid: Time vs iterations")
        else:
            plt.title("Q-Learning for large grid: Time vs iterations")

        plt.xlabel("Iterations")
        plt.ylabel("Time(s)")
        plt.plot(np.arange(len(time_vec)),time_vec)





    def getQValue(self, state, action):
        return self.Qvalues[(tuple(state), action)]


    def computeValueFromQValues(self, state):
        actions = self.mdp.get_actions(state)
        if actions:
          Qvalues = []
          for action in actions:
            Qvalues.append(self.getQValue(state, action))
          return max(Qvals)
        else:
          return 0.0

    def computeActionFromQValues(self, state):
        bestVal = float('-inf')
        bestAction = None
        for action in self.mdp.get_actions(state):
          val = self.getQValue(state, action)
          if val > bestVal:
            bestVal = val
            bestAction = action
        return bestAction

    def getAction(self, state):
        legalActions = self.mdp.get_actions(state)
        action = None

        if legalActions:
            r = random.random()
        if (r < self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        
        successorQVals = []    
        for a in self.mdp.get_actions(nextState):
          successorQVals.append(self.getQValue(nextState, a))
        if successorQVals:
          highestQVal = max(successorQVals)
        else:
          highestQVal = 0
        newVal = (1-self.alpha)*self.Qvalues[(tuple(state), action)]+self.alpha*(reward+self.discount*highestQVal)
        self.Qvalues[(tuple(state), action)] = round(newVal,3)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
