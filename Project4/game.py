import random
class ThreePuzzle:
    def __init__(self, probability=0.8, base_reward=1):
        self.start_state = [3,0,2,1]
        self.goal_state = [1,2,3,0]
        self.probability = probability
        self.base_reward = base_reward

    def get_blank_position(self, state):
        return state.index(0)

    def get_states(self):
        nums = [0,1,2,3]
        perms = [[]]   
        for n in nums:
            new_perms = []
            for perm in perms:
                for i in range(len(perm)+1):   
                    new_perms.append(perm[:i] + [n] + perm[i:])
            perms = new_perms
        return perms

    def get_actions(self, state):
        blank_position = self.get_blank_position(state)
        actions = []
        if(blank_position % 2 == 0):
            actions.append("right")
        else:
            actions.append("left")
        if(blank_position > 1):
            actions.append("up")
        else:
            actions.append("down")
        return actions

    def take_action(self, old_state, action, prob=None):
        if(prob is None):
            prob = self.probability
        
        state = old_state.copy()
        actions = self.get_actions(state)
        
        if(action not in actions):
            raise IOError("Invalid action")

        final_action = action
        if(random.random() > prob):
            actions.remove(action)
            final_action = actions[0]
        
        bp = self.get_blank_position(state)

        if(final_action == "right"):
            state[bp], state[bp+1] = state[bp+1], state[bp]

        elif(final_action == "left"):
            state[bp], state[bp-1] = state[bp-1], state[bp]
        elif(final_action == "up"):
            state[bp], state[bp-2] = state[bp-2], state[bp]
        elif(final_action == "down"):
            state[bp], state[bp+2] = state[bp+2], state[bp]
        else:
            raise IOError("Invalid action")
        return state


    def get_transition_states_probs(self, state, action):
        state1 = self.take_action(state, action, 1)
        state2 = self.take_action(state,action,0)
        return [[state1,self.probability],[state2,1-self.probability]]


    def get_reward(self, state):
        if(state == self.goal_state):
            return 100
        else:
            return self.base_reward
    def isTerminal(self,state):
        if(state == self.goal_state):
            return True
        else:
            return False


class FivePuzzle:
    def __init__(self, probability=0.8, base_reward=1):
        self.start_state = [4,5,0,3,1,2]
        self.goal_state = [1,2,3,4,5,0]
        self.probability = probability
        self.base_reward = base_reward

    def get_blank_position(self, state):
        return state.index(0)

    def get_states(self):
        nums = [0,1,2,3,4,5]
        perms = [[]]   
        for n in nums:
            new_perms = []
            for perm in perms:
                for i in range(len(perm)+1):   
                    new_perms.append(perm[:i] + [n] + perm[i:])
            perms = new_perms
        return perms

    def get_actions(self, state):
        blank_position = self.get_blank_position(state)
        actions = []
        if(blank_position % 3 == 0):
            actions.append("right")
        elif(blank_position % 3 == 1):
            actions.append("left")
            actions.append("right")
        else:
            actions.append("left")
        if(blank_position > 2):
            actions.append("up")
        else:
            actions.append("down")
        return actions

    def take_action(self, old_state, action, prob=None):
        if(prob is None):
            prob = self.probability
        
        state = old_state.copy()
        actions = self.get_actions(state)
        
        if(action not in actions):
            raise IOError("Invalid action")

        final_action = action
        val = random.random()
        if(len(actions) == 3):
            if(val > prob):
                actions.remove(action)
                if(val > (prob + 1)/2):
                    final_action = actions[1]
                else:
                    final_action = actions[0]
        elif(len(actions) == 2):
            if(val > prob):
                actions.remove(action)
                final_action = actions[0]
        
        bp = self.get_blank_position(state)
        if(final_action == "right"):
            state[bp], state[bp+1] = state[bp+1], state[bp]
        elif(final_action == "left"):
            state[bp], state[bp-1] = state[bp-1], state[bp]
        elif(final_action == "up"):
            state[bp], state[bp-3] = state[bp-3], state[bp]
        elif(final_action == "down"):
            state[bp], state[bp+3] = state[bp+3], state[bp]
        else:
            raise IOError("Invalid action")
        return state


    def get_transition_states_probs(self, state, action):
        actions = self.get_actions(state)
        if(len(actions) == 2):
            state1 = self.take_action(state, action, 1)
            state2 = self.take_action(state,action,0)
            return [[state1,self.probability],[state2,1-self.probability]]
        elif(len(actions) == 3):
            state1 = self.take_action(state, action, 1)
            actions.remove(action)
            state2 = self.take_action(state,actions[0],1)
            state3 = self.take_action(state, actions[1],1)
            return [[state1,self.probability],[state2,(1 - self.probability)/2],[state3,(1 - self.probability)/2]]


    def get_reward(self, state):
        if(state == self.goal_state):
            return 100
        else:
            return self.base_reward
    def isTerminal(self,state):
        if(state == self.goal_state):
            return True
        else:
            return False




    
    
