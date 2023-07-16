""""""
"""  		  	   		  		 			  		 			 	 	 		 		 	
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		  		 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			 	 	 		 		 	
or edited.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Student Name: Devansh Panirwala (replace with your name)  		  	   		  		 			  		 			 	 	 		 		 	
GT User ID: dpanirwala3 (replace with your User ID)  		  	   		  		 			  		 			 	 	 		 		 	
GT ID: 903262441 (replace with your GT ID)  		  	   		  		 			  		 			 	 	 		 		 	
"""




import random as rand
import numpy as np
class QLearner(object):
    def author(self):
        return "dpanirwala3"

    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.9,
        rar=0.98,
        radr=0.999,
        dyna=0,
        verbose=False,
    ):
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.exprArray = []
        self.qTable = np.zeros([num_states, num_actions])

    def querysetstate(self, s):
        self.s = s
        return rand.randint(0, self.num_actions-1) if rand.uniform(0, 1) < self.rar else np.argmax(self.qTable[s])

    def query(self, s_prime, r):
        state = self.s
        action = self.a
        self.qTable[self.s, self.a] = (
            1-self.alpha)*self.qTable[state, action] + self.alpha * (r + self.gamma * np.max(self.qTable[s_prime]))

        prime_act = self.querysetstate(s_prime)
        self.rar = self.rar * self.radr

        self.exprArray.append([state, action, s_prime, r])

        if self.dyna > 0:
            y = 0
            for i in np.random.choice(len(self.exprArray), size=self.dyna):
                self.qTable[self.exprArray[i][0], self.exprArray[i][1]] = (
                    1 - self.alpha) * self.qTable[self.exprArray[i][0], self.exprArray[i][1]] + self.alpha * (self.exprArray[i][3] + self.gamma * np.max(self.qTable[self.exprArray[i][2]]))
        else:
            pass

        self.s = s_prime
        self.a = prime_act

        if self.verbose:
            print("state =", s_prime, "action =", action, "reward =", r)

        return prime_act


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
