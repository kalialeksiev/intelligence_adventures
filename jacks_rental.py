import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math


MAX_CARS = 6
RENT_REWARD = 10
PARKING_COST = 4
TRANSFER_COST = 2
MAX_CARS_OVERNIGHT_FREE = 3
SHUTTLE_CARS_FREE = 1
# maximum possible request for renting or returning as
# at any time a maximum of 20 cars can be kept at a location
# also the probability of a request that is greater than this is negligible
MAX_REQUEST = MAX_CARS

poisson_cache = dict()


def poisson(k, lmbd):
    if (k, lmbd) in poisson_cache:
        return poisson_cache[(k, lmbd)]
    else:
        poisson_cache[(k, lmbd)] = (np.exp(-lmbd)*np.power(lmbd, k)) / math.factorial(k)
        return poisson_cache[(k, lmbd)]


class Jack:
    def __init__(self, lambda1_rent=4, lambda1_return=3, lambda2_rent=4, lambda2_return=2, discount=0.9, theta=1e-7):
        self.lambda1_rent = lambda1_rent
        self.lambda1_return = lambda1_return
        self.lambda2_rent = lambda2_rent
        self.lambda2_return = lambda2_return
        self.discount = discount
        self.theta = theta
        self.reset()

    def reset(self):
        self.v = np.zeros([MAX_CARS+1]*2)
        self.policy = np.zeros([MAX_CARS+1]*2, int)

    def bellman_update(self, s1, s2, a):
        bellman_sum = 0.0
        overnight_cost = 0.0
        if a > 0:
            overnight_cost += TRANSFER_COST*(a-1)
        else:
            overnight_cost += TRANSFER_COST*a
        if s1-a > MAX_CARS_OVERNIGHT_FREE:
            overnight_cost += PARKING_COST
        if s2+a > MAX_CARS_OVERNIGHT_FREE:
            overnight_cost += PARKING_COST
        # loop through the 4 possible poisson random variables
        # giving a state change
        for rent_req1 in range(MAX_REQUEST+1):
            for rent_req2 in range(MAX_REQUEST+1):
                # new state after cars for request are taken
                # we implement it in such a way that if we request more
                # than the current number of cars available, the number of cars at that location just becomes 0
                s1_prime = np.max([0, s1 - rent_req1])
                s2_prime = np.max([0, s2 - rent_req2])

                for return_req1 in range(MAX_REQUEST+1):
                    for return_req2 in range(MAX_REQUEST+1):
                        # same as above, just we truncate at maximum possible number of cars, i.e. 20
                        s1_prime = np.min([MAX_CARS, s1_prime+return_req1])
                        s2_prime = np.min([MAX_CARS, s2_prime+return_req2])
                        reward = RENT_REWARD*(rent_req1 + rent_req2) - overnight_cost
                        transition_prob = poisson(rent_req1, self.lambda1_rent) * poisson(rent_req2, self.lambda2_rent) \
                                          * poisson(return_req1, self.lambda1_return) * poisson(return_req2, self.lambda2_return)
                        bellman_sum += transition_prob * (reward + self.discount*self.v[s1_prime][s2_prime])
        return bellman_sum

    def policy_evaluation(self):
        delta = 1
        while delta > self.theta:
            delta = 0
            for s1 in range(len(self.v[0])):
                for s2 in range(len(self.v[0])):
                    v = self.v[s1][s2]
                    self.v[s1][s2] = self.bellman_update(s1, s2, self.policy[s1][s2])
                    delta = np.max([delta, np.abs(v-self.v[s1][s2])])

    def policy_improvement(self):
        change = False
        for s1 in range(len(self.v[0])):
            for s2 in range(len(self.v[0])):
                actions = np.arange(-s2, s1+1)
                action_values = []
                for a in actions:
                    action_values.append(self.bellman_update(s1, s2, a))
                best = actions[np.argmax(action_values)]
                if self.policy[s1][s2] != best:
                    change = True
                    self.policy[s1][s2] = best
        return change

    def policy_iteration(self):
        change = True
        while change:
            self.policy_evaluation()
            change = self.policy_improvement()

    def plot(self):
        plt.figure()
        plt.xlim(0, MAX_CARS + 1)
        plt.ylim(0, MAX_CARS + 1)
        plt.table(cellText=self.policy, loc=(0, 0), cellLoc='center')
        plt.show()


if __name__ == '__main__':
    jack = Jack()
    jack.policy_iteration()
    jack.plot()