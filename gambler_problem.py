import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

matplotlib.use('Agg')

num_states = 101


class Gambler:
    def __init__(self, ph, theta=1e-16, discount=1.0):
        self.policy = None
        self.ph = ph
        self.theta = theta
        self.discount = discount
        self.reward = np.zeros(num_states)
        self.reward[-1] = 1
        self.reset()

    def reset(self):
        self.v_estimates = np.zeros(num_states)

    def bellman(self, state):
        if state == 0:
            return None
        else:
            actions = np.arange(1, np.min([state, num_states-state-1])+1)
            p_heads = self.ph
            p_tails = 1 - self.ph
            f = np.vectorize(lambda a: p_heads*(self.reward[state+a]+self.discount*self.v_estimates[state+a])
                                       + p_tails*(self.reward[state-a]+self.discount*self.v_estimates[state-a]))
            estimations = f(actions)
            return actions, estimations

    def value_iteration(self):
        delta = 1.0
        while delta > self.theta:
            delta = 0.0
            for s in range(1, num_states-1):
                _, estimations = self.bellman(s)
                v = self.v_estimates[s]
                self.v_estimates[s] = np.max(estimations)
                if delta < abs(v - self.v_estimates[s]):
                    delta = abs(v - self.v_estimates[s])

    def assign_policy_pretty(self):
        self.policy = np.zeros(num_states)
        for state in range(1, num_states-1):
            actions, estimates = self.bellman(state)
            self.policy[state] = actions[np.argmax(np.round(estimates, 5))]

    def save_policy(self):
        with open('gambler_policy_%s.bin' % (str(self.ph)), 'wb') as f:
            pickle.dump(self.policy, f)

    def load_policy(self):
        with open('gambler_policy_%s.bin' % (str(self.ph)), 'rb') as f:
            self.policy = pickle.load(f)


def figure_gambler_25(ph=0.25):
    gambler = Gambler(ph=ph)
    gambler.value_iteration()
    gambler.assign_policy_pretty()

    plt.scatter(np.arange(num_states), gambler.policy)
   #plt.plot(gambler.policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy(stake)')
    plt.legend()

    plt.savefig('figure_gambler_25.png')
    plt.close()


if __name__ == '__main__':
    figure_gambler_25()




