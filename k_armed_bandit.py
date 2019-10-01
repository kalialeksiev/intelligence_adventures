import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

matplotlib.use('Agg')


class Bandit:

    def __init__(self, k=10, epsilon=0.1, static=True, true_reward=0.0,  gradient_bandit=False,
                 average=False, td=False, gradient_baseline=True, ucb_param=None,
                 step_size=None, initials=0.):
        # parameters relating to the specification of the problem
        self.k = k
        self.indices = np.arange(self.k)
        self.epsilon = epsilon
        self.static = static
        self.true_reward = true_reward
        # parameters relating to the desired update
        self.gradient_bandit = gradient_bandit
        self.average = average
        self.td = td
        # parameters relating to the desired algorithm:
        self.gradient_baseline = gradient_baseline
        self.ucb_param = ucb_param
        self.step_size = step_size
        self.initials = initials

        self.reset()

    def reset(self):
        # relating to the environment
        self.time = 0
        self.true_q_values = np.random.randn(self.k) + self.true_reward
        best = np.max(self.true_q_values)
        self.best_actions = np.where(self.true_q_values == best)[0]
        # relating to the requirements for updating an action(based on the algorithm used)
        if self.average:
            self.action_count = np.zeros(self.k)
            self.q_estimates = np.zeros(self.k) + self.initials
        elif self.gradient_bandit:
            self.action_preferences = np.zeros(self.k)
            probabilities = np.exp(self.action_preferences)
            self.action_probabilities = probabilities / sum(probabilities)
            if self.gradient_baseline:
                self.action_averages = np.zeros(self.k)
                self.action_count = np.zeros(self.k)
        else:
            self.q_estimates = np.zeros(self.k) + self.initials

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)
        elif self.ucb_param is not None:
            ucb_estimates = self.q_estimates + \
                self.ucb_param * np.sqrt(np.log(self.time+1)/(self.action_count+1e-4))
            q_best = np.max(ucb_estimates)
            return np.random.choice(np.where(ucb_estimates == q_best)[0])
        elif self.gradient_bandit:
            return np.random.choice(self.indices, p=self.action_probabilities)
        else:
            q_best = np.max(self.q_estimates)
            return np.random.choice(np.where(self.q_estimates == q_best)[0])

    def step(self, action):
        # generate a random number ~ N(true_q_values(action), 1)
        action_return = np.random.rand() + self.true_q_values[action]
        # different thing according to the different update procedures
        if self.average:
            self.action_count[action] += 1
            self.q_estimates[action] = (self.q_estimates[action] *
                                        (self.action_count[action] - 1) + action_return)/self.action_count[action]
        if self.td:
            self.q_estimates[action] += self.step_size*(self.q_estimates[action] - action_return)
        if self.gradient_bandit:
            for a in self.indices:
                preference = self.action_preferences[a]
                probability = self.action_probabilities[a]
                baseline = 0.0
                if self.gradient_baseline:
                    baseline = self.action_averages[a]
                if a == action:
                    self.action_preferences[a] = preference + self.step_size*(action_return-baseline)*(1-probability)
                else:
                    self.action_preferences[a] = preference - self.step_size*(action_return-baseline)*probability
            probabilities = np.exp(self.action_preferences)
            self.action_probabilities = probabilities / sum(probabilities)
            if self.gradient_baseline:
                self.action_count[action] += 1
                self.action_averages = (self.action_averages *
                                        (self.action_count[action]-1)+action_return)/self.action_count[action]
        if not self.static:
            for s in self.indices:
                self.true_q_values += np.random.normal(loc=0, scale=0.01)
            best = np.max(self.true_q_values)
            self.best_actions = np.where(self.true_q_values == best)[0]
        return action_return


def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)

    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.choose_action()
                if action in bandit.best_actions:
                    best_action_counts[i][r][t] = 1
                rewards[i][r][t] = bandit.step(action)
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards


def figure_2_1():
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig('figure_2_1.png')
    plt.close()


def figure_2_2(runs=2000, time=1000):
    epsilons = [0, 0.1, 0.01]
    bandits = [Bandit(epsilon=eps, average=True) for eps in epsilons]
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='epsilon = %.02f' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='epsilon = %.02f' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('figure_2_2.png')
    plt.close()


def figure_2_3(runs=2000, time=1000):
    bandits = []
    bandits.append(Bandit(epsilon=0, initials=5., td=True, step_size=0.1))
    bandits.append(Bandit(epsilon=0.1, initials=0., td=True, step_size=0.1))
    best_action_counts, _ = simulate(runs, time, bandits)

    plt.plot(best_action_counts[0], label='epsilon = 0, q = 5')
    plt.plot(best_action_counts[1], label='epsilon = 0.1, q = 0')
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('figure_2_3.png')
    plt.close()


def figure_2_4(runs=2000, time=1000):
    bandits = []
    bandits.append(Bandit(epsilon=0, ucb_param=2, average=True))
    bandits.append(Bandit(epsilon=0.1, average=True))
    _, average_rewards = simulate(runs, time, bandits)

    plt.plot(average_rewards[0], label='UCB c = 2')
    plt.plot(average_rewards[1], label='epsilon greedy epsilon = 0.1')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('figure_2_4.png')
    plt.close()


def figure_2_5(runs=2000, time=1000):
    bandits = []
    bandits.append(Bandit(gradient_bandit=True, step_size=0.1, gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient_bandit=True, step_size=0.1, gradient_baseline=False, true_reward=4))
    bandits.append(Bandit(gradient_bandit=True, step_size=0.4, gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient_bandit=True, step_size=0.4, gradient_baseline=False, true_reward=4))
    best_action_counts, _ = simulate(runs, time, bandits)
    labels = ['alpha = 0.1, with baseline',
              'alpha = 0.1, without baseline',
              'alpha = 0.4, with baseline',
              'alpha = 0.4, without baseline']

    for i in range(len(bandits)):
        plt.plot(best_action_counts[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    plt.savefig('figure_2_5.png')
    plt.close()

def figure_exercise_2_5(runs=100, time=10000):
    bandits = []
    bandits.append(Bandit(epsilon=0.1, initials=0., average=True, static=False))
    bandits.append(Bandit(epsilon=0.1, initials=0., td=True, step_size=0.1, static=False))
    best_action_counts, _ = simulate(runs, time, bandits)

    plt.plot(best_action_counts[0], label='epsilon = 0.1, average')
    plt.plot(best_action_counts[1], label='epsilon = 0.1, step_size=0.1, td style')
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('figure_exercise_2_5.png')
    plt.close()


def figure_2_6(runs=2000, time=1000):
    labels = ['epsilon-greedy', 'gradient bandit',
              'UCB', 'optimistic initialization']
    generators = [lambda epsilon: Bandit(epsilon=epsilon, average=True),
                  lambda alpha: Bandit(gradient_bandit=True, step_size=alpha, gradient_baseline=True),
                  lambda coef: Bandit(epsilon=0, ucb_param=coef, average=True),
                  lambda initial: Bandit(epsilon=0, initials=initial, td=True, step_size=0.1)]
    parameters = [np.arange(-7, -1, dtype=np.float),
                  np.arange(-5, 2, dtype=np.float),
                  np.arange(-4, 3, dtype=np.float),
                  np.arange(-2, 3, dtype=np.float)]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    _, average_rewards = simulate(runs, time, bandits)
    rewards = np.mean(average_rewards, axis=1)

    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i:i+l], label=label)
        i += l
    plt.xlabel('Parameter(2^x)')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('figure_2_6.png')
    plt.close()


if __name__ == '__main__':
   # figure_2_1()
   # figure_2_2()
   # figure_2_3()
   # figure_2_4()
   # figure_2_5()
    figure_2_6()
    figure_exercise_2_5()











