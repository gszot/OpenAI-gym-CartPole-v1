import numpy as np


class NeuralNetwork:
    def __init__(self, observations, learning_rate=0.00001):
        inputs = len(observations)
        np.random.seed(1)
        self.weights = np.random.randn(inputs)/np.sqrt(inputs)
        print(self.weights)
        # position,velocticy,angle, tip velocity
        #self.weights = np.array([-0.3, 0.9, 0.03, 1.04]) <- 200/200 score
        self.X = observations
        self.Y = 200
        self.learning_rate = learning_rate

    def train(self, env):
        result = 0
        for i in range(100000):
            result = simulation(env, 100, lambda x: func(x, self.weights))
            if i == 1:
                print(result)
            error = result - self.Y
            gradient = 2*self.X*error
            self.weights -= self.learning_rate*gradient
            print(self.weights, result, i)
            #print(i)
        print(self.weights, result)


"""class NNtorch:
    def __init__(self):
        dtype = torch.float
        device = torch.device("cpu")
        N, D_in, H, D_out = 64, 1000, 100, 10
        x = torch.randn(N, D_in, device=device, dtype=dtype)
        y = torch.randn(N, D_out, device=device, dtype=dtype)
        print(type(x), x)"""


def agent(observation):
    if observation[3] > 0:
        return 1
    else:
        return 0


def simulation(env, trials, function):
    # position,velocticy,angle, tip velocity
    reward_sum = 0
    for i_episode in range(1, trials + 1):
        observation = env.reset()
        for t in range(200):
            #env.render()
            #print(observation)
            # action = env.action_space.sample()
            action = function(observation)
            observation, reward, done, info = env.step(action)
            if done:
                #print("Episode {} finished after {} timesteps".format(i_episode, t + 1))
                reward_sum += (t + 1)
                break
    #print("Average: {}".format(reward_sum/trials))
    return reward_sum/trials


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def nonlin(x):
    return x*(1-x)


def func(x, weights):
    l1 = np.dot(x, weights)
    l2 = (np.sign(l1) + 1) / 2
    return int(l2)