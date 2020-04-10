import gym
from src.NeuralNetwork import NeuralNetwork


def main():
    env = gym.make('CartPole-v0')
    observation = env.reset()
    NN = NeuralNetwork(observation)
    NN.train(env)
    env.close()


if __name__ == "__main__":
    main()
