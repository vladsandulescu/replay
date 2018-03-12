from lib.envs.bitflip import BitFlipEnv
import numpy as np

class HER:
    n = 15
    np.random.seed(42)
    goal = np.random.choice([0, 1], size=(n, ))
    actions = np.arange(n)

    env = BitFlipEnv(n, goal)
    S = env.reset()
    done = False
    while not done:
        action = np.random.choice(actions)
        S_prime, reward, done, _ = env.step(action)
        print(S_prime, reward)
    print('Goal:', goal)

her = HER()