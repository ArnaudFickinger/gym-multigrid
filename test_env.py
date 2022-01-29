import gym
import time
from gym.envs.registration import register
import argparse

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='soccer', type=str)

args = parser.parse_args()

def main():

    if args.env == 'soccer':
        register(
            id='multigrid-soccer-v0',
            entry_point='gym_multigrid.envs:SoccerGameEnv',
            kwargs={'size': None, 'height': 10, 'width': 15, 'goal_pst': [[1,5], [13,5]], 'goal_index': [1,2], 'num_balls': [1], 
            'agents_index': [1,1,2,2], 'balls_index': [0], 'zero_sum': True}
        )
        env = gym.make('multigrid-soccer-v0')

    else:
        register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGameEnv',
            kwargs={'size': 10,'num_balls': [5], 'agents_index': [1,2,3], 'balls_index': [0], 'balls_reward': [1], 'zero_sum': True, 'partial_obs': False},

        )
        env = gym.make('multigrid-collect-v0')

    _ = env.reset()

    nb_agents = len(env.agents)

    while True:
        env.render(mode='human', highlight=True if env.partial_obs else False)
        time.sleep(0.1)

        ac = [env.action_space.sample() for _ in range(nb_agents)]

        obs, _, done, _ = env.step(ac)

        if done:
            break

if __name__ == "__main__":
    main()