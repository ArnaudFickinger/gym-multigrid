import gym
import time
from gym.envs.registration import register

def main():

    register(
        id='multigrid-soccer-v0',
        entry_point='gym_multigrid.envs:SoccerGame4HEnv10x15N2',
    )
    env = gym.make('multigrid-soccer-v0')

    _ = env.reset()

    nb_agents = len(env.agents)

    while True:
        env.render(mode='human')
        time.sleep(0.1)

        ac = [env.action_space.sample() for _ in range(nb_agents)]

        _, _, done, _ = env.step(ac)

        if done:
            break

if __name__ == "__main__":
    main()