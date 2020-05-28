import gym
import pybullet_envs
import numpy as np
from finalmodel import FinalModel


def main():
    env = gym.make('AntBulletEnv-v0')
    env.render()
    model = FinalModel(env)
    avg = 0
    model.load()
    for i in range(100):
        score = playgame(model, env)
        avg += score
        print(f"run {i + 1} total reward: {score}")
    avg /= 100
    if avg >= 2350:
        real_score = 70
    elif avg >= 2000:
        real_score = 60
    elif avg >= 1000:
        real_score = 50
    elif avg >= 500:
        real_score = 40
    print(f"Average score is: {avg}")
    print(f"FINAL SCORE IS: {real_score}")


def playgame(model, env):
    s = env.reset()
    reward_sum = 0

    while True:
        env.render()

        log_policy, action = model.actor.get_action(s)
        next_s, reward, done, _ = env.step(np.clip(action, env.action_space.low, env.action_space.high))

        s = next_s

        reward_sum += reward
        if done:
            break
    return reward_sum


if __name__ == "__main__":
    main()