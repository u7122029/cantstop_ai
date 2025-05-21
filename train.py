from collections import deque

import gymnasium as gym
import numpy as np
import torch as T
from tqdm import tqdm

from cantstop_env.envs.cant_stop_utils import CantStopState, StopContinueAction, StopContinueChoice, \
    ProgressActionChoice
from model import Agent


def one_hot_encode(size, idx):
    out = T.zeros(size)
    out[idx] = 1
    return out

def main():
    #env = gym.make("LunarLander-v3")
    env_name = "cantstop_env/CantStop-v0"
    env = gym.make(env_name)
    #number_actions = env.action_space.n
    state_size = (11 + 11,)
    learning_rate = 1e-3
    minibatch_size = 100
    discount_factor = 0.99
    replay_buffer_size = int(1e5)
    interpolation_parameter = 1e-3

    agent = Agent(state_size, 2, learning_rate, replay_buffer_size)

    number_episodes = 5000
    maximum_number_timesteps_per_episode = 1000

    epsilon_starting_value = 1.0
    epsilon_ending_value = 0.01
    epsilon_decay_value = 0.995
    epsilon = epsilon_starting_value
    scores_on_100_episodes = deque(maxlen = 100)

    episode_progress = tqdm(range(1, number_episodes + 1))
    postfix = {}

    action_choices = [StopContinueAction.STOP, StopContinueAction.CONTINUE]

    for episode in episode_progress:
        env.close()
        if episode > 100 and episode % 200 == 0:
            env = gym.make(env_name, render_mode="human")
        else:
            env = gym.make(env_name, render_mode=None)
        x = env.reset()
        state: CantStopState = x[0]

        score = 0
        success = False

        for t in range(maximum_number_timesteps_per_episode):
            if isinstance(state.current_action, StopContinueChoice):
                action = agent.select_stop_continue_action(state, epsilon)
            elif isinstance(state.current_action, ProgressActionChoice):
                action = agent.select_dice_action(state, epsilon)
            else:
                raise Exception("invalid")

            next_state, reward, done, _, _ = env.step(action)

            score += reward
            if done:
                break

            agent.step(state,
                       action,
                       reward,
                       next_state,
                       done,
                       minibatch_size,
                       discount_factor,
                       interpolation_parameter)
            state = next_state.copy()

        scores_on_100_episodes.append(score)

        epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
        postfix["episode"] = episode
        postfix["avg_score"] = np.mean(scores_on_100_episodes)
        postfix["episode_success"] = success
        episode_progress.set_postfix(postfix)

        if postfix["avg_score"] >= 200.0:
            print(f"Solved in {episode} episodes with average score {np.mean(scores_on_100_episodes)}.")
            T.save(agent.local_stop_continue_qnetwork.state_dict(), "checkpoint.pt")
            break

if __name__ == "__main__":
    main()

