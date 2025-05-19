import torch
from torch import nn, optim
import torch.nn.functional as F

import numpy as np

from collections import deque
import random

from cantstop_env.envs.cant_stop_utils import CantStopState, CantStopActionType, StopContinueChoice, ProgressActionSet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class StopContinueModel1(nn.Module):
    """
    Model that takes the raw saved/active steps remaining for each column.
    """
    def __init__(self, state_size, hidden_size=16):
        super().__init__()

        # Input layer contains:
        # - Saved steps remaining
        # - Active steps remaining
        self.l1 = nn.Linear(state_size[0], hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size // 2)
        self.l3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        return F.sigmoid(x)

class SelectDiceModel1(nn.Module):
    def __init__(self, hidden_size=55):
        super().__init__()
        self.l1 = nn.Linear(11 + 11 + 77, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size // 2)
        self.l3 = nn.Linear(hidden_size // 4, 77)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.device = DEVICE
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, event):
        self.memory.append(event)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        # Transpose list of lists
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        #print(actions)
        try:
            actions = torch.from_numpy(np.array(actions)).long().unsqueeze(1).to(self.device)
        except:
            print("actions", actions)
            print("error")
            quit()
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().unsqueeze(1).to(self.device)

        return states, next_states, actions, rewards, dones


class Agent:
    def __init__(self, state_size, action_size, learning_rate, replay_buffer_size):
        # First start with learned selection of stop/continue actions
        # and random dice combo selections.

        #self.state_size = state_size
        self.action_size = action_size

        self.local_stop_continue_qnetwork = StopContinueModel1(state_size).to(DEVICE)
        self.target_stop_continue_qnetwork = StopContinueModel1(state_size).to(DEVICE)

        self.local_select_dice_qnetwork = SelectDiceModel1().to(DEVICE)
        self.target_stop_continue_qnetwork = SelectDiceModel1().to(DEVICE)

        self.optimiser = optim.Adam(self.local_stop_continue_qnetwork.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0

    def step(self,
             state: CantStopState,
             action,
             reward,
             next_state: CantStopState,
             done,
             minibatch_size,
             discount_factor,
             interpolation_parameter):
        self.memory.push((state.to_np_embedding(),
                          action,
                          reward,
                          next_state.to_np_embedding(),
                          done))
        self.t_step = (self.t_step + 1) % 4

        if self.t_step == 0 and len(self.memory.memory) > minibatch_size:
            experiences = self.memory.sample(100)
            self.learn(experiences, discount_factor, interpolation_parameter)

    def act(self, state: CantStopState, epsilon):
        """
        Chooses an action to take, given a state.
        :param state:
        :param epsilon:
        :return:
        """
        #state = state.float().unsqueeze(0).to(DEVICE)
        inp_state = torch.from_numpy(state.to_np_embedding()).float().unsqueeze(0).to(DEVICE)
        if isinstance(state.current_action, StopContinueChoice):
            # get action the agent would take, without updating weights.
            self.local_stop_continue_qnetwork.eval()
            with torch.no_grad():
                action_values = self.local_stop_continue_qnetwork(inp_state)
            self.local_stop_continue_qnetwork.train()

            if np.random.sample() > epsilon:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return np.random.choice(np.arange(inp_state.shape[1]))
        elif isinstance(state.current_action, ProgressActionSet):
            self.local_select_dice_qnetwork.eval()
            with torch.no_grad():
                action_values = self.local_select_dice_qnetwork(inp_state)
            self.local_select_dice_qnetwork.train()
        else:
            raise ValueError(f"The state's current action ({state.current_action}) is not valid.")

        if np.random.sample() > epsilon:
            # Choose the action with the highest q value
            return np.argmax(action_values.cpu().data.numpy())
        # Choose a random action.
        return np.random.choice(np.arange(inp_state.shape[1]))


    def learn(self, experiences, discount_factor, interpolation_parameter):
        states, next_states, actions, rewards, dones = experiences
        next_q_targets = self.target_stop_continue_qnetwork(next_states).detach()#.max(1)[0].unsqueeze(1)

        q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
        q_expected = F.sigmoid(self.local_stop_continue_qnetwork(states))#.gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        self.soft_update(interpolation_parameter)

    def soft_update(self, interpolation_parameter):
        for target_param, local_param in zip(self.target_stop_continue_qnetwork.parameters(),
                                             self.local_stop_continue_qnetwork.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data +
                                    (1 - interpolation_parameter) * target_param.data)