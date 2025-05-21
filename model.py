import torch
from torch import nn, optim
import torch.nn.functional as F

import numpy as np

from collections import deque
import random

from cantstop_env.envs.cant_stop_utils import CantStopState, CantStopActionChoice, StopContinueChoice, ProgressActionChoice, \
    CantStopAction, ProgressAction, StopContinueAction

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
    def __init__(self, hidden_size=88):
        super().__init__()
        self.l1 = nn.Linear(11 + 11 + 77, hidden_size)
        self.l2 = nn.Linear(hidden_size, 80)
        self.l3 = nn.Linear(80, 77)

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
        self.target_select_dice_qnetwork = SelectDiceModel1().to(DEVICE)

        self.stop_continue_optimiser = optim.Adam(self.local_stop_continue_qnetwork.parameters(), lr=learning_rate)
        self.select_dice_optimiser = optim.Adam(self.local_select_dice_qnetwork.parameters(), lr=learning_rate)

        self.stop_continue_memory = ReplayMemory(replay_buffer_size)
        self.select_dice_memory = ReplayMemory(replay_buffer_size)

        self.t_step = 0

    def step(self,
             state: CantStopState,
             action: CantStopAction,
             reward,
             next_state: CantStopState,
             done,
             minibatch_size,
             discount_factor,
             interpolation_parameter):
        self.stop_continue_memory.push((state.to_np_embedding(),
                                        action,
                                        reward,
                                        next_state.to_np_embedding(),
                                        done))
        self.t_step = (self.t_step + 1) % 4

        if self.t_step != 0:
            return

        if len(self.stop_continue_memory.memory) > minibatch_size:
            experiences = self.stop_continue_memory.sample(minibatch_size)
            self.learn_stop_continue(experiences, discount_factor, interpolation_parameter)

        if len(self.select_dice_memory.memory) > minibatch_size:
            experiences = self.select_dice_memory.sample(minibatch_size)
            self.learn_select_dice(experiences, discount_factor, interpolation_parameter)

    def select_stop_continue_action(self, state, epsilon):
        inp_state = torch.from_numpy(state.to_np_embedding()).float().unsqueeze(0).to(DEVICE)

        self.local_stop_continue_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_stop_continue_qnetwork(inp_state)
        self.local_stop_continue_qnetwork.train()

        if np.random.sample() > epsilon:
            action_id = np.argmax(action_values.cpu().data.numpy())
        else:
            print(inp_state.shape)
            action_id = np.random.choice(2)
        return [StopContinueAction.STOP, StopContinueAction.CONTINUE][action_id]

    def select_dice_action(self, state, epsilon):
        inp_state = torch.from_numpy(state.to_np_embedding()).float().unsqueeze(0).to(DEVICE)

        self.local_select_dice_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_select_dice_qnetwork(inp_state)
        self.local_select_dice_qnetwork.train()

        # TODO: implement mask to only allow valid actions to be outputted.
        if np.random.sample() > epsilon:
            # Choose the action with the highest q value
            action_id = np.argmax(action_values.cpu().data.numpy())
        else:
            # Choose a random action.
            action_id = np.random.choice(77)
        return ProgressAction.decode(action_id)

    def learn_stop_continue(self, experiences, discount_factor, interpolation_parameter):
        states, next_states, actions, rewards, dones = experiences
        next_q_targets = self.target_stop_continue_qnetwork(next_states).detach()#.max(1)[0].unsqueeze(1)

        q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
        q_expected = F.sigmoid(self.local_stop_continue_qnetwork(states))#.gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)

        self.stop_continue_optimiser.zero_grad()
        loss.backward()
        self.stop_continue_optimiser.step()
        self.stop_continue_soft_update(interpolation_parameter)

    def learn_select_dice(self, experiences, discount_factor, interpolation_parameter):
        states, next_states, actions, rewards, dones = experiences
        next_q_targets = self.target_select_dice_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)

        q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
        q_expected = F.sigmoid(self.local_select_dice_qnetwork(states))#.gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)


        self.select_dice_optimiser.zero_grad()
        loss.backward()
        self.select_dice_optimiser.step()
        self.select_dice_soft_update(interpolation_parameter)

    def stop_continue_soft_update(self, interpolation_parameter):
        for target_param, local_param in zip(self.target_stop_continue_qnetwork.parameters(),
                                             self.local_stop_continue_qnetwork.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data +
                                    (1 - interpolation_parameter) * target_param.data)

    def select_dice_soft_update(self, interpolation_parameter):
        for target_param, local_param in zip(self.target_select_dice_qnetwork.parameters(),
                                             self.local_select_dice_qnetwork.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data +
                                    (1 - interpolation_parameter) * target_param.data)