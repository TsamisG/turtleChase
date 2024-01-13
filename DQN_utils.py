import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQNetwork(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, learning_rate):
        super(DQNetwork, self).__init__()

        self.layers = nn.ModuleList()

        M1 = input_dims
        for M2 in hidden_dims:
            self.layers.append(nn.Linear(M1, M2))
            M1 = M2
        
        self.layers.append(nn.Linear(M1, output_dims))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    
    def forward(self, X):
        for layer in self.layers[:-1]:
            X = F.relu(layer(X))
        X = self.layers[-1](X)
        return X

class Memory:
    def __init__(self, state_dims, mem_size):
        self.mem_size = mem_size
        self.states = np.zeros((mem_size, state_dims))
        self.actions = np.zeros(mem_size)
        self.rewards = np.zeros(mem_size)
        self.new_states = np.zeros((mem_size, state_dims))
        self.dones = np.zeros(mem_size)
        self.running_idx = 0
        self.is_full = False
    
    def save_transition(self, s, a, r, s2, done):
        idx = self.running_idx % self.mem_size
        self.states[idx, :] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.new_states[idx, :] = s2
        self.dones[idx] = done
        self.running_idx = idx + 1
        if self.running_idx == self.mem_size and not self.is_full:
            self.is_full = True
        

    def sample_batch(self, batch_size):
        idx_range = self.mem_size if self.is_full else self.running_idx
        indices = np.random.permutation(idx_range)[:batch_size]
        states_batch = self.states[indices, :]
        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        new_states_batch = self.new_states[indices, :]
        dones_batch = self.dones[indices]
        return states_batch, actions_batch, rewards_batch, new_states_batch, dones_batch

class Agent:
    def __init__(self, gamma, state_dims, hidden_dims, n_actions, alpha, eps, mem_size, batch_size,
                    replace_target_every):
        self.gamma = gamma
        self.epsilon = eps
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.start_learning = False
        self.memory = Memory(state_dims, mem_size)
        self.DQN_eval = DQNetwork(state_dims, hidden_dims, n_actions, alpha)
        self.DQN_target = DQNetwork(state_dims, hidden_dims, n_actions, alpha)
        self.replace_target_every = replace_target_every
        self.learn_counter = 0

    def choose_action_eps_greedy(self, state):
        state = T.FloatTensor(state[None, :])
        Qvalues = self.DQN_eval(state)
        if np.random.random() > self.epsilon:
            action = T.argmax(Qvalues).item()
        else:
            action = np.random.randint(self.n_actions)
        return action
    
    def choose_action_deterministic(self, state):
        state = T.FloatTensor(state[None, :])
        Qvalues = self.DQN_eval(state)
        return T.argmax(Qvalues).item()
    
    def learn(self):
        if self.memory.running_idx == self.batch_size and not self.start_learning:
            self.start_learning = True
        if not self.start_learning:
            return
        states, actions, rewards, new_states, dones = self.memory.sample_batch(self.batch_size)
        states = T.FloatTensor(states)
        new_states = T.FloatTensor(new_states)
        rewards = T.FloatTensor(rewards)
        dones = T.ShortTensor(dones)

        Qvalues = self.DQN_eval(states)[np.arange(actions.shape[0]), actions]
        new_Qvalues = self.DQN_target(new_states) * (1 - dones[:, None])
        targets = rewards + self.gamma*T.max(new_Qvalues, dim=1).values
        self.DQN_eval.zero_grad()
        loss = nn.MSELoss()
        J = loss(targets, Qvalues)
        J.backward()
        self.DQN_eval.optimizer.step()
        self.learn_counter += 1

    def update_target(self):
        if self.learn_counter % self.replace_target_every == 0:
            self.DQN_target.load_state_dict(self.DQN_eval.state_dict())
        
