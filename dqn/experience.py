import numpy as np


class Experience:
    def __init__(self, buffer_size, state_size, action_size):
        self.buffer_size = buffer_size
        self.pointer = 0
        self.state_memory = np.zeros((self.buffer_size, state_size), dtype=np.int32)
        self.action_memory = np.zeros((self.buffer_size, action_size), dtype=np.int32)
        self.next_state_memory = np.zeros((self.buffer_size, state_size), dtype=np.int32)
        self.reward_memory = np.zeros(self.buffer_size, dtype=np.int32)
        self.done_memory = np.zeros(self.buffer_size, dtype=np.bool)

    def add_one_experience(self, state, action, reward, next_state, done):
        index = self.pointer % self.buffer_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index] = done
        self.pointer += 1

    def sample_of_experiences(self, batch_size):
        max_memory = min(self.pointer, self.buffer_size)
        batch_selected = np.random.choice(max_memory, batch_size, replace=False)
        state = self.state_memory[batch_selected]
        action = self.action_memory[batch_selected]
        reward = self.reward_memory[batch_selected]
        next_state = self.next_state_memory[batch_selected]
        done = self.done_memory[batch_selected]
        return state, action, reward, next_state, done
