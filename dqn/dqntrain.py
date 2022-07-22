from dqn.agent import Agent
from snake_game.snake_game import SnakeGame
from utils.utils import create_directory


class DqnTrain:
    def __init__(self, steps, save_rate, number_hide_layers, number_neural_by_layer, buffer_size, batch_size,
                 epsilon_decay, gamma, replace):
        self.env = SnakeGame(display=False)
        self.steps = steps
        self.save_rate = save_rate
        self.path_save = './dqn_model'
        self.agent = Agent(self.env, number_hide_layers, number_neural_by_layer, self.path_save, buffer_size, batch_size,
                           epsilon_decay, gamma, replace)

    def train(self):
        create_directory(self.path_save)
        for step_index in range(1, self.steps + 1):
            done = False
            state = self.env.reset()
            total_reward = 0
            move = 0
            while not done:
                action = self.agent.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.update_memory(state, action, reward, next_state, done)
                self.agent.train()
                state = next_state
                total_reward += reward
                move += 1
                print('> Moves realised : {}, chosen action {}, reward {}, total reward {}'.format(move, action, reward,
                                                                                                   total_reward),
                      flush=True)
                if done:
                    print("Total reward : {}, step : {}, epsilon value : {}".format(total_reward, step_index,
                                                                                    self.agent.epsilon), flush=True)
            if step_index % self.save_rate == 0 and step_index != 0:
                self.agent.save_model(step_index)
