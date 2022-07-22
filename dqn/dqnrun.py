import numpy as np
from tensorflow.keras.models import load_model
from dqn import model
from snake_game import SnakeGame

import cv2

from utils.utils import create_directory


class DqnRun:
    def __init__(self, model_name):
        self.env = SnakeGame()
        path_save = './dqn_model'
        self.model = load_model(path_save + '/' + model_name, custom_objects={'Model': model.Model})

    def run(self, save=False, file_name='output'):
        done = False
        state = self.env.reset()

        if save:
            path = 'video/'
            size = self.env.size_field_large * self.env.SIZE_PIXEL_CELL
            fps = self.env.tick
            create_directory(path)
            out = cv2.VideoWriter(path + file_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size, size))

        while not done:
            state = np.array([state])
            action_nn = self.model.predict(state)
            action = [0, 0, 0]
            action[np.argmax(action_nn)] = 1
            state, _, done, _ = self.env.step(action, True)
            image = self.env.render()
            if save:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                out.write(image)
        if save:
            out.release()



