import math
import random
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import gym
from gym import spaces
import pygame


class SnakeGame(gym.Env):
    """
    Snake snake_game class
    """

    LIST_DIRECTION = ['Up', 'Right', 'Down', 'Left']
    ALL_ACTIONS = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    SIZE_PIXEL_CELL = 20

    def __init__(self, size_field=18, time_speed=0.1, display=True):
        self.size_field = size_field
        self.size_field_large = self.size_field + 2
        self.tick = int(1 / time_speed)
        self.snake_coordinates = []
        self.food_coordinate = None
        self.direction = 'Up'
        self.state_game = True
        self.state_size = spaces.Discrete(11)
        self.action_space = spaces.Discrete(len(self.ALL_ACTIONS))
        self.info = 'Snake Game in python'
        self.display = display

        pygame.init()
        if self.display:
            pygame.display.set_caption('Snake Game')
            self.window = pygame.display.set_mode(
                (self.size_field_large * self.SIZE_PIXEL_CELL, self.size_field_large * self.SIZE_PIXEL_CELL))
        self.clock = pygame.time.Clock()

    def play_game(self):
        self.init_game()
        self.render()
        while self.state_game:
            self.move_by_keyboard()
            self.generate_food()
            self.render()
        print('GAME OVER')
        print('--- Score : %d ---' % len(self.snake_coordinates))

    def init_game(self):
        x_snake_head = int(self.size_field / 2)
        y_snake_head = int(self.size_field / 2)
        self.snake_coordinates.extend([Point(x_snake_head - 2, y_snake_head), Point(x_snake_head - 1, y_snake_head),
                                       Point(x_snake_head, y_snake_head)])
        self.generate_food()

    def generate_food(self):
        if self.food_coordinate is None:
            x_food = random.randrange(0, self.size_field, 1)
            y_food = random.randrange(0, self.size_field, 1)
            food_position = Point(x_food, y_food)
            while self.check_list(self.snake_coordinates, food_position):
                x_food = random.randrange(0, self.size_field, 1)
                y_food = random.randrange(0, self.size_field, 1)
                food_position = Point(x_food, y_food)
            self.food_coordinate = food_position

    def check_list(self, list_of_coordinate, position):
        for coordinate in list_of_coordinate:
            if coordinate.x == position.x and coordinate.y == position.y:
                return True
        return False

    def move_by_keyboard(self):
        self.get_keyboard()
        if self.direction == 'Up':
            self.update_position(Point(self.snake_coordinates[0].x - 1, self.snake_coordinates[0].y))
        elif self.direction == 'Down':
            self.update_position(Point(self.snake_coordinates[0].x + 1, self.snake_coordinates[0].y))
        elif self.direction == 'Left':
            self.update_position(Point(self.snake_coordinates[0].x, self.snake_coordinates[0].y - 1))
        elif self.direction == 'Right':
            self.update_position(Point(self.snake_coordinates[0].x, self.snake_coordinates[0].y + 1))

    def get_keyboard(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = 'Left'
                elif event.key == pygame.K_RIGHT:
                    self.direction = 'Right'
                elif event.key == pygame.K_UP:
                    self.direction = 'Up'
                elif event.key == pygame.K_DOWN:
                    self.direction = 'Down'

    def update_position(self, new_snake_head):
        self.state_game = self.end_of_game(new_snake_head)
        if self.state_game:
            if self.food_coordinate == new_snake_head:
                self.snake_coordinates.insert(0, new_snake_head)
                self.food_coordinate = None
            else:
                self.snake_coordinates.insert(0, new_snake_head)
                self.snake_coordinates.pop()

    def end_of_game(self, new_snake_head):
        if 0 <= new_snake_head.x < self.size_field and 0 <= new_snake_head.y < self.size_field and not self.check_list(
                self.snake_coordinates, new_snake_head):
            return True
        return False

    def render(self, debug=False):
        grid = np.zeros((self.size_field_large, self.size_field_large))
        grid[0] = 1
        grid[self.size_field_large - 1] = 1
        for x in range(self.size_field):
            grid[1 + x][0] = 1
            grid[1 + x][self.size_field_large - 1] = 1
            for y in range(self.size_field):
                point = Point(x, y)
                if point == self.snake_coordinates[0]:
                    grid[1 + point.x][1 + point.y] = 4
                elif self.check_list(self.snake_coordinates[1:], point):
                    grid[1 + point.x][1 + point.y] = 3
                elif point == self.food_coordinate:
                    grid[1 + point.x][1 + point.y] = 2

        if debug:
            print(grid)

        if self.display:
            color = (0, 0, 0)
            self.window.fill(color)
            for x in range(self.size_field_large):
                for y in range(self.size_field_large):
                    if grid[x][y] == 1:
                        color = (191, 191, 191)
                    elif grid[x][y] == 2:
                        color = (255, 0, 0)
                    elif grid[x][y] == 3:
                        color = (0, 255, 0)
                    elif grid[x][y] == 4:
                        color = (255, 255, 0)
                    if grid[x][y] != 0:
                        pygame.draw.rect(self.window, color, pygame.Rect(y * self.SIZE_PIXEL_CELL, x * self.SIZE_PIXEL_CELL,
                                                                         self.SIZE_PIXEL_CELL, self.SIZE_PIXEL_CELL))
            pygame.display.flip()
        self.clock.tick(self.tick)

        if self.display:
            return pygame.surfarray.array3d(pygame.display.get_surface())

    def step(self, action, display=False):
        if display:
            pygame.event.pump()
        reward = self.reward(action)
        self.move(action)
        self.generate_food()
        observation = self.observation()
        done = self.state_game
        return observation, reward, not done, self.info

    def move(self, action):
        index = self.LIST_DIRECTION.index(self.direction)
        if action == [1, 0, 0]:  # LEFT TURN
            new_index = (index - 1) % len(self.LIST_DIRECTION)
        elif action == [0, 0, 1]:  # RIGHT TURN
            new_index = (index + 1) % len(self.LIST_DIRECTION)
        else:  # SAME DIRECTION
            new_index = index
        self.direction = self.LIST_DIRECTION[new_index]

        if self.direction == 'Up':
            self.update_position(Point(self.snake_coordinates[0].x - 1, self.snake_coordinates[0].y))
        elif self.direction == 'Down':
            self.update_position(Point(self.snake_coordinates[0].x + 1, self.snake_coordinates[0].y))
        elif self.direction == 'Left':
            self.update_position(Point(self.snake_coordinates[0].x, self.snake_coordinates[0].y - 1))
        elif self.direction == 'Right':
            self.update_position(Point(self.snake_coordinates[0].x, self.snake_coordinates[0].y + 1))

    def reward(self, action):
        index = self.LIST_DIRECTION.index(self.direction)
        if action == [1, 0, 0]:  # LEFT TURN
            new_index = (index - 1) % len(self.LIST_DIRECTION)
        elif action == [0, 0, 1]:  # RIGHT TURN
            new_index = (index + 1) % len(self.LIST_DIRECTION)
        else:  # SAME DIRECTION
            new_index = index
        next_direction = self.LIST_DIRECTION[new_index]

        if next_direction == 'Up':
            return self.calcul_reward(Point(self.snake_coordinates[0].x - 1, self.snake_coordinates[0].y))
        elif next_direction == 'Down':
            return self.calcul_reward(Point(self.snake_coordinates[0].x + 1, self.snake_coordinates[0].y))
        elif next_direction == 'Left':
            return self.calcul_reward(Point(self.snake_coordinates[0].x, self.snake_coordinates[0].y - 1))
        elif next_direction == 'Right':
            return self.calcul_reward(Point(self.snake_coordinates[0].x, self.snake_coordinates[0].y + 1))

    def calcul_reward(self, new_snake_head):
        if new_snake_head.x < 0 or new_snake_head.x >= self.size_field or new_snake_head.y < 0 or new_snake_head.y >= self.size_field or self.check_list(
                self.snake_coordinates, new_snake_head):
            return -100
        elif self.food_coordinate == new_snake_head:
            return 20
        elif self.euclidian_distance(self.food_coordinate, self.snake_coordinates[0]) < self.euclidian_distance(
                self.food_coordinate, new_snake_head):
            return -1
        else:
            return 1

    def euclidian_distance(self, point_a, point_b):
        return math.sqrt((point_a.x - point_b.x) ** 2 + (point_a.y - point_b.y) ** 2)

    def observation(self):
        point_left = Point(self.snake_coordinates[0].x, self.snake_coordinates[0].y - 1)
        point_right = Point(self.snake_coordinates[0].x, self.snake_coordinates[0].y + 1)
        point_up = Point(self.snake_coordinates[0].x - 1, self.snake_coordinates[0].y)
        point_down = Point(self.snake_coordinates[0].x + 1, self.snake_coordinates[0].y)

        dir_left = self.direction == 'Left'
        dir_right = self.direction == 'Right'
        dir_up = self.direction == 'Up'
        dir_down = self.direction == 'Down'

        straight = ((dir_up and self.is_collision(point_up)) or
                    (dir_down and self.is_collision(point_down)) or
                    (dir_left and self.is_collision(point_left)) or
                    (dir_right and self.is_collision(point_right)))

        right = ((dir_up and self.is_collision(point_right)) or
                 (dir_down and self.is_collision(point_left)) or
                 (dir_left and self.is_collision(point_up)) or
                 (dir_right and self.is_collision(point_down)))

        left = ((dir_up and self.is_collision(point_left)) or
                (dir_down and self.is_collision(point_right)) or
                (dir_left and self.is_collision(point_down)) or
                (dir_right and self.is_collision(point_up)))

        return [
            int(straight),
            int(right),
            int(left),
            int(self.direction == 'Left'),
            int(self.direction == 'Right'),
            int(self.direction == 'Up'),
            int(self.direction == 'Down'),
            int(self.snake_coordinates[0].y < self.food_coordinate.y),
            int(self.snake_coordinates[0].y > self.food_coordinate.y),
            int(self.snake_coordinates[0].x < self.food_coordinate.x),
            int(self.snake_coordinates[0].x > self.food_coordinate.x)
        ]

    def is_collision(self, point):
        if point.x >= self.size_field or point.x < 0 or point.y >= self.size_field or point.y < 0:
            return True
        if point in self.snake_coordinates[1:]:
            return True
        return False

    def reset(self):
        self.snake_coordinates = []
        self.food_coordinate = None
        self.direction = 'Up'
        self.state_game = True
        self.init_game()
        return self.observation()

    def get_random_action(self):
        return self.ALL_ACTIONS[np.random.choice(len(self.ALL_ACTIONS))]

    def debug(self):
        self.init_game()
        self.render(True)
        while self.state_game:
            index_old_direction = self.LIST_DIRECTION.index(self.direction)
            old_direction = self.direction

            self.get_keyboard()

            index_new_direction = self.LIST_DIRECTION.index(self.direction)
            self.direction = old_direction

            if index_old_direction == index_new_direction:
                action = [0, 1, 0]
            elif (index_old_direction + 1) % 4 == index_new_direction:
                action = [0, 0, 1]
            elif (index_old_direction - 1) % 4 == index_new_direction:
                action = [1, 0, 0]
            else:
                action = [0, 1, 0]

            reward = self.reward(action)
            self.move(action)
            self.generate_food()
            observation = self.observation()
            done = self.state_game

            print('---------------------')
            print(observation)
            print(reward)
            print(done)
            self.render(True)

        print('GAME OVER')
        print('--- Score : %d ---' % len(self.snake_coordinates))


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, point):
        return self.x == point.x and self.y == point.y
