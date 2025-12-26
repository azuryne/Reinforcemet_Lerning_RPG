import numpy as np 
import gym 
from gym import spaces
import pygame
import sys

class MMOBossEnv(gym.Env):
    """
    Boss Environment for MMO RPG
    """

    def __init__(self, render_mode=None):
        super(MMOBossEnv, self).__init__()

        # Game parameters 
        self.GAME_DURATION = 30.0  # 30 seconds for training session 
        self.TIME_STEP = 0.1 # 100 milliseconds pass in the simulated game world with each step
        self.ARENA_SIZE = (10, 10) # (x, y)

        # Human Parameter 
        self.HUMAN_INITIAL_HP = 1.0 # Initial full HP  
        self.HUMAN_MOVE_SPEED = 0.5 # 0.5 units per 0.1 time step  (== 0.05 units per time step)
        self.HUMAN_JUMP_DURATION = 1.0 # takes 1s to jump

        # Boss Parameter
        self.BOSS_LASER_DURAION = 3.0 # Boss can shoot for 3 seconds
        self.BOSS_LASER_COOLDOWN = 3.0 # Cooldown between shots
        self.BOSS_MOVE_SPEED = 0.3 # 0.3 units per 0.1 time step

        # Action Space (Discrete)
        self.action_space = spaces.Discrete(7) # Human actions = 7 actions 
        
        # Observation Space : [human_x, human_y, human_hp, human_jumping, boss_x, boss_y, boss_attacking, laser_timer, laser_cooldown, time_remaining]

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        ) # Define what RL agent receives at each step

        # Initialize game state
        self.reset()

        # For rendering 
        self.render_mode = render_mode
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((600,600))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)