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
        """
        Initialize the environment
        """
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
    
    def reset(self):
        """
        Reset the environment to initial state
        """

        # Human state 
        self.human_pos = np.array([1.0,1.0], dtype=np.float32)
        self.human_hp = self.HUMAN_INITIAL_HP
        self.human_jummping = False
        self.jump_tier = 0.0

        # Boss state 
        self.boss_pos = np.array([5.0, 5.0], dtype=np.float32)
        self.boss_direction = np.array([1.0, 0.0], dtype=np.float32)
        self.boss_attacking = False
        self.laser_timer = 0.0
        self.laser_cooldown = 0.0 

        # Game state
        self.time_played = 0.0 
        self.game_over = False
        self.winner = None 

        return self._get_obs()

    def _get_obs(self):
        """
        Convert state to the observation vector
        """
        return np.array([
            self.human_pos[0] / self.ARENA_SIZE[0], # Normalize x to [0, 1]
            self.human_pos[1] / self.ARENA_SIZE[1], # Normalize y to [0, 1]
            self.human_hp, 
            float(self.human_jumping),
            self.laser_timer / self.BOSS_LASER_DURAION, # Normalize laser timer to [0, 1]
            self.laser_cooldown / self.BOSS_LASER_COOLDOWN, # Normalize laser cooldown to [0, 1]
            ])

    def _get_boss_action(self):
        """
        Simple AI for boss to move and attack 
        TODO: Implement RL agent for boss
        """

        # Chase Human 
        direction = self.human_pos - self.boss_pos 
        distance = np.linalg.norm(direction) # CalcuLate Eucladian distance between boss and human 

        if distance > 0: 
            direction = direction / distance # Normalize direction into vector - ensure boss moves at exactly BOSS_MOVE_SPEED (0.3 units per 0.1 time step)

        # Move towards human - This logic ensure boss moves only one direction at a time (horizontal or vertical)
        move = np.array([0.0, 0.0])
        if abs(direction[0]) > abs(direction[1]):
            move[0] = np.sign(direction[0]) # Move horizontally
        else:
            move[1] = np.sign(direction[1]) # Move vertically

        # Attack if close enough and cooldown is ready 
        if distance < 3.0 and self.laser_cooldown <= 0.0:
            self.boss_attacking = True 
            self.laser_timer = 0.0 
            self.boss_direction = direction

        return move 

    def step(self, human_action):
        """
        Execute one timestep
        """            

        # Store old distance for reward calculation 
        old_distance = np.linalg.norm(self.human_pos - self.boss_pos)
        
        