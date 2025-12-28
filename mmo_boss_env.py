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
        self.time_elapsed = 0.0 
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
        
        # 1. Apply human action 
        self._apply_human_action(human_action)
        
        # 2. Apply boss action - simple AI 
        boss_move = self._get_boss_action()
        self.boss_pos += boss_move * self.BOSS_MOVE_SPEED

        # Keep the boss in the arena 
        self.boss_pos = np.clip(self.boss_pos, 0, self.ARENA_SIZE[0])

        # 3. Update game physics 
        self._update_physics()

        # 4. Check for laser hit 
        hit = self._check_laser_hit()

        if hit:
            self.human_hp -= 0.05
            if self.human_hp < 0:
                self.human_hp = 0


        # 5. Update time and check game end 
        self.time_elapsed += self.TIME_STEP 

        # 6. Calculate reward 
        reward = self._calculate_reward(hit, old_distance)

        # 7. Check termination 
        terminated = self._check_termination()

        # 8. Get observation 
        obs = self._get_obs()

        # Optional info 
        info = {
            "human_hp" : self.human_hp, 
            "time_remaining" : self.GAME_DURATION - self.time_elapsed
        }

        return obs, reward, terminated, False, info 

    def _apply_human_action(self, action):
        """
        Convert discrete actions into movement
        """

        move = np.array([0.0, 0.0])

        if action == 0: # Move up
            move[1] = 1

        elif action == 1: # Move down
            move[1] = -1

        elif action == 2: # Move left
            move[0] = -1

        elif action == 3: # Move right
            move[0] = 1

        elif action == 4: # Jump
            if not self.human_jumping:
                self.human_jumping = True
                self.jump_tier = 0.0

        elif action == 5: # Crouch (simplified - just move slower - later in 3d we will apply this)
            move *= 0.5

        # Apply movement 
        self.human_pos += move * self.HUMAN_MOVE_SPEED 

        # Keep human in the arena 
        self.human_pos = np.clip(self.human_pos, 0, self.ARENA_SIZE[0])

    def _update_physics(self):
        """
        Update jump and laser time
        """

        # Update jump 
        if self.human_jumping:
            self.jump_tier += self.TIME_STEP # jump-tier : rep how long human has been jumping for
            if self.jump_tier >= self.HUMAN_JUMP_DURATION: 
                self.human_jumping = False

        #  Update laser 
        if self.boss_attacking:
            self.laser_timer += self.TIME_STEP
            if self.laser_timer >= self.BOSS_LASER_DURATION: 
                self.boss_attacking = False
                self.laser_cooldown = self.BOSS_LASER_COOLDOWN
        elif self.laser_cooldown > 0: # When boss is not attacking 
            self.laser_cooldown -= self.TIME_STEP

    def _check_laser_hit(self):
        """
        Check if laser hit human
        """

        if not self.boss_attacking:
            return False 

        # Simplified hit check: if human is in laser direction and close enough 

        laser_vector = self.boss_direction # direction where the boss aim
        human_vector = self.human_pos - self.boss_pos # calculate vector pointing from boss to human 
        distance = np.linalg.norm(human_vector) # calculate straight-line distance (magnitude of vector) between boss and human 

        if distance > 0: 
            human_vector = human_vector / distance # normalize human vector to unit vector 

        # Dot product to check alignment 
        dot_product = np.dot(laser_vector, human_vector)

        # Hit if aligned and within range 
        if dot_product > 0.9 and distance < 5.0:
            # Jumping can avoid low laser 
            if self.human_jumping and laser_vector[1] < 0 : # check vertical y-axis 
                return False
            return True 
        return False 

    def _calculate_reward(self, hit, old_distance):
        """
        Calculate reward for human

        distance_penalty : designed to teach the AI to stay at a specific "sweet spot" distance from the boss (in this case, 3.0 units), 
        rather than running far away or hugging the boss.
        How it works: 
        -abs(current_distance - optimal_distance) * 0.01 : calculate how far off the player from the perfect 3.0 range. *0.01 to make it doesnt overpower other rewards
        if, too close -> -|1.0 - 3.0| = -2.0 (error)
        if, too far -> -|5.0 - 3.0| = -2.0 (error)
        if, just right -> |3.0 - 3.0| = 0.0 (error)

        """
        reward = 0.0

        # Small survival reward 
        reward += 0.01

        # Big penalty for getting hit 
        if hit:
            reward -= 0.5

        # Reward for maintaining distance 
        current_distance = np.linalg.norm(self.human_pos - self.boss_pos)
        optimal_distance = 3.0 
        distance_penalty = -abs(current_distance - optimal_distance) * 0.01 
        reward += distance_penalty

        # Bonus for surviving 
        if not hit and self.boss_attacking:
            reward += 0.1 

        # Big bonus / penalty at game end 
        if self.game_over:
            if self.winner == "human":
                reward += 5.0 
            else:
                reward -= 3.0

    def _check_termination(self):
        """
        Check if game should end
        """

        # Check HP 
        if self.human_hp <= 0:
            self.game_over = True 
            self.winner = "boss"
            return True 

        # Check time 
        if self.time_elapsed >= self.GAME_DURATION:
            self.game_over = True 
            self.winner = "human"
            return True 

        return False
    
    def render(self):
        """
        Render the game 
        """
        if self.render_mode != "human":
            return 

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
        # Clear screen 
        self.screen.fill((30, 30, 30))

        # Draw arena border 
        pygame.draw.rect(self.screen, (200, 200, 200), (50, 50, 500, 500), 2) # surface, color, rect (l,t,w,h), thickness

        # Draw boss 
        boss_screen_pos = (50 + self.boss_pos[0] * 50, 550 - self.boss_pos[1] * 50)
        boss_color = (255, 255, 255) if self.boss_attacking else (255, 50, 50)
        pygame.draw.circle(self.screen, boss_color, boss_screen_pos, 30)

        # Draw laser if attacking 
        if self.boss_attacking: 
            laser_end = (
                boss_screen_pos[0] + self.boss_direction[0] * 200, 
                boss_screen_pos[1] - self.boss_direction[1] * 200
            )
            pygame.draw.line(self.screen, (255, 255, 100), boss_screen_pos, laser_end, 5)

        # Draw human 
        human_screen_pos = (50 + self.human_pos[0] * 50, 550 - self.human_pos[1] * 50)
        human_color = (100, 100, 255) if self.human_jumping else (100, 150, 255)
        human_size = 20 if self.human_jumping else 15
        pygame.draw.circle(self.screen, human_color, human_screen_pos, human_size)
        
        # Draw HP bar
        hp_width = 200 * self.human_hp
        pygame.draw.rect(self.screen, (50, 50, 50), (200, 570, 204, 24))
        pygame.draw.rect(self.screen, (100, 255, 100), (202, 572, hp_width, 20))
        
        # Draw info text
        time_text = self.font.render(f"Time: {self.GAME_DURATION - self.time_elapsed:.1f}s", True, (255, 255, 255))
        hp_text = self.font.render(f"HP: {self.human_hp:.2f}", True, (255, 255, 255))
        self.screen.blit(time_text, (20, 20))
        self.screen.blit(hp_text, (20, 60))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(30)
    
    def close(self):
        if self.render_mode == "human":
            pygame.quit()
