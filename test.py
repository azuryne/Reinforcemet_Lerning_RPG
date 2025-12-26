# ========================================
# USING GYMNASIUM + STABLE-BASELINES3
# Professional RL Libraries
# ========================================

# Step 1: Install required libraries (run in terminal or notebook)
# !pip install gymnasium stable-baselines3 numpy matplotlib

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

# Step 2: Create Custom Gymnasium Environment
class MonsterAvoidanceEnv(gym.Env):
    """
    Custom Gymnasium environment for monster laser avoidance.
    Follows the standard Gym API for compatibility with all RL libraries.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, grid_size=10, monster_zone=2, attack_buffer=3):
        super().__init__()
        
        # Environment parameters
        self.grid_size = grid_size
        self.monster_zone = monster_zone
        self.attack_buffer = attack_buffer
        
        # Define action and observation spaces (required by Gymnasium)
        # Action space: 0=stay, 1=up, 2=down, 3=left, 4=right
        self.action_space = spaces.Discrete(5)
        
        # Observation space: [human_x, human_y, cooldown, laser_active, laser_row]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([grid_size-1, grid_size-1, attack_buffer, 1, grid_size-1]),
            dtype=np.float32
        )
        
        # Initialize state
        self.human_pos = None
        self.monster_pos = None
        self.attack_cooldown = 0
        self.laser_active = False
        self.laser_row = 0
        self.steps = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.human_pos = np.array([self.grid_size - 1, self.grid_size // 2])
        self.monster_pos = np.array([1, np.random.randint(0, self.grid_size)])
        self.attack_cooldown = 0
        self.laser_active = False
        self.laser_row = 0
        self.steps = 0
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def _get_obs(self):
        """Return current observation"""
        return np.array([
            self.human_pos[0],
            self.human_pos[1],
            self.attack_cooldown,
            float(self.laser_active),
            float(self.laser_row)
        ], dtype=np.float32)
    
    def _get_info(self):
        """Return additional information"""
        return {
            'human_pos': self.human_pos.copy(),
            'monster_pos': self.monster_pos.copy(),
            'steps': self.steps
        }
    
    def step(self, action):
        """Execute action and return (obs, reward, terminated, truncated, info)"""
        self.steps += 1
        old_pos = self.human_pos.copy()
        
        # Execute action
        if action == 1:  # Up
            self.human_pos[1] = min(self.grid_size - 1, self.human_pos[1] + 1)
        elif action == 2:  # Down
            self.human_pos[1] = max(0, self.human_pos[1] - 1)
        elif action == 3:  # Left
            self.human_pos[0] = max(0, self.human_pos[0] - 1)
        elif action == 4:  # Right
            self.human_pos[0] = min(self.grid_size - 1, self.human_pos[0] + 1)
        
        # Monster attack logic
        self.laser_active = False
        if self.attack_cooldown == 0:
            if np.random.random() < 0.4:
                self.laser_active = True
                self.laser_row = self.monster_pos[1]
                self.attack_cooldown = self.attack_buffer
        else:
            self.attack_cooldown -= 1
        
        # Calculate reward
        reward = 0
        terminated = False
        
        if self.laser_active and self.human_pos[1] == self.laser_row:
            reward = -100  # Hit by laser
            terminated = True
        else:
            reward = 1  # Survived
            if self.laser_active and old_pos[1] == self.laser_row and self.human_pos[1] != self.laser_row:
                reward += 10  # Dodged successfully
        
        # Truncate if max steps reached
        truncated = self.steps >= 100
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Visualize the environment"""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Monster zone
        monster_area = Rectangle((-0.5, -0.5), self.monster_zone, self.grid_size, 
                                alpha=0.2, color='red', label='Monster Zone')
        ax.add_patch(monster_area)
        
        # Laser
        if self.laser_active:
            ax.plot([self.monster_pos[0], self.grid_size], 
                   [self.laser_row, self.laser_row], 
                   'r-', linewidth=8, alpha=0.7, label='Laser')
        
        # Monster and Human
        ax.plot(self.monster_pos[0], self.monster_pos[1], 'ro', markersize=20, label='Monster')
        ax.plot(self.human_pos[0], self.human_pos[1], 'bo', markersize=15, label='Human')
        
        ax.legend(loc='upper right')
        ax.set_title(f'Step: {self.steps} | Cooldown: {self.attack_cooldown}')
        plt.show()


# Step 3: Custom Callback for Logging
class TrainingCallback(BaseCallback):
    """Custom callback to log training progress"""
    
    def __init__(self, check_freq=1000):
        super().__init__()
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            print(f"Steps: {self.n_calls} | Episodes: {len(self.episode_rewards)}")
        return True


# Step 4: Train with Different Algorithms
def train_with_sb3(algorithm='PPO', total_timesteps=50000):
    """
    Train using Stable-Baselines3 algorithms
    
    Available algorithms:
    - PPO (Proximal Policy Optimization): Good all-around, stable
    - DQN (Deep Q-Network): Classic deep RL
    - A2C (Advantage Actor-Critic): Fast, good for simple tasks
    """
    print(f"\n{'='*50}")
    print(f"Training with {algorithm}")
    print(f"{'='*50}\n")
    
    # Create environment
    env = MonsterAvoidanceEnv()
    
    # Verify environment follows Gym API
    print("Checking environment compatibility...")
    check_env(env, warn=True)
    print("✓ Environment is compatible!\n")
    
    # Create model based on algorithm choice
    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy',  # Multi-layer perceptron policy
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            tensorboard_log=f"./tensorboard/{algorithm}/"
        )
    elif algorithm == 'DQN':
        model = DQN(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            tensorboard_log=f"./tensorboard/{algorithm}/"
        )
    elif algorithm == 'A2C':
        model = A2C(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            tensorboard_log=f"./tensorboard/{algorithm}/"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Train the model
    callback = TrainingCallback(check_freq=5000)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save the model
    model.save(f"monster_avoidance_{algorithm}")
    print(f"\n✓ Model saved as 'monster_avoidance_{algorithm}.zip'")
    
    return model, env


# Step 5: Evaluate Trained Model
def evaluate_model(model, env, n_episodes=10):
    """Evaluate the trained model"""
    print(f"\n{'='*50}")
    print("EVALUATION")
    print(f"{'='*50}\n")
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Use trained model to predict action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            if terminated and reward > 0:
                success_count += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        print(f"Episode {episode+1}: Reward={total_reward:.1f}, Steps={steps}")
    
    print(f"\n--- Results ---")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Success Rate: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    
    return episode_rewards, episode_lengths


# Step 6: Compare Multiple Algorithms
def compare_algorithms(algorithms=['PPO', 'DQN', 'A2C'], timesteps=50000):
    """Train and compare different RL algorithms"""
    results = {}
    
    for algo in algorithms:
        model, env = train_with_sb3(algo, total_timesteps=timesteps)
        rewards, lengths = evaluate_model(model, env, n_episodes=20)
        results[algo] = {
            'rewards': rewards,
            'lengths': lengths,
            'mean_reward': np.mean(rewards),
            'mean_length': np.mean(lengths)
        }
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Compare rewards
    for algo, data in results.items():
        ax1.bar(algo, data['mean_reward'], alpha=0.7, label=algo)
        ax1.errorbar(algo, data['mean_reward'], 
                    yerr=np.std(data['rewards']), 
                    fmt='o', color='black', capsize=5)
    
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Algorithm Performance Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Compare episode lengths
    for algo, data in results.items():
        ax2.bar(algo, data['mean_length'], alpha=0.7, label=algo)
        ax2.errorbar(algo, data['mean_length'], 
                    yerr=np.std(data['lengths']), 
                    fmt='o', color='black', capsize=5)
    
    ax2.set_ylabel('Average Episode Length')
    ax2.set_title('Survival Time Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return results


# Step 7: Main Execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("REINFORCEMENT LEARNING WITH STABLE-BASELINES3")
    print("="*70)
    
    # Option 1: Train single algorithm
    print("\n[Option 1] Training with PPO (recommended)...")
    model, env = train_with_sb3('PPO', total_timesteps=50000)
    evaluate_model(model, env, n_episodes=10)
    
    # Option 2: Compare algorithms (uncomment to run)
    # print("\n[Option 2] Comparing multiple algorithms...")
    # results = compare_algorithms(['PPO', 'A2C'], timesteps=30000)
    
    print("\n" + "="*70)
    print("DONE! You can now:")
    print("1. Load model: model = PPO.load('monster_avoidance_PPO')")
    print("2. View tensorboard: tensorboard --logdir ./tensorboard/")
    print("3. Test model: obs, _ = env.reset(); action, _ = model.predict(obs)")
    print("="*70)


# ========================================
# ADDITIONAL: OTHER POPULAR RL LIBRARIES
# ========================================

"""
OTHER LIBRARIES YOU CAN USE:

1. RAY RLLIB (Scalable, distributed RL)
   - Best for: Large-scale training, multi-agent RL
   - pip install ray[rllib]
   
2. TENSORFLOW AGENTS (TF-Agents)
   - Best for: TensorFlow users, research
   - pip install tf-agents
   
3. TIANSHOU (PyTorch-based)
   - Best for: Fast prototyping, modular design
   - pip install tianshou
   
4. CLEANRL (Single-file implementations)
   - Best for: Learning, simple implementations
   - pip install cleanrl
   
5. MUSHROOM-RL (Research-oriented)
   - Best for: Academic research, custom algorithms
   - pip install mushroom-rl

STABLE-BASELINES3 is recommended for beginners because:
- Easy to use, well-documented
- Reliable implementations
- Active community support
- Works great with Gymnasium
"""