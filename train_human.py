import numpy as np 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from mmo_boss_env import MMOBossEnv
import os 

# Create logs directory 

log_dir = "./ppo_logs/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment 
print("Creating environment...")
env = DummyVecEnv([lambda: MMOBossEnv()])

# Create PPO model 
print("Creating PPO model...")
model = PPO(
    "MlpPolicy", 
    env,
    learning_rate = 3e-4,
    n_steps=2048,
    batch_size=64, 
    n_epochs=10,
    gamma=0.99, 
    gae_lambda=0.95, 
    clip_range=0.2, 
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=log_dir
)

# Create evaluation callback 
print("Setting up training .....")
eval_callback = EvalCallback(
    env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Train the model 
print("Training model...")
try:
    model.learn(
        total_timesteps=100000,
        callback=eval_callback,
        progress_bar=True
    )
except KeyboardInterrupt:
    print("Training interrupted by user.")


# Save the trained model 
print("Saving the trained model...")
model.save("ppo_mmo_human")
print("Model saved as ppo_mmo_human")

# Test the trained model 
print("Testing the trained model...")
test_env = MMOBossEnv(render_mode="human")
obs = test_env.reset()

total_reward = 0
episode_count = 0
max_episodes = 5

while episode_count < max_episodes:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    total_reward += reward 

    test_env.render()

    if terminated or truncated:
        print(f"Episode {episode_count + 1}: Reward = {total_reward:.2f}, HP={info['human_hp']:.2f}")
        obs = test_env.reset()
        total_reward = 0
        episode_count += 1

test_env.close()
print("\n Demo completed")
    
