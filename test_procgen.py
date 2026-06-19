import gym
import procgen
import gymnasium

try:
    old_env = gym.make("procgen:procgen-coinrun-v0")
    print("Successfully created old gym environment!")
    
    # Try using shimmy
    from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
    env = GymV21CompatibilityV0(env=old_env)
    print("Successfully wrapped with shimmy!")
    
except Exception as e:
    print(f"Error: {e}")
