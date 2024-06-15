import sys
import gymnasium
sys.modules["gym"] = gymnasium

class HighWay(gymnasium.Wrapper):
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = max_steps
        
        self.cummulative_lower_threshold = 0
        self.cummulative_upper_threshold = 0
        self.cumulative_perfect_drive = 0
        self.reset_arrays()

    def step(self, action):
        observation, reward, self.done, truncated, info = self.env.step(action)
        #Count the upper lower limit violations
        if info["speed"] > 25:
            self.episode_upper_threshold += 1
            self.cummulative_upper_threshold += 1
        elif info["speed"] < 18:
            self.episode_lower_threshold += 1
            self.cummulative_lower_threshold += 1
        else:
            self.episode_perfect_drive += 1
            self.cumulative_perfect_drive += 1
        self.episode_step += 1
        truncated_by_step = self.is_episode_done()
        truncated = truncated + truncated_by_step
        return observation, reward, self.done, truncated, info
    
    def reset(self):
        self.episode_step = 0
        self.episode_upper_threshold = 0
        self.episode_lower_threshold = 0
        self.episode_perfect_drive =0
        return self.env.reset()
    
    def is_episode_complete(self):
        return self.episode_step == self.max_steps
    
    def reset_arrays(self):
        self.cummulative_lower_threshold = 0
        self.cummulative_upper_threshold = 0
        self.cumulative_perfect_drive = 0
        
    def return_counts(self):
        return self.cummulative_lower_threshold, self.cummulative_upper_threshold, self.cumulative_perfect_drive, self.episode_lower_threshold, self.episode_upper_threshold, self.episode_perfect_drive

    def is_episode_done(self):
        max_step_criteria = self.episode_step == self.max_steps
        return max_step_criteria or self.done