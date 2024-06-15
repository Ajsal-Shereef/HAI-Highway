import sys
import gymnasium
import numpy as np
sys.modules["gym"] = gymnasium

import highway_env
from gymnasium.core import ObservationWrapper
from utils.utils import record_videos

from env.highway import HighWay


class Env(ObservationWrapper):
    "This class creates the self.environement"
    def __init__(self, max_steps, lane_count):
        self.env = gymnasium.make('highway-fast-v0')#, render_mode=None)
        config = {
                    "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "features_range": {
                                        "x": [-100, 100],
                                        "y": [-100, 100],
                                        "vx": [-20, 20],
                                        "vy": [-20, 20]
                                    },
                    "absolute": False,
                    "order": "sorted",
                    "show_trajectories": True,
                    }
                }
        self.env.configure(config)
        self.env.config["duration"] = max_steps
        self.env.config["right_lane_reward"] = 0
        self.env.config['lanes_count'] = lane_count
        # self.env.config["high_speed_reward"] = 0.4
        # self.env.config["reward_speed_range"] = [20, 30]
        # self.env.config['vehicles_density'] = 1.0
        self.env.reset()
        print(self.env.config)
        #self.env = record_videos(self.env)
        #self.env = HighWay(self.env, max_steps)
    
    def get_env(self):
        return self.env
    
    