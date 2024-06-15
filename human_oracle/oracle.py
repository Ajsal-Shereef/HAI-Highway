import numpy as np
import matplotlib.pyplot as plt
from minigrid.core.constants import OBJECT_TO_IDX

class HumanOracle():
    """This class is a simmulated human which gives the safety feedback"""
    def __init__(self, env, mode):
        self.env = env
        self.cummulative_right_lane = 0
        self.mode = mode
        self.reset_arrays()

    def get_human_feedback(self):
        if self.mode == 'preference':
            return self.episode_right_lane #- self.episode_hitting
        if self.mode == 'avoid':
            return -self.episode_left_lane #- self.episode_hitting
        if self.mode == 'both':
            return -self.episode_left_lane + self.episode_right_lane #- self.episode_hitting
    
    def return_counts(self):
        if self.mode == 'preference':
            return self.cummulative_right_lane, self.episode_right_lane, self.episode_hitting
        if self.mode == 'avoid':
            return self.cummulative_left_lane, self.episode_left_lane, self.episode_hitting
        if self.mode == 'both':
            return self.cummulative_right_lane, self.cummulative_left_lane, self.episode_right_lane, self.episode_left_lane, self.episode_hitting
        
    def update_counts(self, info):
        vehicle = self.env.vehicle
        lane = vehicle.lane_index[-1]
        if lane == 2:
            self.episode_right_lane += 1
            self.cummulative_right_lane += 1
        if lane == 0:
            self.episode_left_lane += 1
            self.cummulative_left_lane += 1
        if info["crashed"]:
            self.episode_hitting = 1
        
    def reset_episode_count(self):
        self.episode_right_lane = 0
        self.episode_left_lane = 0
        self.episode_hitting = 0
    
    def reset_arrays(self):
        self.cummulative_right_lane = 0
        self.cummulative_left_lane = 0