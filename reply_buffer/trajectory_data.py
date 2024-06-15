import numpy as np
from human_oracle.oracle import HumanOracle

class TrajectoryData():
    def __init__(self):
        self.transitions = []
        self.trajectory = []
        
    def add_to_transition(self, transition):
        self.transitions.append(transition)

    def reset_transitions(self):
        self.transitions = []
           
    def add_to_trajectory(self):
        self.trajectory.append(self.transitions)
        
    def get_transitions(self):
        return self.transitions
    
    def get_trajectory(self):
        return self.trajectory