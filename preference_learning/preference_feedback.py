import numpy as np

class FeedBack():
    def __init__(self,  size):
        self.size = size
        self.trajectory_states = []
        self.trajectory_actions = []
        self.feedback_array = []
        self.length_array = []
        
    def get_preference_feedback(self, states, actions, human_feedback, lens):
        #Uniform sampling of data from the collected trajectory data of agent to train the LSTM
        # Create a subset of main lesson buffer to train the LSTM. This should be same as that of the LSTM training
        # states = self.buffer.lstm_reply_buffer.states_buffer
        # actions = self.buffer.lstm_reply_buffer.actions_buffer
        # lens = self.buffer.lstm_reply_buffer.lens_buffer
        # action_one_hot = custom_action_encoding(actions, self.n_actions, self.action_embedding_dim)
        # human_feedback = self.buffer.lstm_reply_buffer.rewards_buffer
        human_feedback = human_feedback[:,-1]
        for i in range(self.size):
            for j in range(i+1, self.size):
                self.feedback_array.append(human_feedback[i] > human_feedback[j])
                self.trajectory_states.append([states[i], states[j]])
                self.trajectory_actions.append([actions[i], actions[j]])
                self.length_array.append([lens[i][0], lens[j][0]])
        return np.array(self.trajectory_states), np.array(self.trajectory_actions), np.array(self.feedback_array), self.length_array