import torch
import argparse
import warnings
import numpy as np

from env.env import Env
from lstm.convo_lstm_model import RRLSTM
from human_oracle.oracle import HumanOracle
from logger.experiment_record_utils import ExperimentLogger
from utils.utils import preprocess_state_for_inference, get_device, get_config_file, return_legal_actions, write_log

device = get_device()

experiment_log_dir = 'tmp/'
wandb_project_name = 'Highway_HAI_additional_baseline'

warnings.filterwarnings("ignore")

def parse_args():
    # configurations
    parser = argparse.ArgumentParser(description="Expalainable AI")
    parser.add_argument("--env", type=str, default="Highway",
                        help="Environement to use")
    parser.add_argument("--use_logger", dest="use_logger", action="store_true", default=True,
                        help="whether store the results in logger")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="total steps in an episode")
    parser.add_argument("--iteration_num", type=int, default= 20,
                        help="total iteration num")
    parser.add_argument("--lstm_config_path", type=str, default="lstm/lstm_cnn_config.yaml",
                        help="lstm config path")
    parser.add_argument("--lstm_model_path", type=str, default="Result/2023-11-07_17-02-12_FVZ39C/lstm_64.tar",
                        help="LSTM model path")
    parser.add_argument("--mode", type=str, default='avoid',
                        help="preference/avoid/both")
    return parser.parse_args()

def create_training_decsription():
    text = 'Additional_baseline_RUDDER'
    text = text + "_{}".format(args.mode)
    #text = text + "_{}".format(args.policy_fusion)
    #text = text + "_{}".format(args.iteration_num)
    #text = text + "_Ablation"
    #text += '_' + wandb_project_name
    return text

def select_action(state, model):
    state = preprocess_state_for_inference(state, device)
    state_input = state.unsqueeze(0).unsqueeze(0).type(torch.float32).to(device)
    size = state_input.size(0)
    q_value = []
    vehicle = env.vehicle
    lane = vehicle.lane_index[-1]
    valid_actions = return_legal_actions(env, lane)
    #state_input = state_input.type(torch.float32)
    for act in range(5):
        action_vec = np.expand_dims(np.array([act]), axis = 0)
        action_tensor = torch.tensor(action_vec).to(device)
        size = state_input.size(0)
        state_encoding = state_input
        selected_action_q_value, _, attn = model(state_encoding, action_tensor, torch.tensor([[size]]))
        selected_action_q_value = selected_action_q_value.detach().cpu().numpy()
        q_value.append(selected_action_q_value[0,-1,0])
    sorted_policy_output = np.argsort(q_value)
    for action in reversed(sorted_policy_output):
        if action in valid_actions:
            selected_action = action
            break
    return selected_action

def run_episode(model):
    previous_state = env.reset()[0]
    previous_state_encoding = previous_state.flatten(order = 'C')
    vehicle = env.vehicle
    lane = vehicle.lane_index[-1]
    previous_state_encoding = np.concatenate((previous_state_encoding, np.array([lane])))
    #previous_encoding = get_state_one_hot(previous_state[0]['agent'])
    episode_step = 0
    done = False
    score = 0
    reward = 0
    while not done:
        episode_step += 1
        action = select_action(previous_state_encoding, model)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated + truncated
        oracle.update_counts(info)
        original_reward = reward
        next_state_encoding = next_state.flatten(order = 'C')
        vehicle = env.vehicle
        lane = vehicle.lane_index[-1]
        next_state_encoding = np.concatenate((next_state_encoding, np.array([lane])))
        previous_state_encoding = next_state_encoding
        previous_state = next_state
        score += original_reward
    return score, episode_step

def rollout():
    model = RRLSTM(lstm_config).to(device)
    params = torch.load(args.lstm_model_path, map_location=device)
    model.load_state_dict(params["lstm_weight"])
    print("[INFO] LSTM model loaded from ", args.lstm_model_path)
    model.eval()
    cumulative_hitting = 0
    for i in range(args.iteration_num):
        oracle.reset_episode_count()
        score, episode_step = run_episode(model)
        if args.mode == 'both':
            cummulative_right_lane, cummulative_left_lane, episode_right_lane, episode_left_lane, episode_hitting = oracle.return_counts()
            cumulative_hitting += episode_hitting 
            log_keys = ["Score" ,"Episode step" , "cummulative_right_lane", "cummulative_left_lane" ,"episode_right_lane", "episode_left_lane" ,"episode_hitting",\
                        "Cumulative hititng" , "Episode"]
            log_values = [score, episode_step, cummulative_right_lane, cummulative_left_lane, episode_right_lane, episode_left_lane, episode_hitting, cumulative_hitting, i]
        elif args.mode == 'preference':
            cummulative_right_lane, episode_right_lane, episode_hitting = oracle.return_counts()
            cumulative_hitting += episode_hitting 
            log_keys = ["Score", "Episode step", "cummulative_right_lane", "episode_right_lane", "episode_hitting",
                        "Cumulative hititng", "Episode"]
            log_values = [score, episode_step, cummulative_right_lane, episode_right_lane, episode_hitting, cumulative_hitting, i]
        else:
            cummulative_left_lane, episode_left_lane, episode_hitting = oracle.return_counts()
            cumulative_hitting += episode_hitting 
            log_keys = ["Score", "Episode step", "cummulative_left_lane", "episode_left_lane", "episode_hitting",\
                        "Cumulative hititng", "Episode"]
            log_values = [score, episode_step, cummulative_left_lane, episode_left_lane, episode_hitting, cumulative_hitting, i]
        write_log(expr_logger, log_keys, log_values)
        
if __name__ == '__main__':
    args = parse_args()
    lstm_config = get_config_file(args.lstm_config_path)
    env = Env(args.max_steps, 4)
    env = env.get_env()
    oracle = HumanOracle(env, args.mode)
    expr_logger = ExperimentLogger(experiment_log_dir, create_training_decsription(), save_trajectories=False)
    #expr_logger = Logger(wandb_project_name, args.use_logger, experiment_name, experiment_log_dir)
    #expr_logger.initialise_neptune()
    expr_logger.set_is_use_wandb(args.use_logger)
    expr_logger.set_wandb_project_name(wandb_project_name)
    expr_logger.initialise_wandb()
    rollout()