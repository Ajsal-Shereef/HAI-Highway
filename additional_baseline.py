import cv2
import torch
import argparse
import warnings
import numpy as np

from addict import Dict
from env.env import Env
from dqn.dqn import DQNAgent
from lstm.convo_lstm_model import RRLSTM
from human_oracle.oracle import HumanOracle
from logger.experiment_record_utils import ExperimentLogger
from utils.utils import get_device, get_config_file, write_video, save_models, range_map, write_log

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
    parser.add_argument("--iteration_num", type=int, default= 50000,
                        help="number of time DQN need to train")
    parser.add_argument("--num_test_episode", type=int, default= 20,
                        help="number of time DQN need to train")
    parser.add_argument("--num_data", type=int, default= 2000,
                        help="number of trajectory data for DQN")
    parser.add_argument("--lstm_config_path", type=str, default="lstm/lstm_cnn_config.yaml",
                        help="lstm config path")
    parser.add_argument("--lstm_model_path", type=str, default="Result/2023-11-07_20-09-19_89IPWJ/lstm_45000.tar",
                        help="LSTM model path")
    parser.add_argument("--dqn_model_path", type=str, default="Result/DQN_model/dqn_model.tar",
                        help="DQN model path")
    parser.add_argument("--datapath", type=str, default="Trajectory/both",
                        help="DQN model training trajectory data")
    parser.add_argument("--mode", type=str, default='avoid',
                        help="preference/avoid/both")
    parser.add_argument("--human_reward_weight", type=float, default= 0.3,
                        help="number of time DQN need to train")
    return parser.parse_args()

def create_training_decsription():
    text = 'Additional_baseline'
    text = text + "_{}".format(args.mode)
    text = text + "_{}".format(args.num_data)
    text = text + "_{}".format(args.human_reward_weight)
    #text = text + "_{}".format(args.policy_fusion)
    #text = text + "_{}".format(args.iteration_num)
    #text = text + "_Ablation"
    text += '_' + wandb_project_name
    return text


def run_episode(dqn_class, total_step):
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
    frame_array = []
    while not done:
        episode_step += 1
        total_step += 1
        action = dqn_class.select_action(previous_state_encoding, 0)
        next_state, reward, terminated, truncated, info = env.step(action[0])
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
        frame = env.observation_type.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_array.append(frame)
    return score, episode_step, frame_array

def load_data_to_buffer(dqn_class, lstm_model, datapath):
    states_buffer = np.load(datapath + '/states_buffer.npy')
    next_states_buffer = np.load(datapath + '/next_states_buffer.npy')
    actions_buffer = np.load(datapath + '/actions_buffer.npy')
    lens_buffer = np.load(datapath + '/lens_buffer.npy')
    rewards_buffer = np.load(datapath + '/rewards_buffer.npy')
    
    indices = np.random.choice(states_buffer.shape[0], size=args.num_data, replace=False)
    states_buffer = states_buffer[indices, :, :]
    next_states_buffer = next_states_buffer[indices, :, :]
    actions_buffer = actions_buffer[indices, :]
    rewards_buffer = rewards_buffer[indices, :]
    lens_buffer = lens_buffer[indices, :]
    
    lstm_out = lstm_model(torch.tensor(states_buffer).to(device), torch.tensor(actions_buffer).to(device), torch.tensor(lens_buffer).to(device))
    pred_g0 = torch.cat([torch.zeros_like(lstm_out[0][:, 0:1, :]), lstm_out[0]], dim=1)
    human_reward = pred_g0[:, 1:, 0] - pred_g0[:, :-1, 0]
    human_reward = human_reward.detach().cpu().numpy()
    human_reward = range_map(human_reward, [np.min(human_reward), np.max(human_reward)], [-1,1])
    rewards_buffer = (1-args.human_reward_weight)*rewards_buffer + args.human_reward_weight*human_reward
    for eps in range(states_buffer.shape[0]):
        for t in range(lens_buffer[eps][0]):
            if t == lens_buffer[eps]-1:
                done = True
            else:
                done = False
            transition = (states_buffer[eps][t], actions_buffer[eps][t], rewards_buffer[eps][t], next_states_buffer[eps][t], done, {})
            dqn_class.add_transition_to_memory(transition)
            
def test(dqn_class):
    total_step = 0
    cumulative_hitting = 0
    for episode in range(args.num_test_episode):
        oracle.reset_episode_count()
        score, episode_step, frame_array = run_episode(dqn_class, total_step)
        if args.mode == 'both':
            cummulative_right_lane, cummulative_left_lane, episode_right_lane, episode_left_lane, episode_hitting = oracle.return_counts()
            cumulative_hitting += episode_hitting 
            log_keys = ["Score" ,"Episode step" , "cummulative_right_lane", "cummulative_left_lane" ,"episode_right_lane", "episode_left_lane" ,"episode_hitting",\
                        "Cumulative hititng" , "Episode"]
            log_values = [score, episode_step, cummulative_right_lane, cummulative_left_lane, episode_right_lane, episode_left_lane, episode_hitting, cumulative_hitting, episode]
        elif args.mode == 'preference':
            cummulative_right_lane, episode_right_lane, episode_hitting = oracle.return_counts()
            cumulative_hitting += episode_hitting 
            log_keys = ["Score", "Episode step", "cummulative_right_lane", "episode_right_lane", "episode_hitting",
                        "Cumulative hititng", "Episode"]
            log_values = [score, episode_step, cummulative_right_lane, episode_right_lane, episode_hitting, cumulative_hitting, episode]
        else:
            cummulative_left_lane, episode_left_lane, episode_hitting = oracle.return_counts()
            cumulative_hitting += episode_hitting 
            log_keys = ["Score", "Episode step", "cummulative_left_lane", "episode_left_lane", "episode_hitting",\
                        "Cumulative hititng", "Episode"]
            log_values = [score, episode_step, cummulative_left_lane, episode_left_lane, episode_hitting, cumulative_hitting, episode]
        write_log(expr_logger, log_keys, log_values)
        write_video(frame_array, str(episode), "additional_baseline/Videos")
    if args.use_logger:
        expr_logger.finish_run()

def train():
    policy_config = Dict(dict(gamma=0.8,
                                target_network_update_interval=1000,
                                #tau=1e-3,
                                buffer_size=int(1.5e5),  # open-ai baselines: int(1e4)
                                batch_size=64,  # open-ai baselines: 32
                                init_random_actions=int(5e3),
                                update_starts_from=int(1e3),  # open-ai baselines: int(1e4)
                                multiple_update=1,  # multiple learning updates
                                train_freq=4,  # in open-ai baselines, train_freq = 4
                                reward_clip=[-1, 1],
                                reward_scale=1.0,
                                gradient_clip=10.0,  # dueling: 10.0
                                # N-Step Buffer
                                n_step=5,  # if n_step <= 1, use common replay buffer otherwise n_step replay buffer
                                w_n_step=1.0,  # used in n-step update
                                # Double Q-Learning
                                use_double_q_update=True,
                                max_entropy_objective = True,
                                # Prioritized Replay Buffer
                                use_prioritized=True,
                                per_alpha=0.6,  # open-ai baselines default: 0.6, alpha -> 1, full prioritization
                                per_beta=0.4,  # beta can start small (for stability concern and anneals towards 1)
                                max_entropy_alpha = 0.1, #Weigh of entropy regularization
                                per_eps=1e-6,
                                std_init=0.5,
                                # Epsilon Greedy
                                max_epsilon=1.0, #1 Means there is no decay
                                min_epsilon=0.5, # open-ai baselines: 0.01
                                epsilon_decay=0.995,  # default: 0.9995
                                # auto-encoder
                                n_random_cae_sample=0,  # 0 if no cae pre-training
                                cae_batch_size=32,
                                ))
    policy_network_cfg = Dict(dict(fc_input_size=26,
                                   nonlinearity=torch.relu,
                                   channels=[32, 64, 64],
                                   kernel_sizes=[11, 5, 3],
                                   strides=[2, 2, 1],
                                   paddings=[0, 0, 0],
                                   fc_hidden_sizes=[256,512,512],
                                   fc_hidden_activation=torch.relu,
                                   # decoder
                                   deconv_input_channel=64,
                                   deconv_channels=[64, 32, 4],
                                   deconv_kernel_sizes=[3, 4, 8],
                                   deconv_strides=[1, 2, 4],
                                   deconv_paddings=[0, 0, 0],
                                   deconv_activation_fn=[torch.relu, torch.relu, torch.sigmoid],
                                   ))
    policy_optim_cfg = Dict(dict(lr_dqn=1e-4,
                                 adam_eps=1e-6,     # default value in pytorch 1e-6
                                 weight_decay=1e-8,
                                 w_q_reg=0,     # use q_value regularization
                                 ))
    #Loading the LSTM model
    lstm_model = RRLSTM(lstm_config).to(device)
    params = torch.load(args.lstm_model_path, map_location=device)
    lstm_model.load_state_dict(params["lstm_weight"])
    lstm_model.eval()
    
    dqn_class = DQNAgent(env, args, Dict(), policy_config,
                         policy_network_cfg, policy_optim_cfg, "additional_baseline", lstm_config, logger=expr_logger)
    dqn_class.load_params(args.dqn_model_path, device)
    load_data_to_buffer(dqn_class, lstm_model, args.datapath)
    for epoch in range(args.iteration_num):
       loss = dqn_class.update_model()
       expr_logger.log_wandb({"loss": loss[0],
                              "epoch":epoch})
    params = {"dqn_state_dict": dqn_class.dqn.state_dict(),
              "dqn_target_state_dict": dqn_class.dqn_target.state_dict(),
              "dqn_optim_state_dict": dqn_class.dqn_optim.state_dict(),
             }
    save_models(params, "additional_baseline", "dqn_human_reward")
    dqn_class.total_step = policy_config.init_random_actions + 1
    test(dqn_class)

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
    train()