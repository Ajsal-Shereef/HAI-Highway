import argparse
import random
import torch
import sys
import numpy as np
import warnings
from addict import Dict
from env.env import Env
from dqn.dqn import DQNAgent
from logger.experiment_record_utils import ExperimentLogger
from utils.utils import create_dump_directory, get_config_file


experiment_log_dir = 'tmp/'
wandb_project_name = 'Highway_ECML'

warnings.filterwarnings("ignore")

def parse_args():
    # configurations
    parser = argparse.ArgumentParser(description="Expalainable AI")
    parser.add_argument("--env", type=str, default="Highway",
                        help="Environement to use")
    parser.add_argument("--use_logger", dest="use_logger", action="store_true", default=True,
                        help="whether store the results in logger")
    parser.add_argument("--log", dest="log", action="store_true",
                        help="turn on logging")
    parser.add_argument("--iteration_num", type=int, default= 2500,
                        help="total iteration num")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="total steps in an episode")
    parser.add_argument("--personalization_num_episode", type=int, default=1, #This is personalization number. Do not change this!!!!!!!!!!!!!!!! 
                        help="Personalization algorithm num_episode")
    parser.add_argument("--dqn_training", type=bool, default=True,
                        help="Whether DQN is training")
    parser.add_argument("--pref_model", type=str, default='LSTM',
                        help="Whether the preference model is LSTM/PEBBLE")
    parser.add_argument("--is_personalization", type=bool, default=False,
                        help="Is personalization of policy is applied")
    parser.add_argument("--lstm_config_path", type=str, default="lstm/lstm_cnn_config.yaml",
                        help="lstm config path")
    parser.add_argument("--model_path", type=str, default="Result/DQN_model/dqn_model.tar",
                        help="Save model_path")
    parser.add_argument("--save_freequency", type=int, default= 1000,
                        help="Save freequency of the DQN model")
    parser.add_argument("--dump_dir", type=str, default="Trajectory",
                        help="lstm config path")
    parser.add_argument("--load_data_dir", type=str, default="Trajectory",
                        help="lstm config path") 
    parser.add_argument("--result_dump_dir", type=str, default="Result",
                        help="Dump the results of the redistribution")
    parser.add_argument("--policy_fusion", type=str, default='entropy_threshold',
                        help="Which policy fusion method, entropy_weighted/product/average/entropy_threshold")
    parser.add_argument("--seed", type=int, default=777,
                        help="Seed")
    parser.add_argument("--lane_count", type=int, default=4,
                        help="Number of lanes in the highway")
    parser.add_argument("--t_max", type=int, default=5,
                        help="Maximum score")
    parser.add_argument("--dqn_temp", type=int, default=0.6,
                        help="Maximum score")
    parser.add_argument("--slop", type=int, default=1,
                        help="slop")
    parser.add_argument("--mode", type=str, default='both',
                        help="preference/avoid/both")
    parser.add_argument("--crt", type=list, default=[0],
                        help="Cumulative reward threshold")
    return parser.parse_args()

def run_game():
    args = parse_args()
    dump_dir = create_dump_directory(args.result_dump_dir)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    
    env = Env(args.max_steps, args.lane_count)
    env = env.get_env()
    policy_args = args
    policy_log_cfg = Dict()
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
                                    min_epsilon=0.10, # open-ai baselines: 0.01
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
    lstm_config = get_config_file(args.lstm_config_path)
    
    def create_training_decsription():
        text = ''
        # text = text + "{}".format(args.pref_model)
        # text = text + '_{}_epoch'.format(lstm_config["REWARD_LEARNING"]["n_update"])
        # text = text + '_{}_Trajectories'.format(lstm_config["REWARD_LEARNING"]["size"])
        # if args.pref_model == 'LSTM':
        #     text = text + '_{}_units'.format(lstm_config["REWARD_LEARNING"]["n_units"])
        # else:
        #     text = text + '_{}_batch_size'.format(lstm_config["REWARD_LEARNING"]["batch_size"])
        text = text + "{}".format(args.seed)
        text = text + "_t_max_{}".format(args.t_max)
        text = text + "_dqn_{}".format(args.dqn_temp)
        text = text + "_slop_{}".format(args.slop)
        # text += "_{}".format(args.lane_count)
        # text += '_{}'.format(policy_config["min_epsilon"])
        text = text + "_{}".format(args.mode)
        #text = text + "_{}".format(args.policy_fusion)
        #text = text + "_{}".format(args.iteration_num)
        #text = text + "_Ablation"
        text += '_' + wandb_project_name
        with open(dump_dir + '/Description.txt', 'w') as f:
            f.write(text)
        return text
    
    experiment_name = create_training_decsription()
    print(experiment_name)
    
    expr_logger = ExperimentLogger(experiment_log_dir, experiment_name, save_trajectories=False)
    #expr_logger = Logger(wandb_project_name, args.use_logger, experiment_name, experiment_log_dir)
    #expr_logger.initialise_neptune()
    expr_logger.set_is_use_wandb(args.use_logger)
    expr_logger.set_wandb_project_name(wandb_project_name)
    expr_logger.initialise_wandb(tags = [args.policy_fusion, args.mode])
    # expr_logger.save_config_wandb(config={"policy_config": policy_config,
    #                                       "policy_network_cfg": policy_network_cfg,
    #                                       "policy_optim_cfg": policy_optim_cfg})
    
    dqn_agent = DQNAgent(env, policy_args, policy_log_cfg, policy_config,
                        policy_network_cfg, policy_optim_cfg, dump_dir, lstm_config, logger=expr_logger)
    dqn_agent.train()
    
if __name__ == '__main__':
    run_game()
