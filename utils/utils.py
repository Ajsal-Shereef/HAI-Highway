import cv2
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import re
import os
import pickle
import yaml
import random
import string
from sklearn import metrics
from widis_lstm_tools.utils.config_tools import ObjectDict
from torch.autograd import Variable
from datetime import datetime
from scipy.special import softmax
from gymnasium.wrappers import RecordVideo

dqn_log_dic_keys = ["episode", "episode step", "total step", 
                "dqn loss", "avg q values", "epsilon", 
                "score", "time per each step", "num_bumps"]
safety_dqn_log_dic_keys = ["episode", "episode step", "total step", 
                "dqn loss", "avg q values", "epsilon", 
                "score", "time per each step", "num_bumps", "safety_num_bumps", "quality"]
lstm_log_dic_keys = ["episodes", "num_training", "aux_loss", "main_loss", "lstm_loss"]
safety_dqn_dic = ["episode", "total_step", "dqn_loss", "avg q values"]

GPU_DEVICE = torch.cuda.device_count() - 1


def convert_gray_scale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def get_config_file(configfile):
    with open(configfile, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

# def state_encoding(agent_pos, action):
#     state_encoding = np.zeros((17,1), dtype=np.float32)
#     if agent_pos == (3, 1):
#        state_encoding[2] = 2
#     elif agent_pos == (4,4) 

def create_log_dict(dict_key, values):
    dict = {}
    for key, value in zip(dict_key, values):
        dict[key] = value
    return dict

def write_log(logger, keys, values):
    dict = create_log_dict(keys, values)
    logger.log_wandb(dict)
    
# def write_log(logger, keys, values, use_logger):
#     for i, key in enumerate(keys):
#         logger.run["train/"+key].append(values[i], step=values[0])
    
def to_one_hot_array(value, n_dim):
    array = np.zeros((n_dim,), dtype=int)
    array[value] = 1
    return array

def to_one_hot(y, n_dims=3):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

def get_device_id():
    return GPU_DEVICE

def get_device():
    return torch.device("cuda:"+str(GPU_DEVICE) if torch.cuda.is_available() else "cpu")

def get_boltzman_probability_measure(q_values, temperature):
    q_values = np.array(q_values)
    boltzman_distribution = 1 - (np.exp(-q_values/temperature)/sum(np.exp(-q_values/temperature)))
    return boltzman_distribution


def custom_action_encoding(action, num_action = 5, dim=512):
    if dim % num_action != 0:
        raise ValueError("Action encoding dimension must be divisible by {}".format(num_action))
        
    if torch.is_tensor(action):
        action_np = action.detach().cpu().numpy()
    else:
        action_np = action
        
    def f(x):
        encoding = np.zeros((dim,), dtype = np.float32)
        if x == 0:
            encoding[:dim//num_action] = 1 
        elif x == 1:
            encoding[(dim//num_action)+1:2*(dim//num_action)] = 1
        elif x == 2:
            encoding[2*(dim//num_action)+1:3*(dim//num_action)] = 1 
        else:
            encoding[3*(dim//num_action)+1:dim] = 1
        return encoding
    vf = np.vectorize(f)
    action_encoded = []
    for array in action_np:
         encoded = list(map(vf, array))
         action_encoded.append(encoded)
    return np.array(action_encoded)

def return_legal_actions(env, num_lane):
        vehicle = env.vehicle
        lane = vehicle.lane_index[-1]
        if lane == 0:
            return [1,2,3,4]
        elif lane == num_lane-1:
            return [0,1,3,4]
        else:
            return [0,1,2,3,4]

def preprocess_state(state):
    state_encoding = state['image'][:,:,0].flatten()
    state_direction = state['direction']
    one_hot_direction = to_one_hot(state_direction, 4)
    state = np.concatenate((state_encoding, one_hot_direction))
    return state

def write_video(frames, episode, dump_dir):
    frameSize = (400,60)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    video = cv2.VideoWriter(dump_dir + '/{}.avi'.format(episode), cv2.VideoWriter_fourcc(*'DIVX'), 10, frameSize, isColor=True)
    for img in frames:
        video.write(img)
    video.release()
    
def record_videos(env, video_folder="videos"):
    wrapped = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped

def to_grayscale(rgb):
    """ the rgb should in the format of (height, width, 3) """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

# def standardize(img):
#     channel_min = np.min(img)
#     channel_max = np.max(img)
#     delta = channel_max - channel_min
#     if delta == 0:
#         img = np.zeros_like(img, dtype=np.float)
#     else:
#         img = (img - channel_min) / (channel_max - channel_min)
#     return img

def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    return img

def softmax_temperature(x, tau):
    """ Returns softmax probabilities with temperature tau
        Input:  x -- 1-dimensional array
        Output: s -- 1-dimensional array
    """
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()

def calculate_discounted_reward(rewards, discount_factor):
    discounted_reward = 0
    # Iterate through the rewards in reverse order
    for reward in reversed(rewards):
        # Apply discount factor and accumulate the discounted reward
        discounted_reward = reward + discount_factor * discounted_reward
    return discounted_reward


def softmax_stable(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

# def encode_state(agent_loc, key_flag):
#     state_encoding = np.zeros((5,5), dtype=np.int64)
#     if np.array_equal(agent_loc, np.array([3,1])) and key_flag:
#         state_encoding[3,1] = 2
#     elif np.array_equal(agent_loc, np.array([4,4])):
#         state_encoding[4,4] = 3
#     else:
#         state_encoding[agent_loc[0], agent_loc[1]] = 1
#     return state_encoding.flatten()

def encode_state(agent_loc):
    return agent_loc + np.array([1,1])

    
def preprocess_state_for_inference(state, device):
    """Preprocess state so that actor selects an action."""
    if not isinstance(state, torch.Tensor):
        state = torch.from_numpy(state).float().to(device)
    return state

def flatten_state(state):
    state_encoding = state['image'][:,:,0].flatten()
    state_direction = state['direction']
    one_hot_direction = to_one_hot(state_direction, 4)
    state = np.concatenate((state_encoding, one_hot_direction))
    return state

def create_dump_directory(path):
    str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    dump_dir = os.path.join(path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_{}'.format(str))
    os.makedirs(dump_dir)
    return dump_dir

def plot_heatmap(data, count, dump_dir):
    ax = sns.heatmap(data, linewidth=0.5)
    plt.savefig(dump_dir + '/{}_heatmap.png'.format(count))
    plt.close()
    
def simple_normalization(x):
    return x/np.sum(x)

def get_cordinate_from_one_hot(val_observations_sliced):
    val_observations_sliced = val_observations_sliced.detach().cpu().numpy()
    val_observations_sliced = np.argmax(val_observations_sliced, axis = -1)
    x = np.squeeze(val_observations_sliced % 5)
    y = np.squeeze(np.ceil(val_observations_sliced/5))
    return x+1, y.astype(int)

def plot_graph(reward, episode, dump_dir, str):
    x = range(len(reward))
    y = reward
    fig1 = plt.figure("Figure 1")
    fig1.set_figwidth(16)
    fig1.set_figheight(4)
    plt.scatter(x,y)
    for x_,y_ in zip(x, y):
        plt.annotate("{}".format(x_), (x_,y_))
    plt.title("Reward redistribution")
    plt.xlabel("Time steps")
    plt.ylabel("Reward redistributions")
    plt.axhline(y=0, color='r')
    plt.xticks(np.arange(0, len(reward), 10))
    plt.savefig(dump_dir + '/{}_{}.png'.format(episode, str))
    plt.close(fig1)

# def plot_graph(rewards, hit_history, episode, dump_dir, val_observations_sliced, action):
#     x_cord, y_cord = get_cordinate_from_one_hot(val_observations_sliced)
#     x = range(len(rewards))
#     y = rewards
#     fig1 = plt.figure("Figure 1")
#     fig1.set_figwidth(16)
#     fig1.set_figheight(4)
#     plt.title("Reward redistribution")
#     df = pd.DataFrame({"x":x, 
#                        "y":y,
#                        "colors":hit_history})
#     cmap = plt.cm.viridis
#     norm = plt.Normalize(df['colors'].values.min(), df['colors'].values.max())
#     label = ["Bad state1", "Bad state2", "Bad state3", "Empty", ]
#     for i, dff in df.groupby("colors"):
#         plt.scatter(dff['x'], dff['y'], s=20, c=cmap(norm(dff['colors'])), 
#                     edgecolors='none', label=label[i])
#     for x_,y_, x_cord_, y_cord_, action_ in zip(x, y, x_cord, y_cord, action):
#         plt.annotate("{},{},{}".format(x_cord_, y_cord_, action_), (x_,y_))
#     plt.xlabel("Time steps")
#     plt.ylabel("Reward redistributions")
#     plt.axhline(y=0, color='r')
#     plt.legend()
#     plt.xticks(np.arange(0, len(rewards), 10))
#     plt.savefig(dump_dir + '/{}.png'.format(episode))
#     plt.close(fig1)
    
def load_list(path):
    for file in glob.glob(path + '**/*.pkl', recursive=True):
        with open(file, 'rb') as f:
            if re.search("bad", file):
                bad_list = pickle.load(f)
            elif re.search("good", file):
                good_list = pickle.load(f)
            elif re.search("intermediate", file):
                intermediate_list = pickle.load(f)
            else:
                raise ValueError("Trajectory data not found")
    return good_list, bad_list

def get_state_one_hot(loc, size=5):
    state_encoding = np.zeros((size,size), dtype=np.float)
    state_encoding[loc[0], loc[1]] = 1
    return state_encoding.flatten(order = 'C')
    
def plot(y_, rewards, redistributed_location_loc, episode):
    y = np.clip(y_, 0, max(y_))
    rewards = -np.array(rewards)
    reward = np.clip(rewards, 0, 1)
    # y[y==0] = np.nan
    # y_mean = np.nanmean(y)
    # minimum = (y_mean + np.max(y_))/2
    # y_ = np.clip(y_, minimum, max(y_))
    x = range(1, len(y_)+1)
    plt.plot(x, y)
    plt.plot(x, reward, color="green")
    plt.xticks(np.arange(0, 51, 3))
    for i, txt in enumerate(redistributed_location_loc):
        plt.annotate(txt, (x[i], y_[i]))
    plt.title("Reward redistribution")
    plt.legend(["RUDDER", "ORIGINAL"])
    plt.savefig("plots/Heirarchial/{}_redistribution.png".format(episode))
    plt.close()
    y = [1 if k>0 else 0 for k in y]
    confusion_matrix = metrics.confusion_matrix(reward, y, labels=[0,1])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.savefig("plots/Heirarchial/{}_confusion_matrix.png".format(episode))
    plt.close()

def dump_list(list, path):
    with open(path, 'wb') as f:
        pickle.dump(list, f)
    
def save_models(checkpoint, saved_dir, prefix):
        """
        Save current model
        :param checkpoint: the parameters of the models, see example in pytorch's documentation: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        :param is_snapshot: whether saved in the snapshot directory
        :param prefix: the prefix of the file name
        :param postfix: the postfix of the file name (can be episode number, frame number and so on)
        """

        path = saved_dir + '/' + prefix + '.tar'
        print("[INFO] DQN model saved succesfully to ", path)
        torch.save(checkpoint, path)

def plot_reconstructed_images(images, reconstructed_images, dir):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap='gray_r')
        reconstructed_image = reconstructed_image.squeeze()
        #cv2.imwrite("reconstructed.jpg", reconstructed_image*255)
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap='gray_r')
    plt.savefig(dir + "/reconstructed.jpg")


def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()
    
def return_valid_actions(state):
    x, y = state[0], state[1]
    if x == 0:
        if y == 0:
            return [0,1]
        elif y == 4:
            return [0,3]
        else:
            return [0,1,3]
    elif x == 4:
        if y == 0:
            return [1,2]
        elif y == 4:
            return [2,3]
        else:
            return [1,2,3]
    elif y == 0:
        return [0,1,2]
    elif y == 4:
        return [0,2,3] 
    else:
        return [0,1,2,3]
        

class W_BCEWithLogitsLoss(torch.nn.Module):
    
    def __init__(self, w_p = None, w_n = None):
        super(W_BCEWithLogitsLoss, self).__init__()
        
        self.w_p = w_p
        self.w_n = w_n
        
    def forward(self, logits, labels, epsilon = 1e-7):
        
        ps = torch.sigmoid(logits.squeeze()) 
        
        loss_pos = -1 * torch.mean(self.w_p * labels * torch.log(ps + epsilon))
        loss_neg = -1 * torch.mean(self.w_n * (1-labels) * torch.log((1-ps) + epsilon))
        
        loss = loss_pos + loss_neg
        
        return loss
    
def range_map(v, x, y):
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])