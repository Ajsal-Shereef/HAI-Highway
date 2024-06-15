import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

class PlotInternal():
    
    """This class plots the internal representation of cnn model and the rudder. Code adapted from
    https://colab.research.google.com/github/Niranjankumar-c/DeepLearning-PadhAI/blob/master/DeepLearning_Materials/6_VisualizationCNN_Pytorch/CNNVisualisation.ipynb#scrollTo=qv-nJbDFuNuN"""
    
    def __init__(self):
        pass
    
    def plot_filters_single_channel_big(self, t):
        #setting the rows and columns
        nrows = t.shape[0]*t.shape[2]
        ncols = t.shape[1]*t.shape[3]
        
        npimg = np.array(t.numpy(), np.float32)
        npimg = npimg.transpose((0, 2, 1, 3))
        npimg = npimg.ravel().reshape(nrows, ncols)
        npimg = npimg.T
        
        fig, ax = plt.subplots(figsize=(ncols/10, nrows/200))
        imgplot = sns.heatmap(npimg, xticklabels=False, yticklabels=False, cmap='gray', ax=ax, cbar=False)
        
    def plot_filters_single_channel(self, t, layer_num, episode, is_dqn):
        #kernels depth * number of kernels
        nplots = t.shape[0]*t.shape[1]
        ncols = 12
        nrows = 1 + nplots//ncols
        
        #convert tensor to numpy image
        npimg = np.array(t.detach().cpu().numpy(), np.float32)
        
        count = 0
        fig = plt.figure(figsize=(ncols, nrows))
    
        #looping through all the kernels in each channel
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                count += 1
                ax1 = fig.add_subplot(nrows, ncols, count)
                npimg = np.array(t[i, j].detach().cpu().numpy(), np.float32)
                npimg = (npimg - np.mean(npimg)) / np.std(npimg)
                npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
                shw = ax1.imshow(npimg)
                plt.colorbar(shw)
                ax1.set_title(str(i) + ',' + str(j))
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
        plt.tight_layout()
        if is_dqn:
            plt.savefig("Internal weight plots/DQN/{}_{}.png".format(layer_num, episode))
        else:
            plt.savefig("Internal weight plots/Rudder/{}_{}.png".format(layer_num, episode))
        plt.close(fig)
        
    def plot_filters_multi_channel(self, t, layer_num, episode):
    
        #get the number of kernals
        num_kernels = t.shape[0]    
    
        #define number of columns for subplots
        num_cols = 12
        #rows = num of kernels
        num_rows = num_kernels
    
        #set the figure size
        fig = plt.figure(figsize=(num_cols,num_rows))
    
        #looping through all the kernels
        for i in range(t.shape[0]):
            ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        
            #for each kernel, we convert the tensor to numpy 
            npimg = np.array(t[i].numpy(), np.float32)
            #standardize the numpy image
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            npimg = npimg.transpose((1, 2, 0))
            ax1.imshow(npimg)
            ax1.axis('off')
            ax1.set_title(str(i))
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
        
        plt.savefig('myimage.png', dpi=100)    
        plt.tight_layout()
        plt.savefig("Internal weight plots/cnn_{}_{}.png".format(layer_num, episode))
        plt.close()

    def plot_weights(self, cnn_model, episode, is_dqn = False, single_channel = True, collated = False):
        for i in range(3):
            weight_tensor = cnn_model.cnn[i].cnn.weight
            if single_channel:            
                if collated:
                    self.plot_filters_single_channel_big(weight_tensor, i, episode, is_dqn)
                else:
                    self.plot_filters_single_channel(weight_tensor, i, episode, is_dqn)
        
            else:
                if weight_tensor.shape[1] == 3:
                    self.plot_filters_multi_channel(weight_tensor, i, episode, is_dqn)
                else:
                    print("Can only plot weights with three channels with single channel = False")
         
    def plot_internals(self, episodes, lstm_model, dump_dir):
        lstm_model.plot_internals(filename=dump_dir + "/{}_internals.png".format(episodes), show_plot=False, mb_index=0, fdict=dict(figsize=(8, 8), dpi=100))