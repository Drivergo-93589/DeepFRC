import numpy as np
import pandas as pd
import numpy as np
import os


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import lr_scheduler
import time


# Plotting Packages
import matplotlib.pyplot as plt
import seaborn as sbn
import matplotlib as mpl

from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import norm


class SrvfClasNet(nn.Module):
    def __init__(self, hidden_to_fc, fun_len, Interp1d, device):  # batch_size,
        super(SrvfClasNet, self).__init__()
        """
        batch_size : The batch size of input data
        fun_len : The length of time seris data --> 150
        
        """
        #### the Learnable Pre-warping Block ####

        self.localization = nn.Sequential(
            nn.Conv1d(1, 16, 3),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, 3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AvgPool1d(8),
        )
        self.Interp1d = Interp1d
        self.fun_len = fun_len
        ########################################################################
        #### Here we use FC layer to make sure the output of                ####
        #### the Learnable Pre-warping Block is equal to T = fun_len  ####
        self.fc = nn.Linear(hidden_to_fc, fun_len)
        ########################################################################
        self.device = device

    def warp(self, v):
        batch, length = v.size()
        r = torch.zeros_like(v)
        for i in range(length):
            r[:, i] = torch.trapz(v[:, : i + 1], dim=1) / torch.trapz(v, dim=1)
        return r

    def smooth_integral(self, v):
        batch, length = v.size()
        r = torch.zeros_like(v)
        for i in range(length):
            r[:, i] = torch.trapz(v[:, : i + 1], dim=1) / torch.trapz(v, dim=1)
        return r

    def gradient(self, f):
        """
        Central difference method with equal space in dx(i.e.1)
        Input : f in tensor (batch size x fun len)
        Output : gradient along axis fun len(dim=1)

        """
        row_num = f.ndim  # y.size()[0]
        dx = torch.arange(f.size()[1])  # only 1d
        output = torch.empty_like(f)
        # i=0
        output[:, 0] = torch.div(
            f[:, 1] - f[:, 0], dx[1] - dx[0]
        )  # y[:,1] - y[:,0]/ dx[1]-dx[0]
        # i = end
        output[:, -1] = torch.div(
            f[:, -1] - f[:, -2], dx[-1] - dx[-2]
        )  # y[:,-1] - y[:,-2]/  dx[-1]-dx[-2]
        # i = 1:end-1
        for i in range(1, len(dx) - 1):
            output[:, i] = torch.div(f[:, i - 1] - f[:, i + 1], dx[i - 1] - dx[i + 1])

        return output

    def srvf(self, v, gradient):
        """
        v (batch size x fun len )
        q = sign(f')* sqrt(|f'|)
        No backward
        """
        f = v.detach()
        q = torch.empty_like(f)
        batch, length = f.size()
        length_adjust = length - 1
        grad_f = gradient(f * length_adjust)
        for i in range(batch):
            q[i, :] = torch.sign(grad_f[i, :]) * torch.sqrt(grad_f[i, :].abs())
        return q

    def Is_strictly_increasing(self, r):
        """
        r (batch size x fun len)
        """
        rr = r.detach()
        for i in range(rr.size()[0]):
            if not all(x < y for x, y in zip(rr[i, :], rr[i, 1:])):
                print(rr[i, :])
                print(
                    "The {}'th warping function violates monotonically increasing".format(
                        i
                    )
                )
                break
        print("Screen monotonical increasing assumptions done!")

    def forward(self, input_f):
        ################# Learnable Pre-warp Block ##################
        # batch_size,_,fun_len = input_f.size()
        v = self.localization(input_f)
        v = v.view(v.size(0), -1)
        # FC Layer
        v = self.fc(v)
        # Now v is the output of the Diffeomorphism Block, which's length is T=150
        # , which is the "g" in paper
        #############################################################

        ################# Diffeomorphism Block ######################
        #### ①. Warping layer ####
        # square v or exponetial
        v_sq = v.pow(2)
        # generate gamma function
        r = self.warp(v_sq)

        #### ②. Smoothing layer ####
        # Smoothing effect 1.double intergral
        r = self.smooth_integral(r)
        # r is the formed "gamma" now
        #############################################################

        #################### Centering Block ########################################
        # Since the gamma is not centered, why not we do it here in the forward pass
        #############################################################################

        #################### Group Action Block #####################
        # Q0(r)
        # interp1d(x, y, r)

        # v.size = (batch_size, T=150)
        T = torch.arange(self.fun_len, device=self.device).repeat(v.size(0), 1)
        length_adjust = self.fun_len - 1
        # 1.
        # q(r)sqrt(r')
        # srvf
        # r'
        grad_r = self.gradient(r * length_adjust)
        q = self.srvf(input_f.squeeze(1), self.gradient)
        #################
        # (qi 。yi) ??
        # 先将r作用到q上 （按照r指示的old/new_t直接移动q曲线）
        q_r = self.Interp1d()(T, q, r * (length_adjust)).to(self.device)
        #################
        # 然后按照论文中的公式，实际的warp后的f所对应的
        # Srvf形式是：Q(f(r)) = q(r) * (grad_r.sqrt())
        # 即这样算出的Q等价于先warp f 再算 Srvf
        Q = torch.mul(q_r, grad_r.sqrt())
        # if I'm getting right,
        # Q is the warped data in Srvf space
        # r is the warping function
        # (As it's written in "3.3 Group Action Block")

        # Return the warped Q and the warping function: gamma (not centered)
        # However, here I think we should feed f to the latter network

        # f @ r
        # 将r作用于f,而不是 Q！
        f_r = self.Interp1d()(T, input_f.squeeze(1), r * (length_adjust)).to(self.device)

        return Q, f_r, r
        #############################################################


def Normalization_on_data(input_f):
    """
    input_f N,C=1,L : () torch.tensor

    Output N,C=1,L: After Z normalization
    """
    length = input_f.size()[2]
    input_f = input_f - torch.mean(input_f, dim=2).repeat(1, length).unsqueeze(1)
    input_f = input_f / input_f.std(dim=2, keepdim=True)
    return input_f


def r_mean_inverse(xid, r_m, fun_len):
    """
    Input:
        xid: x time index : 0,1,...,(fun_len-1)
        r_m : The mean of warping functions
        fun_len : The length of functions

    Output:
        r_m_inv : The inverse of mean warping functions
        [0,length-1] scale
    """
    length_adjust = fun_len - 1
    r_m_inv = np.interp(xid, r_m * (length_adjust), xid)
    r_m_inv[-1] = length_adjust
    return r_m_inv


def compute_signal_statistics(input_data):
    input_data_mean = input_data[:, 0, :].mean(dim=0)
    input_data_std = input_data[:, 0, :].std(dim=0)
    lower = input_data_mean - input_data_std
    upper = input_data_mean + input_data_std
    criterion = nn.MSELoss(reduction="mean")
    loss = criterion(
        input_data[:, 0, :], input_data_mean.repeat(input_data.shape[0], 1).detach()
    )
    return (
        input_data_mean.cpu().detach(),
        input_data_std.cpu().detach(),
        lower.cpu().detach(),
        upper.cpu().detach(),
        loss.cpu().detach(),
    )


##################################################################################

localization = nn.Sequential(
    nn.Conv1d(1, 16, 3),
    nn.ReLU(),
    nn.BatchNorm1d(16),
    nn.MaxPool1d(2, 2),
    nn.Conv1d(16, 32, 3),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.MaxPool1d(2, 2),
    nn.Conv1d(32, 64, 3),
    nn.ReLU(),
    nn.BatchNorm1d(64),
    nn.AvgPool1d(8),
)  # .detach()

##################################################################################


def hidden_to_fc_calculation(input_f, localization=localization):
    """
    input_f : The input data
    localization : Layers to generate hidden features
    """
    vv = localization(input_f.cpu().float()).detach()
    vv = vv.view(input_f.size(0), -1)
    return vv.size(-1), input_f.size(-1)


def Generate_centralized_warping_functions(
    input_train_data, model, device, r_mean_inverse=r_mean_inverse
):
    """
    Input
        input data
        r mean inverse function
        devive ; CPU or GPU

    Output
        r, f(r)
    """
    # del batch_data
    # batch_data_train = input_train_data.float().to(device).clone()
    # del batch_data
    # batch_data_test = input_test_data.float().to(device).clone()

    ############################################ Train data ############################################
    with torch.no_grad():
        ########################
        # data_train.shape: (64, 1, 100)

        batch_data = input_train_data

        bat_data = batch_data[:, :, :-1]
        bat_data_shape = bat_data.shape
        # (64, 1, 99)

        # bat_label = batch_data[:, :, [-1]]
        bat_label = batch_data[:, :, -1]
        bat_label_shape = bat_label.shape
        # size: (val_batch_size, 1)
        # (64, 1)
        ########################################
        input_train_data = bat_data
        ########################################
        y_bar, Q, r_train, f_r = model(input_train_data)
        # The r_train returned by forward pass here is “not centered” yet
        ########################

        # r_train_c1 is in [0,1] scale

        r_train_np = r_train.cpu().detach().numpy()
        # gamma_train_c1  = r_train_c1 .detach().numpy()
        batch_size_train, seq_len = r_train.size()

        ####################################### r or r-cen? #######################################
        # r mean
        r_mean_train = r_train_np.mean(axis=0)

        # r mean inverse
        x_index = np.arange(seq_len)
        r_mean_inverse_train = r_mean_inverse(x_index, r_mean_train, seq_len)
        # r_m_inv is in the [0, length-1] scale

        # Perform centering effect on r
        r_train = r_train.cpu().detach()  # *(seq_len-1)
        r_cen_train = torch.zeros_like(r_train)

        for i in range(batch_size_train):
            r_cen_train[i, :] = torch.from_numpy(
                np.interp(r_mean_inverse_train, x_index, r_train[i, :])
            )
        # Now we get the centered gamma： r_cen_train
        ###########################################################################################

        #####################################################
        #################  Apply f@r  #######################
        length_adjust = seq_len - 1
        xid_train = torch.arange(seq_len).repeat(batch_size_train, 1)

        # f @ r_cen
        # 作用于f，（input_train_data），而不是 Q！

        f_r_cen_train = model.srvfNet.Interp1d()(
            xid_train.to(device),
            # input_train_data.float().squeeze(1), --> Note that we have
            # already set "input_train_data = bat_data"
            #
            # Always remember that Srvf didn't take account of Y, and that's
            # what we should be careful using our new loss structure!!
            input_train_data.float().squeeze(1),
            (r_cen_train * length_adjust).to(device),
        )

        # Interp1d calculate $_r where r is applied on some function $
        # q_r = self.Interp1d()(T, q, r * (length_adjust)).to(self.device)

        #####################################################
        return r_cen_train.cpu().detach(), f_r_cen_train.cpu().detach()


def plot_warped_data_analysis(
    tr_label,    # label = 0 or 1
    te_label,
    warped_training_data,
    warped_test_data,
    input_training_data,
    input_test_data,
    data_name,
    r_train,
    r_test,
    compute_signal_statistics=compute_signal_statistics,
):
    # Title
    fig = plt.figure(1, figsize=[90, 50])
    fig.suptitle("{}".format(data_name), fontsize=90, y=1.01)
    # plt.rcParams.update({'font.size': 25})

    # Time index, figure size
    T = np.arange(input_training_data.size()[2])
    plt.style.use("seaborn-dark")

    # https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    plt.rc("xtick", labelsize=30)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=30)

    #################################### Training data original/warped data ###
    plt.subplot(2, 5, 1)
    plt.title(
        "Training data : orginal data  \nsize {} ".format(
            (input_training_data.size()[0], input_training_data.size()[2])
        ),
        fontsize=50,
    )
    for i in range(input_training_data.cpu().size()[0]):
        # when plotting, highlight each class with different color
        if tr_label[i] == 0:
            plt.plot(input_training_data.cpu()[i, 0, :], "orange")
        else:
            plt.plot(input_training_data.cpu()[i, 0, :], "royalblue")
        
    plt.grid()
    plt.tight_layout()
    (
        training_data_mean,
        training_data_std,
        training_lower,
        training_upper,
        training_loss,
    ) = compute_signal_statistics(input_training_data)

    plt.subplot(2, 5, 2)
    plt.title(
        "Training data \n Within class variance:{:.4f}".format(training_loss.item()),
        fontsize=50,
    )
    plt.plot(training_data_mean, "r", label="Average signal")
    t = range(input_training_data.shape[2])
    plt.fill_between(
        t, training_lower, training_upper, color="r", alpha=0.2, label=r"$\pm\sigma$"
    )
    plt.legend(["Averaged signal", "$\mu \pm\sigma$"], fontsize=40)
    plt.tight_layout()

    plt.subplot(2, 5, 3)
    plt.title(
        "Training data : warped data  \nsize {}".format(
            (warped_training_data.size()[0], warped_training_data.size()[1])
        ),
        fontsize=50,
    )
    for i in range(warped_training_data.size()[0]):
        # plt.plot(warped_training_data[i, :])
        # when plotting, highlight each class with different color
        if tr_label[i] == 0:
            plt.plot(warped_training_data.cpu()[i, :], "orange")
        else:
            plt.plot(warped_training_data.cpu()[i, :], "royalblue")
        plt.grid()
        plt.tight_layout()

    #########################################################################################

    (
        training_warped_mean,
        training_warped_std,
        training_warped_lower,
        training_warped_upper,
        training_warped_loss,
    ) = compute_signal_statistics(warped_training_data.unsqueeze(1))

    plt.subplot(2, 5, 4)
    plt.title(
        "Training data: warped data \n Within class variance:{:.4f}".format(
            training_warped_loss.item()
        ),
        fontsize=50,
    )
    plt.plot(training_warped_mean, "b", label="Average signal")
    # t = range(input_training_data.shape[2])
    plt.fill_between(
        t,
        training_warped_lower,
        training_warped_upper,
        color="b",
        alpha=0.2,
        label=b"$\pm\sigma$",
    )
    plt.legend(["Averaged signal", "$\mu \pm\sigma$"], fontsize=40)
    plt.tight_layout()

    ################## train warp ######################
    plt.subplot(2, 5, 5)
    plt.title("Warping functions", fontsize=50)

    r_x = np.linspace(0.0, 1.0, r_train.size(-1))
    for i in range(r_train.size(0)):
        if tr_label[i] == 0:
            plt.plot(r_x, r_train[i, :], "orange")
        else:
            plt.plot(r_x, r_train[i, :], "royalblue")
    ####################################################

    ##########################################

    ############ Test data original/warped data #######################
    plt.subplot(2, 5, 6)
    plt.title(
        "Test data : orginal data  \nsize {} ".format(
            (input_test_data.size()[0], input_test_data.size()[2])
        ),
        fontsize=50,
    )
    for i in range(input_test_data.cpu().size()[0]):
        # plt.plot(input_test_data.cpu()[i, 0, :])
        # when plotting, highlight each class with different color
        if te_label[i] == 0:
            plt.plot(input_test_data.cpu()[i, 0, :], "orange")
        else:
            plt.plot(input_test_data.cpu()[i, 0, :], "royalblue")
    plt.grid()
    plt.tight_layout()
    
    (
        test_data_mean,
        test_data_std,
        test_lower,
        test_upper,
        test_loss,
    ) = compute_signal_statistics(input_test_data)

    plt.subplot(2, 5, 7)
    plt.title(
        "Test data  \n Within class variance:{:.4f}".format(test_loss.item()),
        fontsize=50,
    )
    plt.plot(test_data_mean, "r", label="Average signal")
    t = range(input_test_data.shape[2])
    plt.fill_between(
        t, test_lower, test_upper, color="r", alpha=0.2, label=r"$\pm\sigma$"
    )
    plt.legend(["Averaged signal", "$\mu \pm\sigma$"], fontsize=40)
    plt.tight_layout()

    plt.subplot(2, 5, 8)
    plt.title(
        "Test data : warped data  \nsize {} ".format(
            (warped_test_data.size()[0], warped_test_data.size()[1])
        ),
        fontsize=50,
    )
    for i in range(warped_test_data.size()[0]):
        # plt.plot(warped_test_data[i, :])
        # when plotting, highlight each class with different color
        if te_label[i] == 0:
            plt.plot(warped_test_data.cpu()[i, :], "orange")
        else:
            plt.plot(warped_test_data.cpu()[i, :], "royalblue")
        
        plt.grid()
        plt.tight_layout()

    ########################################################################
    (
        test_warped_mean,
        test_warped_std,
        test_warped_lower,
        test_warped_upper,
        test_warped_loss,
    ) = compute_signal_statistics(warped_test_data.unsqueeze(1))

    plt.subplot(2, 5, 9)
    plt.title(
        "Test data: warped data \n Within class variance:{:.4f}".format(
            test_warped_loss.item()
        ),
        fontsize=50,
    )
    plt.plot(test_warped_mean, "b", label="Average signal")
    # t = range(input_training_data.shape[2])
    plt.fill_between(
        t,
        test_warped_lower,
        test_warped_upper,
        color="b",
        alpha=0.2,
        label=b"$\pm\sigma$",
    )
    plt.legend(["Averaged signal", "$\mu \pm\sigma$"], fontsize=40)
    plt.tight_layout()

    ############# test warp ##################

    plt.subplot(2, 5, 10)
    plt.title("Warping functions", fontsize=50)

    r_x_test = np.linspace(0.0, 1.0, r_test.size(-1))
    for i in range(r_test.size(0)):
        # plt.plot(r_x_test, r_test[i, :])
        if te_label[i] == 0:
            plt.plot(r_x_test, r_test[i, :], "orange")
        else:
            plt.plot(r_x_test, r_test[i, :], "royalblue")
    mpl.rcParams.update(mpl.rcParamsDefault)

    ##########################################


def Simulate_fun_2(n, time_length, lambda_val=4.0, var=1.2, norm=norm, lowess=lowess):
    """
    n: number of functions
    time_length : functions' length
    lambda_val : variance of warping functions
    var : variance of noise
    lowess 'frac' option can control smoothness level
    """

    t = np.linspace(0, 1, time_length)
    g = np.sin(2.2 * np.pi * t) ** 2 + 0.05 * np.cos((2.2 * np.pi * t) + 1.2)

    # lambda_val = 4
    t_bound = abs(np.random.normal(size=(n, 1))) * lambda_val + 0.2
    rnd_idx = np.random.choice(n, int(np.ceil(n / 2)), replace=False)

    gam = np.zeros((time_length, n))
    g_gam = np.zeros((time_length, n))

    for j in range(n):
        t_gam = np.arange(-t_bound[j], t_bound[j], step=0.01)
        t2 = (t_gam + t_bound[j]) / (t_gam[-1] + t_bound[j])
        st = 1 / (1 + np.exp(-t_gam))
        st2 = st - st[0]
        st2 = st2 / st2[-1]
        gam[:, j] = np.interp(t, t2, st2)

    for j in range(len(rnd_idx)):
        gam[:, rnd_idx[j]] = np.interp(t, gam[:, rnd_idx[j]], t)

    # apply time warping first
    # g(\gamma(t))
    for j in range(n):
        g_gam[:, j] = np.interp(gam[:, j], t, g)

    # scale factor
    a = abs(0.2 * np.random.normal(size=(n, 1)) + 1)

    # additive random signal
    e = var  # variance of random signal
    E = np.random.normal(size=(n, len(t))) * e
    # epsilon = gaussian_filter1d(E,1e-10).transpose() # Use Gaussian kernel instead of lowess
    epsilon = np.zeros((time_length, n))
    for j in range(epsilon.shape[1]):
        temp_result = lowess(E[j, :], t, is_sorted=True, frac=0.3, it=0)
        epsilon[:, j] = temp_result[:, 1]
        # epsilon = epsilon[15:len(t)+14,:]
    F_list = a.transpose() * g_gam + epsilon
    return F_list.transpose(), gam.transpose(), g
