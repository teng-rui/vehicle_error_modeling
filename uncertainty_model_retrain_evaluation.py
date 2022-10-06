import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import wandb

import torch
from torch import nn
from torch.utils.data import DataLoader

# random seed
torch.set_printoptions(precision=8)
torch.autograd.set_detect_anomaly(True)
random.seed(10000)
np.random.seed(10000)
torch.manual_seed(10000)

path = os.getcwd()
rnndata_path = os.path.abspath(os.path.join(path, os.pardir)) + '\\data\\04_rnn_learning_dataset'


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
if device == "cuda":
    torch.cuda.empty_cache()


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, internal_layers=1, hidden_dim=8, input_dim=18, output_dim=2, dropout=0.2):
        super(NeuralNetwork, self).__init__()
        self.internal_layers = internal_layers  # num of layers for each LSTM
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # dim of hidden states
        self.output_dim = output_dim
        self.dropout = dropout

        self.RNN1 = nn.LSTM(self.input_dim, self.hidden_dim, self.internal_layers, batch_first=True,
                            dropout=self.dropout).double()
        self.linear1 = nn.Linear(self.hidden_dim, self.output_dim).double()
        self.hidden1 = self.init_hidden()  # initialize hidden states at 0

    def forward(self, x, initial_hidden=False, trim=False):
        if initial_hidden:
            self.hidden1 = self.init_hidden(size=x.batch_sizes[0])

        if trim:
            self.hidden1 = self.trim_hidden(size=x.batch_sizes[0])

        x, self.hidden1 = self.RNN1(x, self.hidden1)
        self.hidden1 = (self.hidden1[0].detach(), self.hidden1[1].detach())
        x = self.squash_packed(x, fn=self.linear1)

        return x

    def init_hidden(self, size=64):
        # Before training, we dont have any hidden state.
        # Dim of hidden states: (num of layers, size of batch(batchsize), length of hidden state of RNN cell)
        return (torch.zeros(self.internal_layers, size, self.hidden_dim).double().detach().to(device),
                torch.zeros(self.internal_layers, size, self.hidden_dim).double().detach().to(device))

    def trim_hidden(self, size=64):
        # hidden state transmit among batches, size of last batch might differ from that of next batch, so hidden state need to be trimmed.
        # Dim of hidden states: (num of layers, size of batch(batchsize), length of hidden state of RNN cell)
        self.hidden1 = (self.hidden1[0][:, :size, :], self.hidden1[1][:, :size, :])
        # print(self.hidden1.shape)

    def init_hidden_olayer(self, size=64):
        # last layer has different dim for hidden states.
        return (torch.zeros(self.internal_layers, size, self.output_dim).double().detach().to(device),
                torch.zeros(self.internal_layers, size, self.output_dim).double().detach().to(device))

    def squash_packed(self, x, fn=nn.functional.relu):
        return torch.nn.utils.rnn.PackedSequence(fn(x.data), x.batch_sizes,
                                                 x.sorted_indices, x.unsorted_indices)


def pd_to_torch(data):
    data=torch.from_numpy(data.values)
    return data

def load_data():

    # *_yawrate and *_yawrate_difference is in the form of two dim torch with shape (len,1)
    training_inputs=pd_to_torch(pd.read_csv(rnndata_path + '/training_inputs.csv'))
    training_yawrate=pd_to_torch(pd.read_csv(rnndata_path + '/training_yawrate.csv'))
    training_yawrate_difference=pd_to_torch(pd.read_csv(rnndata_path + '/training_yawrate_difference.csv'))
    val_inputs=pd_to_torch(pd.read_csv(rnndata_path + '/val_inputs.csv'))
    val_yawrate=pd_to_torch(pd.read_csv(rnndata_path + '/val_yawrate.csv'))
    val_yawrate_difference=pd_to_torch(pd.read_csv(rnndata_path + '/val_yawrate_difference.csv'))

    training_inputs=torch.concat([training_inputs,val_inputs],axis=0)
    training_yawrate=torch.concat([training_yawrate,val_yawrate],axis=0)
    training_yawrate_difference=torch.concat([training_yawrate_difference,val_yawrate_difference],axis=0)
    testing_inputs=pd_to_torch(pd.read_csv(rnndata_path + '/testing_inputs.csv'))
    testing_yawrate=pd_to_torch(pd.read_csv(rnndata_path + '/testing_yawrate.csv'))
    testing_yawrate_difference=pd_to_torch(pd.read_csv(rnndata_path + '/testing_yawrate_difference.csv'))

    return training_inputs,training_yawrate,training_yawrate_difference,testing_inputs,testing_yawrate,testing_yawrate_difference

def train(training_data, training_target, sequenceLength, model, optimizer):
    size = training_data[0].shape[0]
    model.train()
    loss_logp_fn = nn.GaussianNLLLoss()
    i = 0

    while (i < size):
        X = [torch.tensor(item[i:i + sequenceLength, :]).clone().detach().requires_grad_(True) for item in
             training_data]
        Y = [torch.tensor(item[i:i + sequenceLength, :]).clone().detach().requires_grad_(True) for item in
             training_target]
        X = torch.nn.utils.rnn.pack_sequence(X).to(device)
        Y = torch.nn.utils.rnn.pack_sequence(Y).to(device)

        # Compute prediction error
        if i == 0:
            pred = model(X, initial_hidden=True).data
        else:
            pred = model(X).data

        mu = pred[:, 0]
        sigma = torch.log(1 + torch.exp(pred[:, 1]))
        loss= loss_logp_fn(mu, Y.data[:, 0], sigma)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        i += sequenceLength


def evaluate(training_inputs, training_targets,testing_inputs, testing_targets, model):
    loss_logp_fn = nn.GaussianNLLLoss()
    model.eval()

    print('evaluation on training set')
    inputs_ = [training_inputs]
    res = model(torch.nn.utils.rnn.pack_sequence(inputs_).to(device), initial_hidden=True).data.detach().cpu()
    mu = res[:, 0]
    sigma = torch.log(1 + torch.exp(res[:, 1]))
    mse_rnn_training = torch.mean(np.square(training_targets[:, 0] - mu))
    logp_rnn_training = loss_logp_fn(mu, training_targets[:, 0], sigma)

    print('evaluation on validation set')
    inputs_ = [testing_inputs]
    res = model(torch.nn.utils.rnn.pack_sequence(inputs_).to(device), initial_hidden=True).data.detach().cpu()
    mu = res[:, 0]
    sigma = torch.log(1 + torch.exp(res[:, 1]))
    mse_rnn_val = torch.mean(np.square(testing_targets[:, 0] - mu))
    logp_rnn_val = loss_logp_fn(mu, testing_targets[:, 0], sigma)
    return mse_rnn_training, mse_rnn_val, logp_rnn_training, logp_rnn_val


def iterate_train(model, optimizer, training_inputs, training_targets, testing_inputs, testing_targets, batchsize,
                  sequenceLength, epochs, k):
    size_per_batch = training_inputs.shape[0] // batchsize
    training_input_batch = np.split(training_inputs[:size_per_batch * batchsize, :], batchsize)
    training_target_batch = np.split(training_targets[:size_per_batch * batchsize, :], batchsize)

    # split training set to k folds and iterate following functions for k times, getting k performance for each configuration
    for epoch in range(epochs):
        print('epoch', epoch)
        train(training_input_batch, training_target_batch, sequenceLength, model, optimizer)

    mse_rnn_training, mse_rnn_val, logp_rnn_training, logp_rnn_val = evaluate(training_inputs, training_targets,testing_inputs, testing_targets, model=model)
    print(mse_rnn_training, mse_rnn_val, logp_rnn_training, logp_rnn_val)
    torch.save(model.state_dict(), '../models/uncertainty_aware_batch/retrain_epoch_' + str(epochs) + experiment_name + '.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some hyper-params')
    # parameters that define the neural network
    parser.add_argument('--layers', type=int, default=2, help='num of LSTM layers')
    parser.add_argument('--nodes', type=int, default=128, help='num of LSTM nodes at each layer')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability')
    # parameters concerning data process
    parser.add_argument('--batchsize', type=int, default=8,
                        help='batchsize: num of samples that are propagated through the network')
    parser.add_argument('--sequence_length', type=int, default=128,
                        help='the length of the sequence of input data, ranging from 5 to 500')
    parser.add_argument('--CV', type=int, default=5, help='num of fold for cross-validation')
    parser.add_argument('--epochs', type=int, default=40, help='num of training epochs')

    # other parameters
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate, step size at each iteration while moving toward a minimum of a loss function')
    parser.add_argument('--target', type=str, default='difference')
    args = parser.parse_args()

    experiment_name = f"target_{args.target}_layers_{args.layers}_nodes_{args.nodes}_dropout_{args.dropout}_batchsize_{args.batchsize}_sequencelength_{args.sequence_length}_learningrate_{args.lr}"

    # wandb.init(project='point_prediction_error_model',config=vars(args),name=experiment_name)

    # initialize models
    model = NeuralNetwork(internal_layers=args.layers, hidden_dim=args.nodes, input_dim=18, output_dim=2,
                          dropout=args.dropout).to(device)
    # print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # load data
    training_inputs,training_yawrate,training_yawrate_difference,testing_inputs,testing_yawrate,testing_yawrate_difference = load_data()



    # model training and evaluating
    if args.target == 'yawrate':
        training_targets = training_yawrate
        testing_targets = testing_yawrate
    else:
        training_targets = training_yawrate_difference
        testing_targets = testing_yawrate_difference

    training_kwargs = {"model": model,
                       "optimizer": optimizer,
                       "training_inputs": training_inputs,
                       "training_targets": training_targets,
                       "testing_inputs": testing_inputs,
                       "testing_targets": testing_targets,
                       "batchsize": args.batchsize,
                       "sequenceLength": args.sequence_length,
                       "epochs": args.epochs,
                       "k": args.CV}

    iterate_train(**training_kwargs)
