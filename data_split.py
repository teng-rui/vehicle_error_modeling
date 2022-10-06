import os
import numpy as np
import random
import pandas as pd
import torch

# random seed
random.seed(10000)
np.random.seed(10000)
torch.manual_seed(10000)

# some const
num_datasets = 46
new_scope = 2
new_min = -1
inputs_variable = ['payloadFraction', 'steer11', 'brakePosition[percent]', 'wheelspeed', 'resFrontVx[Km_per_h]',
                   'resFrontVy[Km_per_h]', 'resRearVx[Km_per_h]', 'resRearVy[Km_per_h]', 'resFrontYWRT[d/s]',
                   'resFrontRRT[d/s]', 'resFrontPRT[d/s]', 'resRearYWRT[d/s]', 'resRearRRT[d/s]', 'resRearPRT[d/s]',
                   'resFrontACLNY[g]', 'resFrontACLNX[g]', 'resRearACLNY[g]', 'resRearACLNX[g]']
path = os.getcwd()
simulation_path = os.path.abspath(os.path.join(path, os.pardir)) + '\\data\\03_simulation'
rnndata_path = os.path.abspath(os.path.join(path, os.pardir)) + '\\data\\04_rnn_learning_dataset'
variable_property = pd.DataFrame(index=['min', 'max'], columns=inputs_variable)


def process_signals(data):
    data.loc[:, 'steer11'] = data['steer11'].clip(-0.7, 0.7)
    data.loc[:, 'payloadFraction'] = data['payloadFraction'].clip(0, 1)
    data.loc[:, 'brakePosition[percent]'] = data['brakePosition[percent]'].clip(0, 1)
    data.loc[:, 'sensorFrontLeftSpeed[Km_per_h]'] = data['sensorFrontLeftSpeed[Km_per_h]'].clip(0, 100)
    data.loc[:, 'sensorFrontRightSpeed[Km_per_h]'] = data['sensorFrontRightSpeed[Km_per_h]'].clip(0, 100)
    data.loc[:, 'sensorRearLeftSpeed[Km_per_h]'] = data['sensorRearLeftSpeed[Km_per_h]'].clip(0, 100)
    data.loc[:, 'sensorRearRightSpeed[Km_per_h]'] = data['sensorRearRightSpeed[Km_per_h]'].clip(0, 100)
    data.loc[:, 'wheelspeed'] = data[['sensorFrontLeftSpeed[Km_per_h]', 'sensorFrontRightSpeed[Km_per_h]',
                                      'sensorRearLeftSpeed[Km_per_h]', 'sensorRearRightSpeed[Km_per_h]']].sum(
        axis=1) / 3.6 / 4
    data.loc[:, 'IMURearYWRT[d/s]'] = data['IMURearYWRT[d/s]'].mul(-1)


def load_data():
    # calculate variable properties for further normalization
    data_concat = pd.DataFrame()

    for i in range(num_datasets):
        data = pd.read_csv(simulation_path + '/simulated_dataset_' + str(i) + '.csv')
        process_signals(data)
        data_concat = pd.concat((data_concat, data), axis=0)

    whole_len = data_concat.shape[0]
    training_len = int(round(whole_len * 0.64))
    val_len = int(round(whole_len * 0.16))
    training_data_concat = data_concat[:training_len]
    val_data_concat = data_concat[training_len:training_len + val_len]
    testing_data_concat = data_concat[training_len + val_len:]

    for variable in inputs_variable:
        variable_property.loc['min', variable] = min(training_data_concat.loc[:, variable])
        variable_property.loc['max', variable] = max(training_data_concat.loc[:, variable])

    # normalization
    training_yawrate = training_data_concat['IMURearYWRT[d/s]']
    training_yawrate_difference = training_data_concat['IMURearYWRT[d/s]'] - training_data_concat[
        'resRearYWRT[d/s]']

    val_yawrate = val_data_concat['IMURearYWRT[d/s]']
    val_yawrate_difference = val_data_concat['IMURearYWRT[d/s]'] - val_data_concat[
        'resRearYWRT[d/s]']

    testing_yawrate = testing_data_concat['IMURearYWRT[d/s]']
    testing_yawrate_difference = testing_data_concat['IMURearYWRT[d/s]'] - testing_data_concat[
        'resRearYWRT[d/s]']

    for variable in inputs_variable:
        max_ = variable_property.loc['max', variable]
        min_ = variable_property.loc['min', variable]
        training_data_concat.loc[:, variable] = (training_data_concat[variable] - min_) / (
                    max_ - min_) * new_scope + new_min
        val_data_concat.loc[:, variable] = (val_data_concat[variable] - min_) / (max_ - min_) * new_scope + new_min
        testing_data_concat.loc[:, variable] = (testing_data_concat[variable] - min_) / (
                    max_ - min_) * new_scope + new_min

    training_inputs = training_data_concat[inputs_variable]
    val_inputs = val_data_concat[inputs_variable]
    testing_inputs = testing_data_concat[inputs_variable]

    training_inputs.to_csv(rnndata_path + '/training_inputs.csv',index=False)
    training_yawrate.to_csv(rnndata_path + '/training_yawrate.csv',index=False)
    training_yawrate_difference.to_csv(rnndata_path + '/training_yawrate_difference.csv',index=False)
    val_inputs.to_csv(rnndata_path + '/val_inputs.csv',index=False)
    val_yawrate.to_csv(rnndata_path + '/val_yawrate.csv',index=False)
    val_yawrate_difference.to_csv(rnndata_path + '/val_yawrate_difference.csv',index=False)
    testing_inputs.to_csv(rnndata_path + '/testing_inputs.csv',index=False)
    testing_yawrate.to_csv(rnndata_path + '/testing_yawrate.csv',index=False)
    testing_yawrate_difference.to_csv(rnndata_path + '/testing_yawrate_difference.csv',index=False)


if __name__ == '__main__':
    load_data()
    variable_property.to_csv(rnndata_path + '/variable_property.csv')
