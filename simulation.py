from json import load
from re import S
import pyfmi
import os
import numpy as np
from random import uniform
import matplotlib.pyplot as plt
import pandas as pd
import glob


def simulate(vehicle_data, brake=False):
    m1 = pyfmi.load_fmu('FH2000_2.fmu')

    # set constant inputs for friction and road input
    m1.set('friction[1]', 0.7)
    m1.set('friction[2]', 0.7)
    m1.set('friction[3]', 0.7)
    m1.set('friction[4]', 0.7)
    m1.set('friction[5]', 0.7)
    m1.set('friction[6]', 0.7)
    m1.set('friction[7]', 0.7)
    m1.set('friction[8]', 0.7)
    m1.set('friction[9]', 0.7)
    m1.set('friction[10]', 0.7)

    # z
    m1.set('z[1]', 0)
    m1.set('z[2]', 0)
    m1.set('z[3]', 0)
    m1.set('z[4]', 0)
    m1.set('z[5]', 0)
    m1.set('z[6]', 0)
    m1.set('z[7]', 0)
    m1.set('z[8]', 0)
    m1.set('z[9]', 0)
    m1.set('z[10]', 0)

    # dzdx
    m1.set('dz[1,1]', 0)
    m1.set('dz[2,1]', 0)
    m1.set('dz[3,1]', 0)
    m1.set('dz[4,1]', 0)
    m1.set('dz[5,1]', 0)
    m1.set('dz[6,1]', 0)
    m1.set('dz[7,1]', 0)
    m1.set('dz[8,1]', 0)
    m1.set('dz[9,1]', 0)
    m1.set('dz[10,1]', 0)

    # dzdy
    m1.set('dz[1,2]', 0)
    m1.set('dz[2,2]', 0)
    m1.set('dz[3,2]', 0)
    m1.set('dz[4,2]', 0)
    m1.set('dz[5,2]', 0)
    m1.set('dz[6,2]', 0)
    m1.set('dz[7,2]', 0)
    m1.set('dz[8,2]', 0)
    m1.set('dz[9,2]', 0)
    m1.set('dz[10,2]', 0)

    m1.set('gearSelection', 4)  # 1 2 3 4 <-> P R N D
    # park reverse n drive

    ##################################################################################################################
    ################################### For the case of open loop simulation #########################################
    ##################################################################################################################

    # create input structure. Here I will redeclare the speed request and frictions as an input array
    ''' 49 alternative input:
    friction[1]-friction[10], z[1]-z[10], dz[1,1]-dz[10,1], dz[1,2]-dz[10,2],
    acceleration, acceleratorPosition, brakePosition, deceleration, gearSelection, payloadFraction, steer11, torque, speed.
    all defined except for acceleration, deceleration, acceleratorPosition, torque
    '''

    load = vehicle_data.loc[:, 'payloadFraction']
    # steering wheel angle
    SWA = vehicle_data.loc[:, 'steer11']
    # brake pedal position
    BPP = vehicle_data.loc[:, 'brakePosition[percent]']
    # acceleration pedel position
    # APP=vehicle_data.loc[:,'acceleratorPosition[percent]']
    speed = vehicle_data[['sensorFrontLeftSpeed[Km_per_h]', 'sensorFrontRightSpeed[Km_per_h]', 'sensorRearLeftSpeed[Km_per_h]' , 'sensorRearRightSpeed[Km_per_h]']].sum(axis=1) / 3.6 / 4

    t = np.linspace(0., (len(load) - 1) / 100.0, len(load))

    if brake == False:
        u_traj = np.transpose(np.vstack((t, load, SWA, speed)))
        input_string = ['payloadFraction', 'steer11', 'speed']
        input_object = (input_string, u_traj)
    else:
        u_traj = np.transpose(np.vstack((t, load, SWA, BPP, speed)))
        input_string = ['payloadFraction', 'steer11', 'brakePosition', 'speed']
        input_object = (input_string, u_traj)

    # set the first input
    m1.set(input_string, u_traj[0, :])

    # simulate (open loop simulation)
    # print((len(load)-1)/100.0)
    res = m1.simulate(final_time=(len(load) - 1) / 100.0, input=input_object, options={'ncp': len(load) - 1})

    return res


def readRes(data, res):
    g = 9.8
    front_yaw_res = res["vTMOutputs.tractorOutputs.frontIMU.Local_CS_angular_rates.yawrate"] * 180 / np.pi
    rear_yaw_res = res["vTMOutputs.tractorOutputs.rearIMU.Local_CS_angular_rates.yawrate"] * 180 / np.pi
    front_pitch_res = res["vTMOutputs.tractorOutputs.frontIMU.Local_CS_angular_rates.pitchrate"] * 180 / np.pi
    rear_pitch_res = res["vTMOutputs.tractorOutputs.rearIMU.Local_CS_angular_rates.pitchrate"] * 180 / np.pi
    front_roll_res = res["vTMOutputs.tractorOutputs.frontIMU.Local_CS_angular_rates.rollrate"] * 180 / np.pi
    rear_roll_res = res["vTMOutputs.tractorOutputs.rearIMU.Local_CS_angular_rates.rollrate"] * 180 / np.pi

    front_acceleration_x_res = res['vTMOutputs.tractorOutputs.frontBody.Local_CS_accelerations.ax'] / g
    front_acceleration_y_res = res['vTMOutputs.tractorOutputs.frontBody.Local_CS_accelerations.ay'] / g
    rear_acceleration_x_res = res['vTMOutputs.tractorOutputs.rearIMU.Local_CS_accelerations.ax'] / g
    rear_acceleration_y_res = res['vTMOutputs.tractorOutputs.rearIMU.Local_CS_accelerations.ay'] / g

    res_front_vx = res['vTMOutputs.tractorOutputs.frontBody.Local_CS_velocities.vx'] * 3.6
    res_front_vy = res['vTMOutputs.tractorOutputs.frontBody.Local_CS_velocities.vy'] * 3.6
    res_rear_vx = res['vTMOutputs.tractorOutputs.rearIMU.Local_CS_velocities.vx'] * 3.6
    res_rear_vy = res['vTMOutputs.tractorOutputs.rearIMU.Local_CS_velocities.vy'] * 3.6

    sim_t = res['time']

    data['resFrontVx[Km_per_h]'] = res_front_vx
    data['resFrontVy[Km_per_h]'] = res_front_vy
    data['resRearVx[Km_per_h]'] = res_rear_vx
    data['resRearVy[Km_per_h]'] = res_rear_vy

    data['resFrontYWRT[d/s]'] = front_yaw_res
    data['resFrontRRT[d/s]'] = front_roll_res
    data['resFrontPRT[d/s]'] = front_pitch_res
    data['resRearYWRT[d/s]'] = rear_yaw_res
    data['resRearRRT[d/s]'] = rear_roll_res
    data['resRearPRT[d/s]'] = rear_pitch_res

    data['resFrontACLNY[g]'] = front_acceleration_y_res
    data['resFrontACLNX[g]'] = front_acceleration_x_res
    data['resRearACLNY[g]'] = rear_acceleration_y_res
    data['resRearACLNX[g]'] = rear_acceleration_x_res

    return (data)

def saveplot():
    fig, ax = plt.subplots(47, 1, figsize=(50, 250))
    for i in range(47):
        data = pd.read_csv(simulation_data_path + '/simulated_dataset_' + str(i) + '.csv')
        ax[i].plot(range(len(data)), data['IMUFrontYWRT[d/s]'], label='sensor front yaw rate [d/s]')
        ax[i].plot(range(len(data)), data['resFrontYWRT[d/s]'], label='fmu front yaw rate [d/s]')
        ax[i].legend()
    plt.savefig(os.path.abspath(os.path.join(path, os.pardir)) + '/plots/simulation_yawRate.png')

if __name__ == '__main__':
    path = os.getcwd()
    orignal_data_path = os.path.abspath(os.path.join(path, os.pardir)) + '\\data\\01_original_dataset'
    processed_data_path = os.path.abspath(os.path.join(path, os.pardir)) + '\\data\\02_preprocessed_dataset'
    simulation_data_path= os.path.abspath(os.path.join(path, os.pardir)) + '\\data\\03_simulation'

    # load dataset collected from vehicle
    original_data = pd.read_csv(orignal_data_path + '/test_data_minimal_1.csv', header=0, sep=';')
    driveline_engaged = pd.read_csv(orignal_data_path + '/TransmissionDrivelineEngaged.csv', header=0, sep=';')
    driveline_engaged.rename(columns={'Time[s]': 'time38'}, inplace=True)
    combination_data = pd.concat([original_data, driveline_engaged], axis=1)


    for i in range(47):  # csv_files[:1]:
        data = pd.read_csv(processed_data_path + '/dataset_' + str(i) + '.csv')
        res = simulate(data,brake=True)
        resdataset=readRes(data,res)
        resdataset.to_csv(simulation_data_path + '/simulated_dataset_' + str(i) + '.csv')
    saveplot()

    # remove the data set where vehicle speed should be opposite number
    '''
    section27_name = simulation_data_path + '/simulated_dataset_27.csv'
    abandon_section_name = simulation_data_path + '/simulated_dataset_27_abandon.csv'
    original_section46_name = simulation_data_path + '/simulated_dataset_46.csv'
    # Renaming the file
    os.rename(section27_name, abandon_section_name)
    os.rename(original_section46_name, section27_name)
    '''

    section27_name = simulation_data_path + '/simulated_dataset_27.csv'
    abandon_section_name = simulation_data_path + '/simulated_dataset_27_abandon.csv'
    os.rename(section27_name, abandon_section_name)

    for i in range(28,47):
        original_name=simulation_data_path + '/simulated_dataset_'+str(i)+'.csv'
        new_name=simulation_data_path + '/simulated_dataset_'+str(i-1)+'.csv'
        os.rename(original_name, new_name)




