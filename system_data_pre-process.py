import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import math
import os

def basicPreprocess(combination_data):
    # Delete empty columns
    data1=combination_data.dropna(axis=1,how='all')
    # Replace ',' in numbers with '.'
    data2=data1.replace(to_replace=r',', value='.',regex=True)
    # Change type of data to numeric
    data3=data2.astype(float)
    return data3


def basicAnalysis(data3):
    variables_character_index = ['start_time', 'end_time', 'duration', 'period', 'period_std', 'mean', 'variance','isconst']
    variables_character_columns = data3.columns[1::2]
    variables_character = pd.DataFrame(index=variables_character_index, columns=variables_character_columns)
    variables_character.at['isconst'] = False

    for i in range(0, len(variables_character_columns)):
        temp_col = data3.iloc[:, i * 2:i * 2 + 2].dropna(axis=0)
        # character regarding collecting period
        temp_period = temp_col.iloc[1:, 0].reset_index().sub(temp_col.iloc[:-1, 0].reset_index(), axis=0).drop(columns=['index'])
        variables_character.at['start_time', variables_character_columns[i]] = temp_col.iloc[0, 0]
        variables_character.at['end_time', variables_character_columns[i]] = temp_col.iloc[-1, 0]
        variables_character.at['duration', variables_character_columns[i]] = temp_col.iloc[-1, 0] - temp_col.iloc[0, 0]
        variables_character.at['period', variables_character_columns[i]] = temp_period.mean()[0]
        variables_character.at['period_std', variables_character_columns[i]] = temp_period.std()[0]
        if temp_period.std()[0] > temp_period.mean()[0] * 0.2:
            print('check missing period for' + str(variables_character_columns[i]))

        # character regarding variable
        variables_character.at['mean', variables_character_columns[i]] = temp_col.mean()[1]
        variables_character.at['variance', variables_character_columns[i]] = temp_col.std()[1]
        if temp_col.std()[1] < 0.01:
            variables_character.at['isconst', variables_character_columns[i]] = True
            print(str(variables_character_columns[i]) + 'seems being const')

    return variables_character

def dropData(data3,variables_character):
    # drop data out of [460s-9600s)
    data4=data3.copy(deep=True)
    for i in range(0,int(len(data3.columns)/2)):
        data4.iloc[:int(round(start_point/variables_character.at['period',data3.columns[i*2+1]])),i*2:i*2+2]=np.nan
        data4.iloc[int(round(end_point/variables_character.at['period',data3.columns[i*2+1]])):,i*2:i*2+2]=np.nan
    return data4

def motionlessCapture(data4):
    # motionless status
    FRWS=data4.loc[:,'Backbone1J1939-1.45.0::HRW_X_EBS::FrontAxleRightWheelSpeed[Km_per_h]'].dropna(axis=0)
    zero_count=pd.DataFrame(index=range(len(FRWS)))
    # row index of zero_count range from 0 to len(FRWS).
    # df.at[100,0] refer to the name of index and colomn so it could be the any row that named as 100, while df.iloc[] refers to actual position
    for i in range(len(FRWS)):
        zero_count.at[i,0]=np.sum(FRWS.iloc[max(0,int(i-last_time/2)):min(int(i+last_time/2),len(FRWS))]<= 0.001)
    motionless_time=np.where(zero_count==last_time)[0]
    motionless_time_dense=sorted(list(motionless_time)+[i+0.5 for i in motionless_time])
    # here the motionless_time represent the timestamp after cutting 0-460 and 9600-end.
    # e.g. timestamp 100 is motionless, it correspond to 100/samplingfrequency of FRWS (50) +460, i.e. 462 second in the original dataset.
    return motionless_time,motionless_time_dense


def frequencyProperty(data5,variables_character,motionless_time,motionless_time_dense):
    index = ['motionless_value_mean', 'motionless_value_std', 'maximal_frequency', 'minmal_frequency','sampling_frequency', 'order', 'filter_type']
    columns = data5.columns[1::2]
    frequency_property = pd.DataFrame(index=index, columns=columns)

    FRWS_period = variables_character.at['period', 'Backbone1J1939-1.45.0::HRW_X_EBS::FrontAxleRightWheelSpeed[Km_per_h]']
    for i in range(len(columns)):
        period = variables_character.at['period', data5.columns[i * 2 + 1]]
        if period >= FRWS_period:
            temp_variable = data5.iloc[:, i * 2:i * 2 + 2].dropna().iloc[sorted(list(set((motionless_time / period * FRWS_period).round(0).astype(int)))), :]
        else:
            temp_variable = data5.iloc[:, i * 2:i * 2 + 2].dropna().iloc[sorted(list(set((motionless_time_dense / period * FRWS_period).round(0).astype(int)))), :]
        frequency_property.loc['motionless_value_mean', data5.columns[i * 2 + 1]] = np.mean(temp_variable.iloc[:, -1])
        frequency_property.loc['motionless_value_std', data5.columns[i * 2 + 1]] = np.std(temp_variable.iloc[:, -1])
        frequency_property.loc['sampling_frequency', data5.columns[i * 2 + 1]] = 1 / variables_character.at['period', data5.columns[i * 2 + 1]]
    frequency_property.loc['filter_type'] = 'lowpass'
    return frequency_property

# denoising (frequency domain filter)
def butter_lowpass(lowcut, fs, order=5):
    return signal.butter(order, lowcut, fs=fs, btype='lowpass')

def butter_lowpass_filtfilt(data, lowcut, fs, order=5):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def denoiseInterpolate(data5, frequency_property):
    temp_df = data5.copy(deep=True)
    temp_df.iloc[:, 0::2] = temp_df.iloc[:, 0::2].apply(lambda x: round(x * 100))
    to_concatenate = pd.DataFrame(index=range(1000000))

    gross_weight = temp_df.iloc[:, 0:2].dropna()
    gross_weight.drop_duplicates(subset=gross_weight.columns[0], keep='first', inplace=True)
    gross_weight.iloc[:, 0] = gross_weight.iloc[:, 0].astype(int)
    gross_weight.set_index(gross_weight.columns[0], inplace=True)
    to_concatenate = pd.concat([to_concatenate, gross_weight], axis=1)

    maximal_frequency = [np.nan, 10, 10, 10, 10, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    bias = [np.nan, 0, 0, 0, 0, 0, 0, 0, -0.698934, 0, -0.004627, 0, -0.144189, -0.60913, 0, -0.257702, 0, 0.149287]
    for i in range(1, 18):
        concatenating = temp_df.iloc[:, i * 2:i * 2 + 2].dropna()
        samplingFrequency = frequency_property.loc['sampling_frequency', concatenating.columns[-1]]
        concatenating.iloc[:, 1] = butter_lowpass_filtfilt(concatenating.iloc[:, 1], maximal_frequency[i],
                                                           fs=samplingFrequency, order=9) - bias[i]
        concatenating.drop_duplicates(subset=concatenating.columns[0], keep='first', inplace=True)
        concatenating.iloc[:, 0] = concatenating.iloc[:, 0].astype(int)
        concatenating.set_index(concatenating.columns[0], inplace=True)
        to_concatenate = pd.concat([to_concatenate, concatenating], axis=1)

    drivelineEngaged = temp_df.iloc[:, -2:].dropna()
    drivelineEngaged.drop_duplicates(subset=drivelineEngaged.columns[0], keep='first', inplace=True)
    drivelineEngaged.iloc[:, 0] = drivelineEngaged.iloc[:, 0].astype(int)
    drivelineEngaged.set_index(drivelineEngaged.columns[0], inplace=True)
    to_concatenate = pd.concat([to_concatenate, drivelineEngaged], axis=1)

    to_concatenate = to_concatenate.iloc[start_point * 100:end_point * 100, :]
    to_concatenate.interpolate(method='linear', axis=0, inplace=True)

    # Input: load fraction, steering angle, brake pedal position, acceleration pedal position
    # combination weight has been transformed to load fraction.
    # Here steering angle is divided by 18.6, pedal position is scaled to 0-1

    # Output: acceleration x and acceleration y, yaw, roll, and pitch rate in front and rear part of tractor.

    to_concatenate.loc[:, 'Backbone1J1939-1.45.0::VDC2_X_EBS::SteeringWheelAngle[rad]'] = to_concatenate.loc[:,'Backbone1J1939-1.45.0::VDC2_X_EBS::SteeringWheelAngle[rad]'].apply(lambda x: x / steer_transformation)
    to_concatenate.loc[:, 'Backbone1J1939-1.45.0::VMCU_BB1_01P::BrakePedalPosition[Percent]'] = to_concatenate.loc[:,'Backbone1J1939-1.45.0::VMCU_BB1_01P::BrakePedalPosition[Percent]'].apply(lambda x: x / 100)
    to_concatenate.loc[:,'Backbone1J1939-1.45.0::VMCU_BB1_01P::AcceleratorPedalPosition1[Percent]'] = to_concatenate.loc[:,'Backbone1J1939-1.45.0::VMCU_BB1_01P::AcceleratorPedalPosition1[Percent]'].apply(lambda x: x / 100)

    # uniform variables name
    name_reset = {}
    name_reset['Backbone1J1939-1.45.0::HRW_X_EBS::FrontAxleRightWheelSpeed[Km_per_h]'] = 'sensorFrontRightSpeed[Km_per_h]'
    name_reset['Backbone1J1939-1.45.0::HRW_X_EBS::RearAxleLeftWheelSpeed[Km_per_h]'] = 'sensorRearLeftSpeed[Km_per_h]'
    name_reset['Backbone1J1939-1.45.0::HRW_X_EBS::RearAxleRightWheelSpeed[Km_per_h]'] = 'sensorRearRightSpeed[Km_per_h]'
    name_reset['Backbone1J1939-1.45.0::HRW_X_EBS::FrontAxleLeftWheelSpeed[Km_per_h]'] = 'sensorFrontLeftSpeed[Km_per_h]'
    name_reset['Backbone1J1939-1.45.0::CVW_X_EBS::GrossCombinationVehicleWeight[kg]'] = 'payloadFraction'
    name_reset['Backbone1J1939-1.45.0::VDC2_X_EBS::SteeringWheelAngle[rad]'] = 'steer11'
    name_reset['Backbone1J1939-1.45.0::VMCU_BB1_01P::BrakePedalPosition[Percent]'] = 'brakePosition[percent]'
    name_reset['Backbone1J1939-1.45.0::VMCU_BB1_01P::AcceleratorPedalPosition1[Percent]'] = 'acceleratorPosition[percent]'
    name_reset['Backbone1J1939-1.45.0::ETC1_X_TECU::TransmissionDrivelineEngaged'] = 'drivelineEngaged'
    name_reset['VANCANA::IMU_Chassi_Front_ID1::YWRT[�/s]'] = 'IMUFrontYWRT[d/s]'
    name_reset['VANCANA::IMU_Chassi_Front_ID1::ACLNY[g]'] = 'IMUFrontACLNY[g]'
    name_reset['VANCANA::IMU_Chassi_Front_ID4::RRT[�/s]'] = 'IMUFrontRRT[d/s]'
    name_reset['VANCANA::IMU_Chassi_Front_ID4::ACLNX[g]'] = 'IMUFrontACLNX[g]'
    name_reset['VANCANA::IMU_Chassi_Front_ID7::PRT[�/s]'] = 'IMUFrontPRT[d/s]'
    name_reset['VANCANB::IMU_Chassi_Rear_ID1::YWRT[�/s]'] = 'IMURearYWRT[d/s]'
    name_reset['VANCANB::IMU_Chassi_Rear_ID1::ACLNY[g]'] = 'IMURearACLNY[g]'
    name_reset['VANCANB::IMU_Chassi_Rear_ID4::RRT[�/s]'] = 'IMURearRRT[d/s]'
    name_reset['VANCANB::IMU_Chassi_Rear_ID4::ACLNX[g]'] = 'IMURearACLNX[g]'
    name_reset['VANCANB::IMU_Chassi_Rear_ID7::PRT[�/s]'] = 'IMURearPRT[d/s]'
    to_concatenate.rename(columns=name_reset, inplace=True)

    # modify acceleration pedal position
    to_concatenate.loc[to_concatenate['drivelineEngaged'] == 0, 'acceleratorPosition[percent]'] = 0
    return to_concatenate

def saveData(to_concatenate):
    data_path = 'dataset_drivelineEngaged'

    deduplicated_motionless_time = list(motionless_time)
    deduplicated_motionless_time_2 = deduplicated_motionless_time.copy()
    for i in range(len(motionless_time) - 1):
        if motionless_time[i] == motionless_time[i + 1] - 1:
            deduplicated_motionless_time.remove(motionless_time[i])
            deduplicated_motionless_time_2.remove(motionless_time[i + 1])
    final_motionless = [deduplicated_motionless_time[i] + deduplicated_motionless_time_2[i] for i in
                        range(len(deduplicated_motionless_time))]

    data_path = 'dataset_drivelineEngaged'
    dataset = []
    for i in range(len(final_motionless) - 1):
        dataset.append(to_concatenate.iloc[final_motionless[i]:final_motionless[i + 1], :])
    for i in range(len(dataset)):
        dataset[i].to_csv(save_path + '/dataset_' + str(i) + '.csv')


if __name__ == '__main__':
    # conctant variables
    start_point = 460
    end_point = 9600
    last_time = 200
    vehicle_weight = 8932.3
    full_load = 24000
    steer_transformation = 18.6
    path = os.getcwd()
    orignaldata_path = os.path.abspath(os.path.join(path, os.pardir)) + '\\data\\01_original_dataset'
    save_path=os.path.abspath(os.path.join(path, os.pardir)) + '\\data\\02_preprocessed_dataset'

    # load dataset collected from vehicle
    original_data = pd.read_csv(orignaldata_path + '/test_data_minimal_1.csv', header=0, sep=';')
    driveline_engaged = pd.read_csv(orignaldata_path + '/TransmissionDrivelineEngaged.csv', header=0, sep=';')
    driveline_engaged.rename(columns={'Time[s]': 'time38'}, inplace=True)
    combination_data = pd.concat([original_data, driveline_engaged], axis=1)

    # Basic Pre-process
    data3=basicPreprocess(combination_data)

    # basic analysis
    variables_character=basicAnalysis(data3)

    # drop imperfect data
    data4 =dropData(data3,variables_character)

    # motionless timestamp
    motionless_time,motionless_time_dense=motionlessCapture(data4)

    # columns_position tells the position of desired variables among the whole dataset, need to be manually set
    columns_position=[0,5,6,7,8,10,12,13,22,23,24,25,26,29,30,31,32,33,38]
    data5=data4.iloc[:,sorted([i*2 for i in columns_position]+[i*2+1 for i in columns_position])]
    data5.loc[:,'Backbone1J1939-1.45.0::CVW_X_EBS::GrossCombinationVehicleWeight[kg]']=data5.loc[:,'Backbone1J1939-1.45.0::CVW_X_EBS::GrossCombinationVehicleWeight[kg]'].apply(lambda x: (x-vehicle_weight)/full_load)

    # frequency property
    frequency_property=frequencyProperty(data5,variables_character,motionless_time,motionless_time_dense)

    # Denoise and Interpolate
    to_concatenate=denoiseInterpolate(data5,frequency_property)

    # save data
    saveData(to_concatenate)