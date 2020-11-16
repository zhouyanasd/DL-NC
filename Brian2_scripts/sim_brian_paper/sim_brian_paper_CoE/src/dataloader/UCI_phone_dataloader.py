# -*- coding: utf-8 -*-
"""
    The functions for preparing the data.

:Author: Yan Zhou

:License: BSD 3-Clause, see LICENSE file.
"""

import numpy as np
import scipy as sp
import pandas as pd
import math

from glob import glob

from scipy.signal import medfilt
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
from scipy.fftpack import ifft


class UCI_classification():
    def __init__(self):
        # Creating a dictionary for all types of activities
        # The first 6 activities are called Basic Activities as(BAs) 3 dynamic and 3 static
        # The last 6 activities are called Postural Transitions Activities as (PTAs)
        self.Acitivity_labels = self.AL = {
            1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',  # 3 dynamic activities
            4: 'SITTING', 5: 'STANDING', 6: 'LIYING',  # 3 static activities

            7: 'STAND_TO_SIT', 8: 'SIT_TO_STAND', 9: 'SIT_TO_LIE', 10: 'LIE_TO_SIT',
            11: 'STAND_TO_LIE', 12: 'LIE_TO_STAND',  # 6 postural Transitions
        }
        self.sampling_freq = 50
        self.dt = 0.02  # dt=1/50=0.02s time duration between two rows

    def convert(self, data, label):
        pass

    def loadData(self, path):
        # Scraping RawData files paths
        Raw_data_paths = sorted(glob(path))

        # creating an empty dictionary where all dataframes will be stored
        raw_dic = {}

        # creating list contains columns names of an acc file
        raw_acc_columns = ['acc_X', 'acc_Y', 'acc_Z']

        # creating list contains gyro files columns names
        raw_gyro_columns = ['gyro_X', 'gyro_Y', 'gyro_Z']

        # loop for to convert  each "acc file" into data frame of floats and store it in a dictionnary.
        for path_index in range(0, 61):
            # extracting the file name only and use it as key:[expXX_userXX] without "acc" or "gyro"
            key = Raw_data_paths[path_index][-16:-4]

            # Applying the function defined above to one acc_file and store the output in a DataFrame
            raw_acc_data_frame = self.import_raw_signals(Raw_data_paths[path_index], raw_acc_columns)

            # By shifting the path_index by 61 we find the index of the gyro file related to same experiment_ID
            # Applying the function defined above to one gyro_file and store the output in a DataFrame
            raw_gyro_data_frame = self.import_raw_signals(Raw_data_paths[path_index + 61], raw_gyro_columns)

            # concatenate acc_df and gyro_df in one DataFrame
            raw_signals_data_frame = pd.concat([raw_acc_data_frame, raw_gyro_data_frame], axis=1)

            # Store this new DataFrame in a raw_dic , with the key extracted above
            raw_dic[key] = raw_signals_data_frame

        #################################
        # creating a list contains columns names of "labels.txt" in order
        raw_labels_columns = ['experiment_number_ID', 'user_number_ID', 'activity_number_ID', 'Label_start_point',
                              'Label_end_point']

        # The path of "labels.txt" is last element in the list called "Raw_data_paths"
        labels_path = Raw_data_paths[-1]

        # apply the function defined above to labels.txt
        # store the output  in a dataframe
        Labels_Data_Frame = self.import_labels_file(labels_path, raw_labels_columns)

        freq1 = 0.3  # freq1=0.3 hertz [Hz] the cuttoff frequency between the DC compoenents [0,0.3]
        #           and the body components[0.3,20]hz
        freq2 = 20

        time_sig_dic = {}  # An empty dictionary will contains dataframes of all time domain signals
        raw_dic_keys = sorted(raw_dic.keys())  # sorting dataframes' keys

        for key in raw_dic_keys:  # iterate over each key in raw_dic

            raw_df = raw_dic[key]  # copie the raw dataframe associated to 'expXX_userYY' from raw_dic

            time_sig_df = pd.DataFrame()  # a dataframe will contain time domain signals

            for column in raw_df.columns:  # iterate over each column in raw_df

                t_signal = np.array(raw_df[column])  # copie the signal values in 1D numpy array

                med_filtred = self.median(
                    t_signal)  # apply 3rd order median filter and store the filtred signal in med_filtred

                if 'acc' in column:  # test if the med_filtered signal is an acceleration signal

                    # the 2nd output DC_component is the gravity_acc
                    # The 3rd one is the body_component which in this case the body_acc
                    _, grav_acc, body_acc, _ = self.components_selection_one_signal(med_filtred, freq1,
                                                                               freq2)  # apply components selection

                    body_acc_jerk = self.jerk_one_signal(body_acc)  # apply the jerking function to body components only

                    # store signal in time_sig_dataframe and delete the last value of each column
                    # jerked signal will have the original lenght-1(due to jerking)

                    time_sig_df['t_body_' + column] = body_acc[
                                                      :-1]  # t_body_acc storing with the appropriate axis selected
                    #                                             from the column name

                    time_sig_df['t_grav_' + column] = grav_acc[
                                                      :-1]  # t_grav_acc_storing with the appropriate axis selected
                    #                                              from the column name

                    # store  t_body_acc_jerk signal with the appropriate axis selected from the column name
                    time_sig_df['t_body_acc_jerk_' + column[-1]] = body_acc_jerk

                elif 'gyro' in column:  # if the med_filtred signal is a gyro signal

                    # The 3rd output of components_selection is the body_component which in this case the body_gyro component
                    _, _, body_gyro, _ = self.components_selection_one_signal(med_filtred, freq1,
                                                                         freq2)  # apply components selection

                    body_gyro_jerk = self.jerk_one_signal(body_gyro)  # apply the jerking function to body components only

                    # store signal in time_sig_dataframe and delete the last value of each column
                    # jerked signal will have the original lenght-1(due to jerking)

                    time_sig_df['t_body_gyro_' + column[-1]] = body_gyro[
                                                               :-1]  # t_body_acc storing with the appropriate axis selected
                    #                                                       from the column name

                    time_sig_df['t_body_gyro_jerk_' + column[
                        -1]] = body_gyro_jerk  # t_grav_acc_storing with the appropriate axis
                    #                                                            selected from the column name

            # all 15 axial signals generated above are reordered to facilitate magnitudes signals generation
            new_columns_ordered = ['t_body_acc_X', 't_body_acc_Y', 't_body_acc_Z',
                                   't_grav_acc_X', 't_grav_acc_Y', 't_grav_acc_Z',
                                   't_body_acc_jerk_X', 't_body_acc_jerk_Y', 't_body_acc_jerk_Z',
                                   't_body_gyro_X', 't_body_gyro_Y', 't_body_gyro_Z',
                                   't_body_gyro_jerk_X', 't_body_gyro_jerk_Y', 't_body_gyro_jerk_Z']

            # create new dataframe to order columns
            ordered_time_sig_df = pd.DataFrame()

            for col in new_columns_ordered:  # iterate over each column in the new order
                ordered_time_sig_df[col] = time_sig_df[col]  # store the column in the ordred dataframe

            # Generating magnitude signals
            for i in range(0, 15, 3):  # iterating over each 3-axial signals

                mag_col_name = new_columns_ordered[i][
                               :-1] + 'mag'  # Create the magnitude column name related to each 3-axial signals

                col0 = np.array(ordered_time_sig_df[new_columns_ordered[i]])  # copy X_component
                col1 = ordered_time_sig_df[new_columns_ordered[i + 1]]  # copy Y_component
                col2 = ordered_time_sig_df[new_columns_ordered[i + 2]]  # copy Z_component

                mag_signal = self.mag_3_signals(col0, col1, col2)  # calculate magnitude of each signal[X,Y,Z]
                ordered_time_sig_df[mag_col_name] = mag_signal  # store the signal_mag with its appropriate column name

            time_sig_dic[
                key] = ordered_time_sig_df  # store the ordred_time_sig_df in time_sig_dic with the appropriate key

        return time_sig_dic, Labels_Data_Frame


    def jerk_one_signal(self, signal):
        ##########################################Jerk Signals Functions #####################################

        # d(signal)/dt : the Derivative
        # jerk(signal(x0)) is equal to (signal(x0+dx)-signal(x0))/dt
        # Where: signal(x0+dx)=signal[index[x0]+1] and  signal(x0)=signal[index[x0]]

        # Input: 1D array with lenght=N (N:unknown)
        # Output: 1D array with lenght=N-1
        return np.array([(signal[i + 1] - signal[i]) / self.dt for i in range(len(signal) - 1)])

    def mag_3_signals(self, x, y, z):  # Euclidian magnitude
        return [math.sqrt((x[i] ** 2 + y[i] ** 2 + z[i] ** 2)) for i in range(len(x))]

    def components_selection_one_signal(self, t_signal, freq1, freq2):
        t_signal = np.array(t_signal)
        t_signal_length = len(t_signal)  # number of points in a t_signal

        # the t_signal in frequency domain after applying fft
        f_signal = fft(t_signal)  # 1D numpy array contains complex values (in C)

        # generate frequencies associated to f_signal complex values
        freqs = np.array(
            sp.fftpack.fftfreq(t_signal_length, d=1 / float(self.sampling_freq)))  # frequency values between [-25hz:+25hz]

        # DC_component: f_signal values having freq between [-0.3 hz to 0 hz] and from [0 hz to 0.3hz]
        #                                                             (-0.3 and 0.3 are included)

        # noise components: f_signal values having freq between [-25 hz to 20 hz[ and from ] 20 hz to 25 hz]
        #                                                               (-25 and 25 hz inculded 20hz and -20hz not included)

        # selecting body_component: f_signal values having freq between [-20 hz to -0.3 hz] and from [0.3 hz to 20 hz]
        #                                                               (-0.3 and 0.3 not included , -20hz and 20 hz included)

        f_DC_signal = []  # DC_component in freq domain
        f_body_signal = []  # body component in freq domain numpy.append(a, a[0])
        f_noise_signal = []  # noise in freq domain

        for i in range(len(freqs)):  # iterate over all available frequencies

            # selecting the frequency value
            freq = freqs[i]

            # selecting the f_signal value associated to freq
            value = f_signal[i]

            # Selecting DC_component values
            if abs(freq) > freq1:  # testing if freq is outside DC_component frequency ranges
                f_DC_signal.append(float(
                    0))  # add 0 to  the  list if it was the case (the value should not be added)
            else:  # if freq is inside DC_component frequency ranges
                f_DC_signal.append(value)  # add f_signal value to f_DC_signal list

            # Selecting noise component values
            if (abs(freq) <= freq2):  # testing if freq is outside noise frequency ranges
                f_noise_signal.append(float(0))  # # add 0 to  f_noise_signal list if it was the case
            else:  # if freq is inside noise frequency ranges
                f_noise_signal.append(value)  # add f_signal value to f_noise_signal

            # Selecting body_component values
            if (abs(freq) <= freq1 or abs(freq) > freq2):  # testing if freq is outside Body_component frequency ranges
                f_body_signal.append(float(0))  # add 0 to  f_body_signal list
            else:  # if freq is inside Body_component frequency ranges
                f_body_signal.append(value)  # add f_signal value to f_body_signal list

        ################### Inverse the transformation of signals in freq domain ########################
        # applying the inverse fft(ifft) to signals in freq domain and put them in float format
        t_DC_component = ifft(np.array(f_DC_signal)).real
        t_body_component = ifft(np.array(f_body_signal)).real
        t_noise = ifft(np.array(f_noise_signal)).real

        total_component = t_signal - t_noise  # extracting the total component(filtered from noise)
        #  by substracting noise from t_signal (the original signal).

        # return outputs mentioned earlier
        return (total_component, t_DC_component, t_body_component, t_noise)

    def median(self, signal):  # input: numpy array 1D (one column)
        array = np.array(signal)
        # applying the median filter
        med_filtered = sp.signal.medfilt(array, kernel_size=3)  # applying the median filter order3(kernel_size=3)
        return med_filtered  # return the med-filtered signal: numpy array 1D

    def import_labels_file(self, path, columns):
        #    FUNCTION: import_raw_labels_file(path,columns)
        #    #######################################################################
        #    #      1- Import labels.txt                                           #
        #    #      2- convert data from txt format to int                         #
        #    #      3- convert integer data to a dataframe & insert columns names  #
        #    #######################################################################

        ######################################################################################
        # Inputs:                                                                            #
        #   path: A string contains the path of "labels.txt"                                 #
        #   columns: A list of strings contains the columns names in order.                  #
        # Outputs:                                                                           #
        #   dataframe: A pandas Dataframe contains labels  data in int format                #
        #             with columns names.                                                    #
        ######################################################################################

        # open the txt file
        labels_file = open(path, 'r')

        # creating a list
        labels_file_list = []

        # Store each row in a list ,convert its list elements to int type
        for line in labels_file:
            labels_file_list.append([int(element) for element in line.split()])
        # convert the list of lists into 2D numpy array
        data = np.array(labels_file_list)

        # Create a pandas dataframe from this 2D numpy array with column names
        data_frame = pd.DataFrame(data=data, columns=columns)

        # returning the labels dataframe
        return data_frame

    def import_raw_signals(self, file_path, columns):
        #    FUNCTION: import_raw_signals(path,columns)
        #    ###################################################################
        #    #           1- Import acc or gyro file                            #
        #    #           2- convert from txt format to float format            #
        #    #           3- convert to a dataframe & insert column names       #
        #    ###################################################################

        ######################################################################################
        # Inputs:                                                                            #
        #   file_path: A string contains the path of the "acc" or "gyro" txt file            #
        #   columns: A list of strings contains the column names in order.                   #
        # Outputs:                                                                           #
        #   dataframe: A pandas Dataframe contains "acc" or "gyro" data in a float format    #
        #             with columns names.                                                    #
        ######################################################################################

        # open the txt file
        opened_file = open(file_path, 'r')

        # Create a list
        opened_file_list = []

        # loop over each line in the opened_file
        # convert each element from txt format to float
        # store each raw in a list
        for line in opened_file:
            opened_file_list.append([float(element) for element in line.split()])

        # convert the list of lists into 2D numpy array(computationally efficient)
        data = np.array(opened_file_list)

        # Create a pandas dataframe from this 2D numpy array with column names
        data_frame = pd.DataFrame(data=data, columns=columns)

        # return the data frame
        return data_frame

    def encoding_latency_UCI(self, coding_f, analog_data, coding_n, min=0, max=np.pi):
        pass

    def get_series_data(self, data_frame, is_group=False):
        pass

    def get_series_data_list(self, data_frame, is_group=False):
        pass

    def select_data(self, fraction, data_frame, is_order=True, **kwargs):
        pass