import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, iirnotch, cheby2


def notch(data):
    # Create/view notch filter
    samp_freq = 10000  # Sample frequency (Hz)
    notch_freq = 50.0  # Frequency to be removed from signal (Hz)
    quality_factor = 30.0  # Quality factor
    b_notch, a_notch = iirnotch(notch_freq, quality_factor, samp_freq)

    y = filtfilt(b_notch, a_notch, data)
    return y


def prep_data() -> (np.ndarray, np.ndarray, np.ndarray):
    data_amount = 5526017

    df = pd.read_csv("test_data.csv", header=None)
    df = df.values.astype('float64')
    signal = df[:, 1]
    voltage = df[:, 2]
    time_steps = np.array([i/10000 for i in range(1, data_amount)])
    print(signal.shape, time_steps.shape)

    return signal, voltage, time_steps


def make_plot(secs, first, second, hz_amount, save=False, file_name='', first_name='signal', second_name='voltage'):
    plt.figure(figsize=(30, 6))
    plt.subplot(2, 1, 1)

    plt.plot(secs[0:hz_amount], first[0:hz_amount])
    plt.ylabel(first_name)

    plt.subplot(2, 1, 2)

    plt.plot(secs[0:hz_amount], second[0:hz_amount])
    plt.ylabel(second_name)

    if save:
        plt.savefig(file_name)
    else:
        plt.show()


def check_interval(interval, dif):
    lowest = min(interval)
    highest = max(interval)
    if abs(highest-lowest) > dif:
        return True
    return False


def split_time(volt, secs) -> list:
    time_intervals = []
    frame = 5000
    i = 0
    while True:
        start = i
        stop = i+frame
        interval = volt[start:stop]

        if check_interval(interval, 0.3):
            # interval is bad
            time_interval = (start, stop)
            time_intervals.append(time_interval)

        i += frame

        if i+frame > len(secs-1):
            break

    return time_intervals


def write_bad_sectors_to_file(sectors):
    with open('bad_sectors.txt', 'w') as file:
        for i in sectors:
            file.write(str(i[0]) + ' ' + str(i[1]) + '\n')


def read_bad_sectors():
    b_s = []

    with open('bad_sectors.txt', 'r') as file:
        for i in file:
            line = i.split()
            b_s.append((int(line[0]), int(line[1])))
    return b_s


def get_good_time(t, v, s, bad_sectors):
    for i in reversed(bad_sectors):
        s = np.delete(s, [j for j in range(i[0], i[1])])
        v = np.delete(v, [j for j in range(i[0], i[1])])
        t = np.delete(t, [j for j in range(i[0], i[1])])

    return t, v, s


def save_csvs_to_dir(t, v, s):
    if 'intervals' not in os.listdir():
        os.mkdir('intervals')
    os.chdir('intervals')

    length = len(t)

    start = 0
    counter = 0
    for i in range(1, length):
        if t[i] - t[i-1] > 0.0002 or i == length-1:

            # if counter > 100:
            #     print('ABOVE 100 FILES')
            #     break

            stop = i

            if stop - start < 100:
                start = i
                continue

            currtime = t[start:stop]
            currvolt = v[start:stop]
            currsign = s[start:stop]

            currsign = notch(currsign)
            currsign = cheby_filter_high(currsign)
            currsign = cheby_filter_low(currsign)

            csv_name = str(t[start]) + '_' + str(t[stop-1]) + '.csv'

            zipped = list(zip(currtime, currsign, currvolt))
            df = pd.DataFrame(zipped)
            df.reset_index()
            df.to_csv(csv_name, index=False, header=False)

            start = i


def cheby_filter_low(data):
    b, a = cheby2(4, 40, 1000, 'low', fs=10000)
    y = filtfilt(b, a, data)
    return y


def cheby_filter_high(data):
    b, a = cheby2(4, 40, 5, 'high', fs=10000)
    y = filtfilt(b, a, data)
    return y


def save_interval(start, stop, s, t):
    """ here's how to save single interval """
    zeros = np.array([0 for i in range(start, stop)])

    zipped = list(zip(t, s, zeros))
    df = pd.DataFrame(zipped)
    df.reset_index()
    print(df.head())

    df.to_csv('1.csv', index=False, header=False)


def make_intervals(sig, volt, tim):
    """here's how to make intervals in folder"""

    splitted_time = split_time(volt, tim)
    # print('time split...')
    write_bad_sectors_to_file(splitted_time)
    # print('bad sectors written...')

    bad_sectors = read_bad_sectors()
    # print('bad sectors read...')
    tim, volt, sig = get_good_time(tim, volt, sig, bad_sectors)
    # print('good times got...')

    # print('time shape:')
    # print(tim.shape)

    save_csvs_to_dir(tim, volt, sig)


if __name__ == '__main__':
    signal, voltage, time = prep_data()
    first_few_points_start = 0
    first_few_points_stop = 6000

    print('total length: ', len(voltage), '\n')

    time = time[first_few_points_start:first_few_points_stop]
    voltage = voltage[first_few_points_start:first_few_points_stop]
    signal = signal[first_few_points_start:first_few_points_stop]

    # make_plot(time, signal, voltage, len(voltage), save=False, file_name='before.png', second_name='voltage')


    # make_intervals(signal, voltage, time)

    # make_plot(time, better_signal, voltage, len(voltage), save=False, file_name='after02.png')

    save_interval(first_few_points_start, first_few_points_stop, signal, time)

    # make_plot(time, signal, better_signal, len(signal), first_name='signal', second_name='better signal', save=True, file_name='filter_compare.png')