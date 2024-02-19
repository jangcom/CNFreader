#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for reading a Canberra Nuclear File (CNF) form GENIE2000 software.

It can be used as a stand alone script or as a module.

Optionally, it generates a text file with the relevant information read from
the CNF file. The output file name is the input file plus the '.txt' extension.



Examples
--------
    >>> python read_cnf.py name_of_the_file.CNF

    ('name_of_the_file.CNF.txt' is automatically created)

References
----------

This script was made as a copy of the c program 'cnfconv' written for the same
porpouse. That software can be found here:

https://github.com/messlinger/cnfconv

All the information of the binary file encoding was taken from the file
'cnf_file_format.txt' of the above repository.


"""
import os
import sys
import re
import numpy as np
import time
import struct
import pandas as pd


def show_file_gen(fname):
    """Show a file has been generated.

    Parameters
    ----------
    fname : str
        A file name to be reported.

    Returns
    -------
    None.
    """
    print(f'[{fname}] generated.')


def read_cnf_file(filename, out_path,
                  write_output=False, is_str_kev_to_MeV=True):
    """
    Reads data of a Canberra Nuclear File used by the Genie2000 software.

    Parameters
    ----------
    filename : str
        Name of the file to be read.
    out_path : str
        The path to which output files will be saved.
    write_output : bool, optional
        Indicate weather to write an output file or not.
    is_str_kev_to_MeV : bool, optional
        If True, the string 'MeV' is recast to 'keV' while the actual energy
        values remain unchanged.

    Returns
    -------
    read_dic : dictionary
        Dictionary with all the magnitudes read. Depending on the data
        available,the dictionaries keys may change. Some possible keys are:
        Sample id
        Channels
        Sample unit
        Sample name
        Channels data
        Energy unit
        Energy coefficients
        Shape coefficients
        Left marker
        Total counts
        Number of channels
        Start time
        Counts in markers
        Right marker
        Sample type
        Sample description
        User name
        Live time
        Energy
        Real time
        Measurement mode
        MCA type
        Data source

    Examples
    --------
        >>> from read_cnf import lee_cnf_file
        >>> read_dic = read_cnf_file('name_of_the_file.CNF')
        >>> read_dic['Live time']

    TODO
    ----
    - Markers information is not being read correctly.
    - If the CNF file are obtained in a MCA mode, the live and real time are
    not read correctly.
    - Additional data must be read in case of a file from MCA mode
    (mainly dwell time).

    """
    # Names of the output files
    out_fmts = ['dat', 'csv', 'xlsx']
    out_bname = os.path.splitext(os.path.basename(filename))[0]
    out_bname_full = '{}/{}'.format(out_path, out_bname)
    out_fnames_full = {fmt: '{}.{}'.format(out_bname_full, fmt)
                       for fmt in out_fmts}

    # Dictionary with all the information read
    read_dic = {}
    with open(filename, 'rb') as f:
        i = 0
        while True:
            # List of available section headers
            sec_header = 0x70 + i*0x30
            i += 1
            # Section id in header
            sec_id_header = uint32_at(f, sec_header)

            # End of section list
            if sec_id_header == 0x00:
                break

            # Location of the begining of each sections
            sec_loc = uint32_at(f, sec_header+0x0a)
            # Known section id's:
            # Parameter section (times, energy calibration, etc)
            if sec_id_header == 0x00012000:
                offs_param = sec_loc
                read_dic.update(get_energy_calibration(f, offs_param))
                read_dic.update(get_date_time(f, offs_param))
                read_dic.update(get_shape_calibration(f, offs_param))
            # String section
            elif sec_id_header == 0x00012001:
                offs_str = sec_loc
                read_dic.update(get_strings(f, offs_str))
            # Marker section
            elif sec_id_header == 0x00012004:
                offs_mark = sec_loc
                read_dic.update(get_markers(f, offs_mark))
            # Channel data section
            elif sec_id_header == 0x00012005:
                offs_chan = sec_loc
                read_dic.update(get_channel_data(f, offs_param, offs_chan))
            else:
                continue

            # For known sections: section header ir repeated in section block
            if (sec_id_header != uint32_at(f, sec_loc)):
                print('File {}: Format error\n'.format(filename))

    # Once the file is read, some derived magnitudes can be obtained

    # Convert channels to energy
    if set(('Channels', 'Energy coefficients')) <= set(read_dic):
        read_dic.update(chan_to_energy(read_dic))

    # Compute ingegration between markers
    if set(('Channels', 'Left marker')) <= set(read_dic):
        read_dic.update(markers_integration(read_dic))

    # Recast the energy unit from MeV to keV.
    if is_str_kev_to_MeV:
        read_dic['Energy unit'] = re.sub('MeV', 'keV', read_dic['Energy unit'])

    print(50*'=')
    print(10*' '+'File '+str(filename)+' succesfully read!' + 10*' ')
    print(50*'=')

    # Create a DF.
    the_data = {
        'Channel (ch)': read_dic['Channels'],
        'Energy ({})'.format(read_dic['Energy unit']): read_dic['Energy'],
        'Count (cnt)': read_dic['Channels data'],
    }
    df = pd.DataFrame(the_data)
    df.set_index('Channel (ch)', inplace=True)

    # File writing
    if write_output:
        if not os.path.exists(out_path):
            os.mkdir(out_path)
            show_file_gen(out_path)
        # Write the .cnf data to a plain text (.dat) file.
        write_to_file(out_fnames_full['dat'], read_dic,
                      is_str_kev_to_MeV=is_str_kev_to_MeV)
        # Write the .cnf data to .csv and .xlsx files using pandas.
        df.to_csv(out_fnames_full['csv'])
        df.style.background_gradient().to_excel(out_fnames_full['xlsx'])

        # Report file generation.
        for fmt in out_fmts:
            show_file_gen(out_fnames_full[fmt])

    return read_dic

##########################################################
# Definitions for reading some data types
##########################################################


def uint8_at(f, pos):
    f.seek(pos)
    return np.fromfile(f, dtype=np.dtype('<u1'), count=1)[0]


def uint16_at(f, pos):
    f.seek(pos)
    return np.fromfile(f, dtype=np.dtype('<u2'), count=1)[0]


def uint32_at(f, pos):
    f.seek(pos)
    return np.fromfile(f, dtype=np.dtype('<u4'), count=1)[0]


def uint64_at(f, pos):
    f.seek(pos)
    return np.fromfile(f, dtype=np.dtype('<u8'), count=1)[0]


def pdp11f_at(f, pos):
    """
    Convert PDP11 32bit floating point format to
    IEE 754 single precision (32bits)
    """
    f.seek(pos)
    # Read two int16 numbers
    tmp16 = np.fromfile(f, dtype=np.dtype('<u2'), count=2)
    # Swapp positions
    mypack = struct.pack('HH', tmp16[1], tmp16[0])
    f = struct.unpack('f', mypack)[0]/4.0
    return f


def time_at(f, pos):
    return ~uint64_at(f, pos)*1e-7


def datetime_at(f, pos):
    return uint64_at(f, pos) / 10000000 - 3506716800


def string_at(f, pos, length):
    f.seek(pos)
    # In order to avoid characters with not utf8 encoding
    return f.read(length).decode('utf8').rstrip('\00').rstrip()

###########################################################
# Definitions for locating and reading data inside the file
###########################################################


def get_strings(f, offs_str):
    """Read strings section."""

    sample_name = string_at(f, offs_str + 0x0030, 0x40)
    sample_id = string_at(f, offs_str + 0x0070, 0x10)
    # sample_id   = string_at(f, offs_str + 0x0070, 0x40)
    sample_type = string_at(f, offs_str + 0x00b0, 0x10)
    sample_unit = string_at(f, offs_str + 0x00c4, 0x40)
    user_name = string_at(f, offs_str + 0x02d6, 0x18)
    sample_desc = string_at(f, offs_str + 0x036e, 0x100)

    out_dic = {
               'Sample name': sample_name,
               'Sample id': sample_id,
               'Sample type': sample_type,
               'Sample unit': sample_unit,
               'User name': user_name,
               'Sample description': sample_desc
              }

    return out_dic


def get_energy_calibration(f, offs_param):
    """Read energy calibration coefficients."""

    offs_calib = offs_param + 0x30 + uint16_at(f, offs_param + 0x22)
    A = np.empty(4)
    A[0] = pdp11f_at(f, offs_calib + 0x44)
    A[1] = pdp11f_at(f, offs_calib + 0x48)
    A[2] = pdp11f_at(f, offs_calib + 0x4c)
    A[3] = pdp11f_at(f, offs_calib + 0x50)

    # Assuming a maximum length of 0x11 for the energy unit
    energy_unit = string_at(f, offs_calib + 0x5c, 0x11)

    # MCA type
    MCA_type = string_at(f, offs_calib + 0x9c, 0x10)

    # Data source
    data_source = string_at(f, offs_calib + 0x108, 0x10)

    out_dic = {'Energy coefficients': A,
               'Energy unit': energy_unit,
               'MCA type': MCA_type,
               'Data source': data_source
               }

    return out_dic


def get_shape_calibration(f, offs_param):
    """
    Read Shape Calibration Parameters :
        FWHM=B[0]+B[1]*E^(1/2)  . B[2] and B[3] probably tail parameters
    """

    offs_calib = offs_param + 0x30 + uint16_at(f, offs_param + 0x22)
    B = np.empty(4)
    B[0] = pdp11f_at(f, offs_calib + 0xdc)
    B[1] = pdp11f_at(f, offs_calib + 0xe0)
    B[2] = pdp11f_at(f, offs_calib + 0xe4)
    B[3] = pdp11f_at(f, offs_calib + 0xe8)

    out_dic = {'Shape coefficients': B}

    return out_dic


def get_channel_data(f, offs_param, offs_chan):
    """Read channel data."""

    # Total number of channels
    n_channels = uint8_at(f, offs_param + 0x00ba) * 256
    # Data in each channel
    f.seek(offs_chan + 0x200)
    chan_data = np.fromfile(f, dtype='<u4', count=n_channels)
    # Total counts of the channels
    total_counts = np.sum(chan_data)
    # Measurement mode
    meas_mode = string_at(f, offs_param + 0xb0, 0x03)

    # Create array with the correct channel numbering
    channels = np.arange(1, n_channels+1, 1)

    out_dic = {'Number of channels': n_channels,
               'Channels data': chan_data,
               'Channels': channels,
               'Total counts': total_counts,
               'Measurement mode': meas_mode
               }

    return out_dic


def get_date_time(f, offs_param):
    """Read date and time."""

    offs_times = offs_param + 0x30 + uint16_at(f, offs_param + 0x24)

    start_time = datetime_at(f, offs_times + 0x01)
    real_time = time_at(f, offs_times + 0x09)
    live_time = time_at(f, offs_times + 0x11)

    # Convert to formated date and time
    start_time_str = time.strftime('%d-%m-%Y, %H:%M:%S', time.gmtime(start_time))

    out_dic = {'Real time': real_time,
               'Live time': live_time,
               'Start time': start_time_str
               }
    return out_dic


def get_markers(f, offs_mark):
    """Read left and right markers."""

    # TODO: not working properly
    marker_left = uint32_at(f, offs_mark + 0x007a)
    marker_right = uint32_at(f, offs_mark + 0x008a)

    out_dic = {'Left marker': marker_left,
               'Right marker': marker_right,
               }

    return out_dic


def chan_to_energy(dic):
    """ Convert channels to energy using energy calibration coefficients."""

    A = dic['Energy coefficients']
    ch = dic['Channels']
    energy = A[0] + A[1]*ch + A[2]*ch*ch + A[3]*ch*ch*ch

    out_dic = {'Energy': energy}

    return out_dic


def markers_integration(dic):
    # Count between left and right markers
    # TODO: check integral counts limits
    chan_data = dic['Channels data']
    l_marker = dic['Left marker']
    r_marker = dic['Right marker']
    marker_counts = np.sum(chan_data[l_marker-1:r_marker-1])

    out_dic = {'Counts in markers': marker_counts}

    return out_dic

###########################################################
# Format of the output text file
###########################################################


def write_to_file(filename, dic,
                  cmt_symb='#', sep='\t',
                  is_str_kev_to_MeV=True):
    """Write data to a text file."""
    if is_str_kev_to_MeV:
        dic['Energy unit'] = re.sub('MeV', 'keV', dic['Energy unit'])
    # Measurement settings information
    keys_info = [
        '',
        'Sample name',
        '',
        'Sample id',
        'Sample type',
        'User name',
        'Sample description',
        '',
        'Start time',
        'Real time',
        'Live time',
        '',
        'Total counts',
        '',
        'Left marker',
        'Right marker',
        'Counts in markers',
        '',
        'Energy unit',
        '',
    ]
    lines_info = []
    for k in keys_info:
        if k in dic:
            k_w_unit = k
            if re.search('(?i)Real|Live\s*time', k):
                k_w_unit += ' (s)'
            lines_info.append('{} {}: {}\n'.format(cmt_symb, k_w_unit, dic[k]))
        else:
            lines_info.append(f'{cmt_symb}\n')

    # Column headers
    _headers = ['Channel (ch)',
                'Energy ({})'.format(dic['Energy unit']),
                'Count (cnt)',
                'Count rate (cps)']
    header = sep.join(_headers)

    # Write the content to a file.
    with open(filename, 'w') as f:
        # Measurement settings information
        for line in lines_info:
            f.write(line)

        # Calibration information
        f.write('# Energy calibration coefficients (E = sum(Ai * n**i))\n')
        for j, co in enumerate(dic['Energy coefficients']):
            f.write('{} - A{} = {}\n'.format(cmt_symb, j, co))
        f.write('# Shape calibration coefficients (FWHM = B0 + B1*E^(1/2);'
                + ' Lower tail= B2 + B3*E)\n')
        for j, co in enumerate(dic['Shape coefficients']):
            f.write('{} - B{} = {}\n'.format(cmt_symb, j, co))
        f.write('{}\n\n'.format(cmt_symb))

        # Measured data
        f.write('{}\n'.format(cmt_symb))
        f.write('{} {}\n'.format(cmt_symb, header))
        f.write('{}\n'.format(cmt_symb))
        for i, j, k in zip(dic['Channels'],
                           dic['Energy'],
                           dic['Channels data']):
            the_data = sep.join(map(str, [i, j, k, k/dic['Live time']]))
            f.write('{}\n'.format(the_data))


if __name__ == "__main__":
    # Check if command line argument is given
    if len(sys.argv) < 2:
        # Default name if not provided
        directory = "Examples"
        name = "cs137.CNF"
        filename = os.path.join(directory, name)
        print('*'*10 + 'No input file was given\n')
        print('*'*10 + 'Reading file:' + filename + '\n')
    else:
        filename = sys.argv[1]
        if len(sys.argv) >= 3:
            out_path = sys.argv[2]
        else:
            out_path = os.path.dirname(filename)

    c = read_cnf_file(filename, out_path, 'TRUE')

    # print('Sample id: {}'.format(c['Sample id']))
    # print('Measurement mode: {}'.format(c['Measurement mode']))

    chan = c['Channels']
    n_chan = c['Number of channels']
    chan_data = c['Channels data']
    energy = c['Energy']
    # print('Number of channels used: '+str(n_chan))

    # Testing channels and energy calibration
    inchan = 250
    # print('At channel {}:'.format(inchan))
    # print('\t Counts: {}'.format(chan_data[np.where(chan == inchan)][0]))
    # print('\t Energy: {}'.format(energy[np.where(chan == inchan)][0]))

    is_plot = False
    if is_plot:
        import matplotlib.pyplot as plt
        fig1 = plt.figure(1, figsize=(8, 8))

        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel(u'Channels')
        ax1.set_ylabel(u'Counts')
        ax1.plot(chan, chan_data, 'k.')
        ax1.set_title('File read: ' + filename)

        plt.show()
