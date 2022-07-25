import pandas as pd
import numpy as np
import re


#peer review for Cameron Boeder


# Find the indexes of where each packet starts
def get_pkt_line_srt_idx(_data):
    line_num_idx = []
    pattern = re.compile('|'.join(hex_line_nums[:, 0] + r'\s\s[0-9a-f]+[0-9a-f]\s'))
    matches = pattern.finditer(_data)
    matches = list(matches)
    idx = [m.start(0) for m in matches]
    for i in range(len(matches)):
        line_num_idx.append([matches[i][0], idx[i]])
    return np.array(line_num_idx)


# Function to convert hexadecimal to binary
def convert_hex_to_bin(_byte):
    scale = 16
    num_of_bits = 8
    return bin(int(_byte, scale))[2:].zfill(num_of_bits)


# Function to parse each packets hexdump data into an array of binary values
def parse_pkt_hex_data(_data):
    hex_srt = 6
    hex_end = 53
    pkt = ''
    pkt_bytes = []
    line_num_idx = get_pkt_line_srt_idx(_data)
    for i in range(len(line_num_idx)):
        if i < len(line_num_idx) - 1:
            temp = _data[int(line_num_idx[i, 1]):int(line_num_idx[i + 1, 1])]
            temp = temp[hex_srt:hex_end].replace(' ', '')
            pkt = pkt + temp
            if line_num_idx[i + 1, 0][:4] == hex_line_nums[0]:
                pkt = "{:0<3036}".format(pkt)
                pkt_hex_bytes = re.findall('..', pkt)
                for j in range(len(pkt_hex_bytes)):
                    pkt_bytes.append(convert_hex_to_bin(pkt_hex_bytes[j]))
                if len(pkt_bytes) <= 1518:
                    f = open(pkt_export_file_path, 'a')
                    f.write(str(pkt_bytes)[1:-1] + '\n')
                    f.close()
                pkt = ''
                pkt_bytes = []
        else:
            temp = _data[int(line_num_idx[i, 1]):]
            temp = temp[hex_srt:hex_end].replace(' ', '')
            pkt = pkt + temp
            pkt = "{:0<3036}".format(pkt)
            pkt_hex_bytes = re.findall('..', pkt)
            for j in range(len(pkt_hex_bytes)):
                pkt_bytes.append(convert_hex_to_bin(pkt_hex_bytes[j]))
            if len(pkt_bytes) <= 1518:
                f = open(pkt_export_file_path, 'a')
                f.write(str(pkt_bytes)[1:-1])
                f.close()
            pkt = ''
            pkt_bytes = []


if __name__ == '__main__':
    hexdump_file_path = input('Enter hexdump txt file path:')
    pkt_export_file_path = input('Enter file path for where to save packet bytes file:')

    # Read in hex line number list
    hex_line_nums = pd.read_csv('hex_line_nums.csv', header=None)
    hex_line_nums = np.array(hex_line_nums)

    # Read in pcap txt hexdump data
    txt = open(hexdump_file_path, 'r')
    data = txt.read()

    parse_pkt_hex_data(data)

