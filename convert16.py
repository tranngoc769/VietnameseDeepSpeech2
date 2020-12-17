#Tool convert file wav to 16000Hz
import wave
import audioop
import sys
import os
import argparse

parser = argparse.ArgumentParser(description='Huong dan su dung')
parser.add_argument('-pw','--src', help='folder wav file', required=True)
parser.add_argument('-pt','--dst', help='folder wav with 16000Hz', required=True)


ok = 0
failed = 0
from scipy.io import wavfile
import scipy.io
def downsampleWav(src, dst, outrate=16000):
    with wave.open(src, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        inchannels=wave_file.getnchannels()
        outchannels=inchannels
    global ok, failed
    if not os.path.exists(src):
        print('Source not found!')
        failed += 1
        return False
    try:
        s_read = wave.open(src, 'r')
        s_write = wave.open(dst, 'w')
    except:
        print('Failed to open files!')
        failed += 1
        return False

    n_frames = s_read.getnframes()
    data = s_read.readframes(n_frames)
    try:
        converted = audioop.ratecv(data, 2, inchannels, frame_rate, outrate, None)
        if outchannels == 1 & inchannels != 1:
            converted[0] = audioop.tomono(converted[0], 2, 1, 0)
    except:
        print('Failed to downsample wav')
        failed += 1
        return False

    try:
        s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
        s_write.writeframes(converted[0])
    except:
        print('Failed to write wav')
        failed += 1
        return False

    try:
        s_read.close()
        s_write.close()
    except:
        print('Failed to close wav files')
        failed += 1
        return False
    ok += 1
    return True

if __name__ == "__main__":
    args = vars(parser.parse_args())

    src = args['src']
    dst = args['dst']
    downsampleWav(src, dst, outrate=16000)
