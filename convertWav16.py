from os import path
from pydub import AudioSegment
import argparse
import wave
import audioop
import sys
import os
from convert16 import downsampleWav
from convertToWav import convertToWav

parser = argparse.ArgumentParser(description='Huong dan su dung')
parser.add_argument('-pw','--src', help='folder wav file', required=True)
parser.add_argument('-pt','--dst', help='folder txt file', required=True)


def convertMp3ToWav16(src, dst):     
    temp=os.path.dirname(src)+"temp.wav"                                                             
    # convert wav to mp3                                                            
    convertToWav(src, temp)
    #convert wav to 16000Hz
    downsampleWav(temp, dst, outrate=16000)
    #os.remove(temp)

if __name__ == "__main__":
    args = vars(parser.parse_args())

    src = args['src']
    dst = args['dst']
    convertMp3ToWav16(src, dst)