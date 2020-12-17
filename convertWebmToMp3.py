from pydub import AudioSegment

#Tool convert file webm  to mp3
import wave
import audioop
import sys
import os
import argparse

parser = argparse.ArgumentParser(description='Huong dan su dung')
parser.add_argument('-pw','--src', help=' wav file', required=True)
parser.add_argument('-pt','--dst', help=' wav with 16000Hz', required=True)


def convertWebmToMp3(src, dst, bitrate="125k"):
    song = AudioSegment.from_file(src,"webm")
    song.export(dst, format="mp3", bitrate=bitrate)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    src = args['src']
    dst = args['dst']
    convertWebmToMp3(src, dst,bitrate="125k")