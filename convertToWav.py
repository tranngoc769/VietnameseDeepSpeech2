from os import path
from pydub import AudioSegment
from pydub.utils import mediainfo
import argparse
parser = argparse.ArgumentParser(description='Huong dan su dung')
parser.add_argument('-pw','--src', help='folder wav file', required=True)
parser.add_argument('-pt','--dst', help='folder txt file', required=True)


# files                                                                         

# convert wav to mp3

def convertToWav(src, dst,bitrate="256k" ):
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav", bitrate=bitrate)

if __name__ == "__main__": 
    args = vars(parser.parse_args())

    src = args['src']
    dst = args['dst']                                                        
    convertToWav(src, dst )