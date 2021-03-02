import json
from typing import List
import sys
sys.path.append(".")
import torch
from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.decoder import Decoder
from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from deepspeech_pytorch.model import DeepSpeech
from deepspeech_pytorch.utils import load_decoder, load_model
from Punction import transcribe_comma
def getTranscript(filename):
    ok = ""
    with open(filename) as f:
        content = f.readlines()
    for line in content:
        ok += line
    print(" Input transcript : \33[33m",  line  ,"\33[0m")
def decode_results(decoded_output: List,
                   decoded_offsets: List,
                   cfg: TranscribeConfig):
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "path": cfg.model.model_path
            },
            "language_model": {
                "path": cfg.lm.lm_path
            },
            "decoder": {
                "alpha": cfg.lm.alpha,
                "beta": cfg.lm.beta,
                "type": cfg.lm.decoder_type.value,
            }
        }
    }
    
    for b in range(len(decoded_output)):
        for pi in range(min(cfg.lm.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            if cfg.offsets:
                result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)
    # print("\33[33m", decoded_output[0][0]  ,"\33[0m") 
    return results
import time

def transcribe(cfg: TranscribeConfig):
    commo_model,dict_data,  word_dict, char_dict = transcribe_comma.loadModel()
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model = load_model(device=device,
                       model_path=cfg.model.model_path,
                       use_half=cfg.model.use_half)

    decoder = load_decoder(labels=model.labels,
                           cfg=cfg.lm)

    spect_parser = SpectrogramParser(audio_conf=model.audio_conf,
                                     normalize=True)

    #Đối với beamsearch decoded_putput cho ra mảng (1xbeam_width) với các phần tử là các câu có thể xảy ra:
    #VD: [["toi đi hộc", "tôi di hoc", "tôi đi ho",...]] 512 phần tử (beam_width=512)
    
    tim1 = time.time()
    decoded_output, decoded_offsets = run_transcribe(audio_path=cfg.audio_path,
                                                     spect_parser=spect_parser,
                                                     model=model,
                                                     decoder=decoder,
                                                     device=device,
                                                     use_half=cfg.model.use_half)
    results = decode_results(decoded_output=decoded_output,
                             decoded_offsets=decoded_offsets,
                             cfg=cfg)
    resp = json.dumps(results, ensure_ascii=False)
    
    tim2 = time.time()
    print("Audio transcribe cost : "+ str(tim2 - tim1))
    results['output'][0]['transcription'] = transcribe_comma.runTranscribe(commo_model,dict_data,  word_dict, char_dict,results['output'][0]['transcription'] )
    #print("DEBUG : ", resp)
    return results['output'][0]['transcription'], results['_meta']


def run_transcribe(audio_path: str,
                   spect_parser: SpectrogramParser,
                   model: DeepSpeech,
                   decoder: Decoder,
                   device: torch.device,
                   use_half: bool):
    # audio_path
    # try:
    #     # inTranscript = audio_path.replace("wav", "txt")
    #     # print(inTranscript)
    #     # getTranscript(inTranscript)
    #     pass
    # except Exception as asd:
    #     print(asd)
    #     pass
    spect = spect_parser.parse_audio(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    if use_half:
        spect = spect.half()
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    return decoded_output, decoded_offsets
