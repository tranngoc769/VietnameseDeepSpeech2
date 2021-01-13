from dataclasses import dataclass
from deepspeech_pytorch.enums import DecoderType


@dataclass
class LMConfig:
    #decoder_type: DecoderType = DecoderType.greedy
    decoder_type: DecoderType = DecoderType.beam#quyennn
    lm_path: str = "/work/languagemodel/ARPA_BINARY/final-1234.binary" # Path to an (optional) kenlm language model for use with beam search (req\'d with trie)
    top_paths: int = 1  # Number of beams to return, có rất nhìu câu sau khi beam trả về
    alpha: float = 2.0  # Language model weight, hihi quyen từ 0.0 ->
    beta: float =  1.0  # Language model word bonus (all words)    # alpha: float = 0  # Language model weight, hihi quyen từ 0.0 ->
    # beta: float = 0  # Language model word bonus (all words)
    cutoff_top_n: int = 300  # Cutoff_top_n characters with highest probs in vocabulary will be used in beam search  40->300
    cutoff_prob: float = 1.0  # Cutoff probability in pruning,default 1.0, no pruning.
    beam_width: int = 1024  # Beam width to use  quyen 10->
    lm_workers: int = 4  # Number of LM processes to use


@dataclass
class ModelConfig:
    use_half: bool = True  # Use half precision. This is recommended when using mixed-precision at training time
    cuda: bool = True
    model_path: str = "/work/Source/deepspeech.pytorch/models/deepspeech_1600_vinfpt_25_50.pth"


@dataclass
class InferenceConfig:
    lm: LMConfig = LMConfig()
    model: ModelConfig = ModelConfig()


@dataclass
class TranscribeConfig(InferenceConfig):
    audio_path: str = "/work/dataset_product/wav/product_timzjz6u.wav"  # Audio file to predict on
    offsets: bool = False  # Returns time offset information,  bảng ascii


@dataclass
class EvalConfig(InferenceConfig):
    test_manifest: str ="/dataset/vi_dev.csv"  # Path to validation manifest csv
    verbose: bool = True  # Print out decoded output and error of each sample
    save_output: str = "/dataset/lm_outtest/outtest"  # Saves output of model from test to this file_path
    batch_size: int = 25  # Batch size for testing quyen 20->
    num_workers: int = 0  #quyen 4->


@dataclass
class ServerConfig(InferenceConfig):
    host: str = '127.0.0.1'
    port: int = 8888