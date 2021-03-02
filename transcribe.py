import hydra
from hydra.core.config_store import ConfigStore

from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.inference import transcribe

cs = ConfigStore.instance()
cs.store(name="config", node=TranscribeConfig)

from Punction import transcribe_comma

@hydra.main(config_name="config")
def hydra_main(cfg: TranscribeConfig):
   transcript, meta = transcribe(cfg=cfg)
   
   print("\n Output transcript : \033[94m",  transcript  ,"\033[0m")
if __name__ == '__main__':
    hydra_main()
