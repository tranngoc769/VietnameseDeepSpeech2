import hydra
import torch
from tqdm import tqdm

from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.utils import load_model, load_decoder


@torch.no_grad()
def evaluate(cfg: EvalConfig):
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model = load_model(device=device,
                       model_path=cfg.model.model_path,
                       use_half=cfg.model.use_half)

    decoder = load_decoder(labels=model.labels,
                           cfg=cfg.lm)#ở đây chọn xem loại beam hay greedy trong file ultis
    target_decoder = GreedyDecoder(model.labels,
                                   blank_index=model.labels.index('_'))
    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
                                      manifest_filepath=hydra.utils.to_absolute_path(cfg.test_manifest),
                                      labels=model.labels,
                                      normalize=True)
    test_loader = AudioDataLoader(test_dataset,
                                  batch_size=cfg.batch_size,
                                  num_workers=cfg.num_workers)
    wer, cer, output_data,wer2,cer2 = run_evaluation(test_loader=test_loader,
                                           device=device,
                                           model=model,
                                           decoder=decoder,
                                           target_decoder=target_decoder,
                                           save_output=cfg.save_output,
                                           verbose=cfg.verbose,
                                           use_half=cfg.model.use_half)

    print('Test Summary \t'
          'Average WER-Beam {wer:.3f}\t'
          'Average CER-Beam {cer:.3f}\t'.format(wer=wer, cer=cer))
    print(
          'Average WER-Greedy {wer:.3f}\t'
          'Average CER-Greedy {cer:.3f}\t'.format(wer=wer2, cer=cer2))          
    if cfg.save_output:
        torch.save(output_data, hydra.utils.to_absolute_path(cfg.save_output))


@torch.no_grad()
def run_evaluation(test_loader,
                   device,
                   model,
                   decoder,
                   target_decoder,
                   save_output=None,
                   verbose=False,
                   use_half=False):
    model.eval()
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()#độ dài 1 dòng trong spect của mẫu, input_sizes là 32 mẫu
        inputs = inputs.to(device)
        if use_half:
            inputs = inputs.half()#không thay đổi nhiều
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        out, output_sizes = model(inputs, input_sizes)


        decoded_output, _ = decoder.decode(out, output_sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)

        if save_output is not None:
            # add output to data array, and continue
            output_data.append((out.cpu(), output_sizes, target_strings))
    #     for x in range(len(target_strings)):
    #         transcript, reference = decoded_output[x][0], target_strings[x][0]
    #         wer_inst = decoder.wer(transcript, reference)
    #         cer_inst = decoder.cer(transcript, reference)
    #         total_wer += wer_inst
    #         total_cer += cer_inst
    #         num_tokens += len(reference.split())
    #         num_chars += len(reference.replace(' ', ''))
    #         if verbose:
    #             print("Ref:", reference.lower())
    #             print("Hyp:", transcript.lower())
    #             print("WER:", float(wer_inst) / len(reference.split()),
    #                   "CER:", float(cer_inst) / len(reference.replace(' ', '')), "\n")
    # wer = float(total_wer) / num_tokens
    # cer = float(total_cer) / num_chars

    ############
        decoder2 = GreedyDecoder(labels=model.labels,
                                blank_index=model.labels.index('_'))  
        old_out, out_offsets=decoder2.decode(out,output_sizes)
        total_cer2, total_wer2, num_tokens2, num_chars2 = 0, 0, 0, 0
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = decoder.wer(transcript, reference)
            cer_inst = decoder.cer(transcript, reference)
            total_wer += wer_inst
            total_cer += cer_inst
            num_tokens += len(reference.split())
            num_chars += len(reference.replace(' ', ''))
            if verbose:
                print("Truth :", reference.lower())
                print("Beam  :", transcript.lower())
                print("WER:", float(wer_inst) / len(reference.split()),
                        "CER:", float(cer_inst) / len(reference.replace(' ', '')))

            transcript2=old_out[x][0]
            wer_inst2 = decoder2.wer(transcript2, reference)
            cer_inst2 = decoder2.cer(transcript2, reference)
            total_wer2 += wer_inst2
            total_cer2 += cer_inst2
            num_tokens2 += len(reference.split())
            num_chars2 += len(reference.replace(' ', ''))
            if verbose:
                print("Greedy:",transcript2.lower())
                print("WER2:", float(wer_inst2) / len(reference.split()),
                    "CER2:", float(cer_inst2) / len(reference.replace(' ', '')), "\n")
    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars
    wer2 = float(total_wer2) / num_tokens2
    cer2 = float(total_cer2) / num_chars2 
    ##########

    # for x in range(len(target_strings)):
    #     transcript2=old_out[x][0]
    #     wer_inst2 = decoder2.wer(transcript2, reference)
    #     cer_inst2 = decoder2.cer(transcript2, reference)
    #     total_wer2 += wer_inst2
    #     total_cer2 += cer_inst2
    #     num_tokens2 += len(reference.split())
    #     num_chars2 += len(reference.replace(' ', ''))
    #     if verbose:
    #         print("Old:",transcript2.lower())
    #         print("WER2:", float(wer_inst2) / len(reference.split()),
    #             "CER2:", float(cer_inst2) / len(reference.replace(' ', '')), "\n")
       
################
    return wer * 100, cer * 100, output_data,wer2*100,cer*100
