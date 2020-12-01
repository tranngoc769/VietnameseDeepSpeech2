import math
import os
from tempfile import NamedTemporaryFile

import librosa
import numpy as np
import soundfile as sf
import sox
import torch
from torch.utils.data import Dataset, Sampler, DistributedSampler, DataLoader

from deepspeech_pytorch.configs.train_config import SpectConfig, AugmentationConfig
from deepspeech_pytorch.loader.spec_augment import spec_augment, visualization_spectrogram

####
import matplotlib.pyplot as plt
import librosa.display
####

def load_audio(path):
    sound, sample_rate = sf.read(path)
    # TODO this should be 32768.0 to get twos-complement range.
    # TODO the difference is negligible but should be fixed for new models.
    #sound = sound.astype('float32') / 32767  # normalize audio
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 sample_rate=16000,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        if not os.path.exists(path):
            print("Directory doesn't exist: {}".format(path))
            raise IOError
        self.paths = path is not None and librosa.util.find_files(path)
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_len = sox.file_info.duration(noise_path)
        data_len = len(data) / self.sample_rate
        noise_start = np.random.rand() * (noise_len - data_len)
        noise_end = noise_start + data_len
        noise_dst = audio_with_sox(noise_path, self.sample_rate, noise_start, noise_end)
        assert len(data) == len(noise_dst)
        noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
        data_energy = np.sqrt(data.dot(data) / data.size)
        data += noise_level * noise_dst * data_energy / noise_energy
        return data


class SpectrogramParser(AudioParser):
    def __init__(self,
                 audio_conf: SpectConfig,
                 normalize: bool = False,
                 augmentation_conf: AugmentationConfig = None):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augmentation_conf(Optional): Config containing the augmentation parameters
        """
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf.window_stride
        self.window_size = audio_conf.window_size
        self.sample_rate = audio_conf.sample_rate
        self.window = audio_conf.window.value
        self.normalize = normalize
        self.aug_conf = augmentation_conf
        if augmentation_conf and augmentation_conf.noise_dir:
            self.noise_injector = NoiseInjection(path=augmentation_conf.noise_dir,
                                                 sample_rate=self.sample_rate,
                                                 noise_levels=augmentation_conf.noise_levels)
        else:
            self.noise_injector = None

    def parse_audio(self, audio_path):
        if self.aug_conf and self.aug_conf.speed_volume_perturb:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = load_audio(audio_path)
        if self.noise_injector:
            add_noise = np.random.binomial(1, self.aug_conf.noise_prob)
            if add_noise:
                y = self.noise_injector.inject_noise(y)

        
        ##get ten file de ve hinh
        # nanlist=audio_path.split("/")
        # nanLs = nanlist[len(nanlist)-1]
        # name = nanLs.split(".")[0]+"_"+nanLs.split(".")[1]

        #tín hiệu thô
        # fig1,ax22= plt.subplots()
        # plt.title('Tín hiệu thô của câu nói \'anh có thể gọi cho tôi không\'')
        # plt.plot(y)
        # plt.xlabel('Sample')
        # plt.ylabel('Amplitude')
        # fig1.savefig('/work/Source/deepspeech.pytorch/deepspeech_pytorch/quyenImg/'+name+'tinhieutho'+'.png')

        n_fft = int(self.sample_rate * self.window_size)#320
        win_length = n_fft#320
        hop_length = int(self.sample_rate * self.window_stride)#160
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)#array([[ 2.42148260e-01+0.00000000e+00j, -1.00020550e-01+0.00000000e+00j,

        spect, phase = librosa.magphase(D)# S = log(S+1)##array([[ 2.42148260e-01+0.00000000e+00j, -1.00020550e-01+0.00000000e+00j,



        # print("++**",audio_path)
        # fig, ax = plt.subplots()
        # img = librosa.display.specshow(librosa.amplitude_to_db(spect,ref=np.max), y_axis='log', x_axis='time', ax=ax)
        # ax.set_title(audio_path)
        # fig.savefig('/work/Source/deepspeech.pytorch/deepspeech_pytorch/quyenImg/'+name+'.png')

#có thanh độ lớn biên độ
        # log_spectrogram = librosa.amplitude_to_db(spect)
        # plt.figure(figsize=(12,8))
        # librosa.display.specshow(log_spectrogram, sr=self.sample_rate,
        # y_axis='log', x_axis='time',hop_length=160)
        # plt.xlabel("Time")
        # plt.ylabel("Frequency")
        # plt.colorbar(format="%+2.0f dB")
        # plt.title("Spectrogram (dB)")
        # plt.savefig('/work/Source/deepspeech.pytorch/deepspeech_pytorch/quyenImg/'+name+'.png')

        spect = np.log1p(spect)#tensor([[2.1684e-01, 9.5329e-02, 1.0469e-01,  ..., 1.2308e-03, 2.3625e-03,
        
        # fig2, ax2 = plt.subplots()
        # img2 = librosa.display.specshow(librosa.amplitude_to_db(spect,ref=np.max), y_axis='log', x_axis='time', ax=ax)
        # ax2.set_title(audio_path+"(log)")
        # fig2.savefig('/work/Source/deepspeech.pytorch/deepspeech_pytorch/quyenImg/'+name+"(log)"+'.png')

        spect = torch.FloatTensor(spect)#tensor([[2.1684e-01, 9.5329e-02, 1.0469e-01,  ..., 1.2308e-03, 2.3625e-03,
        if self.normalize:
            
            mean = spect.mean()#tính trung bình cộng
            #mean=np.log1p(mean)

            std = spect.std()
            #std=np.log1p(std)#độ lệch chuẩn

            if (mean==torch.tensor(0) or std ==torch.tensor(0)):
                print("nan nan")
            
            spect.add_(-mean)
            spect.div_(std)

        if self.aug_conf and self.aug_conf.spec_augment:
            spect = spec_augment(spect)
        
        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError


# train_dataset = SpectrogramDataset(audio_conf=model.audio_conf,
#                                     manifest_filepath=to_absolute_path(cfg.data.train_manifest),
#                                     labels=model.labels,
#                                     normalize=True,
#                                     augmentation_conf=cfg.data.augmentation)

#Trả về xử lí spect và trainscript của từng  row trong csv. 
#Như ví dụ thì train_dataset có 26k dòng (FPT_VIVO), mỗi dòng là 1 cặp xử lí (spect, trainscript)
#SpectrogramDataset làm đầu vào cho AudioDataLoader để xử lí collate_fn 

#trong class này có xử lí STFT, normolize
class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self,
                 audio_conf: SpectConfig,
                 manifest_filepath: str,
                 labels: list,
                 normalize: bool = False,
                 augmentation_conf: AugmentationConfig = None):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Config containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: List containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augmentation_conf(Optional): Config containing the augmentation parameters
        """
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(SpectrogramDataset, self).__init__(audio_conf, normalize, augmentation_conf)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        spect = self.parse_audio(audio_path)#có xử lí STFT, normolize
        transcript = self.parse_transcript(transcript_path)
        return spect, transcript

# cho đường dẫn txt nó đọc nội dung
    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return self.size

#hàm riêng tính tham chiếu được gọi trong class AudioDataLoader
#the collate_fn argument is used to collate(đối chiếu) lists of samples into batches.

#batch_sampler có 835 indicas, mỗi indice có 32 mẫu là 1 dòng của dataset
# for indices in batch_sampler:
#     batch=[dataset[i] for i in indices];//32 dòng
#     yield collate_fn(batch)
def _collate_fn(batch):
    def func(p):
        return p[0].size(1)
#sample là 1 shape (spect, ...) => sample[0] là spect phổ của câu nói, sample[1] là array các ascii của câu nói gốc
#spect này là ma trận 2 chiều các số thực (161, ...)=> sample[0].size(0)-> lấy số hàng (do kiểu tensor nên khác)
#batch được sắp xếp theo giảm dần
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)# sắp xếp batch theo sample[0].size(1), số phần tử của cột của ma trận 2  chiều spect
    longest_sample = max(batch, key=func)[0]#lấy âm thanh mà có spect có ma trận nhìu cột nhất
    freq_size = longest_sample.size(0)#lấy số dòng của file âm thanh có nhìu cột nhất ->161
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)#lấy số cột của file âm thanh có nhìu cột nhất ->380
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)#ma trận 4 chiều, input[x][0] là spect thứ x trong batchsize
    input_percentages = torch.FloatTensor(minibatch_size)#ma trận phần trăm cho từng mẫu âm thanh trong minibatch
    target_sizes = torch.IntTensor(minibatch_size)# là mảng kích thước transcript của 32  câu gốc [141, 55, ..] ,=>mẫu 1 có 141 kí tự, mẫu 2 có 55 kí tự ban đầu random
    targets = []#mảng 1 chiều các kí tự của cả 32 transcript liên tục . len(targets) là tất cả các kí tự có trong toàn bộ 32 transcript, độ dài 1898778 kí tự
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]#tensor([[0.4504, -0.987..]] là cột [0] trong sample
        target = sample[1]#ascii [9,45,93,32,..]
        seq_length = tensor.size(1)#tensor là 1 mảng 2 chiều có (161, 1108), lấy số cột của spect
        #print("###", tensor.size())
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)# thu hẹp ma trận theo chiều 1 (4 chiều 0 1 2 3), bắt đầu ở 0 và giữ lại seq_length cột
        input_percentages[x] = seq_length / float(max_seqlength)# lấy số cột của spect hiện tại / số cột max-> phần trăm
        #print("$$$$$$$$$$$$",input_percentages[x])
        target_sizes[x] = len(target)#số kí tự trong câu nói "cách để đi"->10
        targets.extend(target)# add 2 kiểu dữ liệu khác nhau vào list, vd 'aaa' và 1,  2
    targets = torch.IntTensor(targets)
    #inputs là mảng 4 chiều từng sample của 32 sample. 1 sample có tensor và ascii
    return inputs, targets, input_percentages, target_sizes

#  train_loader = AudioDataLoader(dataset=train_dataset,
#                                    num_workers=cfg.data.num_workers,
#                                    batch_sampler=train_sampler), được gọi trong main

#Link: https://pytorch.org/docs/stable/data.html
# DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
#            batch_sampler=None, num_workers=0, collate_fn=None,
#            pin_memory=False, drop_last=False, timeout=0,
#            worker_init_fn=None, *, prefetch_factor=2, 
#            persistent_workers=False)

#help
    # map-style and iterable-style datasets,
    # customizing data loading order,
    # automatic batching,
    # single- and multi-process data loading,
    # automatic memory pinning.

#tham số dataset: chỉ nguồn để load data từ đó, when accessed with dataset[idx], could read the idx-th image (kiểu map-style)
#tham số : shuffle lấy mẫu tuần tự hay xáo trộn
#tham số: sampler: chỉ định cụ thể samples nào bị xáo trộn. samples là 1 tập gồm n(batch_size) mẫu trong dataset. vd 32 mẫu
#tham số: batch_sampler là 32 mẫu
#AudioLoader trả ra cuối cùng là mảng có 835 phần tử
class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class DSRandomSampler(Sampler):
    """
    Implementation of a Random Sampler for sampling the dataset.
    Added to ensure we reset the start index when an epoch is finished.
    This is essential since we support saving/loading state during an epoch.
    """

    def __init__(self, dataset, batch_size=1, start_index=0):
        super().__init__(data_source=dataset)

        self.dataset = dataset
        self.start_index = start_index
        self.batch_size = batch_size
        ids = list(range(len(self.dataset)))
        self.bins = [ids[i:i + self.batch_size] for i in range(0, len(ids), self.batch_size)]

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = (
            torch.randperm(len(self.bins) - self.start_index, generator=g)
                .add(self.start_index)
                .tolist()
        )
        for x in indices:
            batch_ids = self.bins[x]
            np.random.shuffle(batch_ids)
            yield batch_ids

    def __len__(self):
        return len(self.bins) - self.start_index

    def set_epoch(self, epoch):
        self.epoch = epoch

    def reset_training_step(self, training_step):
        self.start_index = training_step


class DSElasticDistributedSampler(DistributedSampler):
    """
    Overrides the ElasticDistributedSampler to ensure we reset the start index when an epoch is finished.
    This is essential since we support saving/loading state during an epoch.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, start_index=0, batch_size=1):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)
        self.start_index = start_index
        self.batch_size = batch_size
        ids = list(range(len(dataset)))
        self.bins = [ids[i:i + self.batch_size] for i in range(0, len(ids), self.batch_size)]
        self.num_samples = int(
            math.ceil(float(len(self.bins) - self.start_index) / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = (
            torch.randperm(len(self.bins) - self.start_index, generator=g)
                .add(self.start_index)
                .tolist()
        )

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples
        for x in indices:
            batch_ids = self.bins[x]
            np.random.shuffle(batch_ids)
            yield batch_ids

    def __len__(self):
        return self.num_samples

    def reset_training_step(self, training_step):
        self.start_index = training_step
        self.num_samples = int(
            math.ceil(float(len(self.bins) - self.start_index) / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas


def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate,
                                                                                               tar_filename, start_time,
                                                                                               end_time)
        os.system(sox_params)
        y = load_audio(tar_filename)
        return y


def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                      augmented_filename,
                                                                                      " ".join(sox_augment_params))
        os.system(sox_params)
        y = load_audio(augmented_filename)
        return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio
