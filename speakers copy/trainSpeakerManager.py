import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.encoder.configs.speaker_encoder_config import SpeakerEncoderConfig

output_path = os.path.dirname(os.path.abspath(__file__))


import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig, CharactersConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

CURRENT_PATH = os.getcwd()

OUTPUT_PATH = os.path.join(
    CURRENT_PATH, "output/"
)  # path to save the train logs and checkpoint
CONFIG_OUT_PATH = os.path.join(OUTPUT_PATH, "config_se.json")
RESTORE_PATH = None  # Checkpoint to use for transfer learning if None ignore


output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(
    formatter="koneveeb",
    meta_file_train="metadata.csv",
    path="/home/egert/korpused/16kHz/"
)


config = SpeakerEncoderConfig()

# add the dataset to the config
config.datasets = [dataset_config]


#### TRAINING CONFIG ####
# The encoder data loader balancer the dataset item equally to guarantee better training and to attend the losses requirements
# It have two parameters to control the final batch size the number total of speaker used in each batch and the number of samples for each speaker

# number total of speaker in batch in training
config.num_classes_in_batch = 14
# number of utterance per class/speaker in the batch in training
config.num_utter_per_class = 8
# final batch size = config.num_classes_in_batch * config.num_utter_per_class

# number total of speaker in batch in evaluation
config.eval_num_classes_in_batch = 14
# number of utterance per class/speaker in the batch in evaluation
config.eval_num_utter_per_class = 4

# number of data loader workers
config.num_loader_workers = 6
config.num_eval_loader_workers = 6

# number of epochs
config.epochs = 10000
# loss to be used in training
config.loss = "softmaxproto"

# run eval
config.run_eval = False

# output path for the checkpoints
config.output_path = OUTPUT_PATH

# Save local checkpoint every save_step steps
config.save_step = 2000

### Model Config ###
config.model_params = {
    "model_name": "resnet",  # supported "lstm" and "resnet"
    "input_dim": 64,
    "use_torch_spec": True,
    "log_input": True,
    "proj_dim": 512,  # embedding dim
}

### Audio Config ###
# To fast train the model divides the audio in small parts. it parameter defines the length in seconds of these "parts"
config.voice_len = 2.0
# all others configs
config.audio = {
    "fft_size": 512,
    "win_length": 400,
    "hop_length": 160,
    "frame_shift_ms": None,
    "frame_length_ms": None,
    "stft_pad_mode": "reflect",
    "sample_rate": 16000,
    "resample": False,
    "preemphasis": 0.97,
    "ref_level_db": 20,
    "do_sound_norm": False,
    "do_trim_silence": False,
    "trim_db": 60,
    "power": 1.5,
    "griffin_lim_iters": 60,
    "num_mels": 64,
    "mel_fmin": 0.0,
    "mel_fmax": 8000.0,
    "spec_gain": 20,
    "signal_norm": False,
    "min_level_db": -100,
    "symmetric_norm": False,
    "max_norm": 4.0,
    "clip_norm": False,
    "stats_path": None,
    "do_rms_norm": True,
    "db_level": -27.0,
}


### Augmentation Config ###
# config.audio_augmentation = {
#     # additive noise and room impulse response (RIR) simulation similar to: https://arxiv.org/pdf/2009.14153.pdf
#     "p": 0.5,  # probability to the use of one of the augmentation - 0 means disabled
#     "rir": {"rir_path": RIR_SIMULATED_PATH, "conv_mode": "full"},  # download: https://www.openslr.org/17/
#     "additive": {
#         "sounds_path": MUSAN_PATH,
#         "speech": {"min_snr_in_db": 13, "max_snr_in_db": 20, "min_num_noises": 1, "max_num_noises": 1},
#         "noise": {"min_snr_in_db": 0, "max_snr_in_db": 15, "min_num_noises": 1, "max_num_noises": 1},
#         "music": {"min_snr_in_db": 5, "max_snr_in_db": 15, "min_num_noises": 1, "max_num_noises": 1},
#     },
#     "gaussian": {"p": 0.7, "min_amplitude": 0.0, "max_amplitude": 1e-05},
# }

config.save_json(CONFIG_OUT_PATH)

print(CONFIG_OUT_PATH)
if RESTORE_PATH is not None:
    command = f"python $HOME/TTS/TTS/bin/train_encoder.py --config_path {CONFIG_OUT_PATH} --restore_path {RESTORE_PATH}"
else:
    command = f"python $HOME/TTS/TTS/bin/train_encoder.py --config_path {CONFIG_OUT_PATH}"

os.system(command)
