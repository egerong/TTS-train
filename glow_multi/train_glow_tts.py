import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

test_sentences = [
    ["Ongi tõestatud, et inimene on tööl palju produktiivsem, kui talle antakse ka aega oma kodustega olla, kui ta ei pea liigselt üle töötama.Kõik sellised asjad."],
    ["Ja ka oskus kuulata sageli ju aitab."],
    ["Vähemalt ma arvan, et mures olijat aitab väga palju see, kui keegi teda lihtsalt kuulab ja on olemas."],
    ["Aga inimesed ei oska seda teha sageli."],
    ["Mul on tunne, et tegelikult me kõik vajame mingi hetk seda kohta, kus me saamegi vastutuse ära anda."],
    ["See ei ole tegelikult üldse halb."],
    ["See on osa elust ja me igaüks vajame seda."],
    ["Mulle meeldib ka otsustada, mulle meeldib ka vastutada."],
    ["Mul ei ole midagi selle vastu, aga vahepeal on tunne, et ma tahan, et keegi lihtsalt minu eest mingid asjad ära teeks."],
    ["Sest tegelikult, kui ma vaatan praegu tagasi, meil on nii ägedad kaks kuud olnud selles maamajakeses."],
    ["Teeme kaminat, ja me ei vaata enam nii palju telekat."],
    ["ee noh kindlasti sina oled üks väheseid ee pigem väheseid inimesi, kes oskab kuulata ja oskab seda nii-öelda mingisugust nõu anda või lohutust anda."],
    ["Aga paljud ei oskagi, nad jäävad kohmetuks."],
    ["Aga kui nüüd ee mõelda vananemise peale, siis ee noh, me muutume aasta-aastalt vanemaks, targemaks ka muidugi."],
    ["Ilusamaks ka, ma loodan."],
    ["ee mina küll tunnen, et ma nagu, mõnes mõttes lähen aastatega paremaks."],
    ["ee ja siis mul oli küll muide see mõte et ee."],
    ["Ma olen alati mõelnud, et ma tahaks ee seda inimest nagu rohkem ee tunda, või nagu saanud tunda tol hetkel."],
    ["Et selles suhtes on, see kirjaidee on küll noh päris ee tore."],
    ["Ta rääkis täpselt seda, mida nagu mina olin nagu mõelnud, aga tema lihtsalt pani selle palju paremini nagu sõnadesse ja oskas seda nii ilusasti nagu serveerida."],
    ["Ma siiamaani mõtlen selle peale."],
    ["Me oleme teineteisele väga palju praktilisi asju ka kinkinud ja ja teinud ee siin-seal mingeid ee selliseid väikseid üllatusi."],
    ["Ja ja need on ka vajalikud."]
]

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(CURRENT_PATH, "output")


# define dataset config
dataset_config = BaseDatasetConfig(
    formatter="koneveeb",
    meta_file_train="korp.sisukord",
    path="/home/egert/korpused/16kHz/"
)

# define audio config
# ❗ resample the dataset externally using `TTS/bin/resample.py` and set `resample=False` for faster training
audio_config = BaseAudioConfig(sample_rate=16000, do_trim_silence=True, trim_db=23)

# define model config
config = GlowTTSConfig(
    audio=audio_config,
    batch_size=16,
    eval_batch_size=8,
    num_loader_workers=6,
    num_eval_loader_workers=6,
    precompute_num_workers=6,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="multilingual_cleaners",
    use_phonemes=True,
    phoneme_language="et",
    phoneme_cache_path=os.path.join(OUTPUT_PATH, "phoneme_cache"),
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=OUTPUT_PATH,
    datasets=[dataset_config],
    use_speaker_embedding=True,
    min_text_len=0,
    max_text_len=600,
    min_audio_len=0,
    max_audio_len=500000,
    test_sentences=[sent[0] for sent in test_sentences],
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.num_speakers = speaker_manager.num_speakers

# init model
model = GlowTTS(config, ap, tokenizer, speaker_manager=speaker_manager)

# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.
trainer = Trainer(
    TrainerArgs(),
    config,
    OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples
)

# AND... 3,2,1... 🚀
trainer.fit()
