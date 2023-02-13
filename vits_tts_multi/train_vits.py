import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig, CharactersConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager

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

CURRENT_PATH = os.getcwd()

OUTPUT_PATH = os.path.join(
    CURRENT_PATH, "output/"
)  # path to save the train logs and checkpoint
CONFIG_OUT_PATH = os.path.join(OUTPUT_PATH, "config_se.json")
RESTORE_PATH = None  # Checkpoint to use for transfer learning if None ignore

dataset_config = BaseDatasetConfig(
    formatter="koneveeb",
    meta_file_train="korp.sisukord",
    path="/home/egert/korpused/16kHz/"
)
audio_config = VitsAudioConfig(
    sample_rate=16000,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=8000


)

config = VitsConfig(
    audio=audio_config,
    # start_by_longest=True,
    batch_size=16,
    eval_batch_size=8,
    batch_group_size=5,
    num_loader_workers=12,
    num_eval_loader_workers=6,
    run_eval=True,
    precompute_num_workers=12,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="multilingual_cleaners",
    use_phonemes=True,
    phoneme_language="et",
    phoneme_cache_path=os.path.join(OUTPUT_PATH, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=100,
    print_eval=True,
    mixed_precision=True,
    min_text_len=0,
    max_text_len=400,
    min_audio_len=0,
    max_audio_len=16000 * 30,
    test_sentences=test_sentences,
    output_path=OUTPUT_PATH,
    datasets=[dataset_config],
    # use_speaker_embedding=True,
    

    characters=CharactersConfig(
        characters_class="TTS.tts.utils.text.characters.IPAPhonemes",
  
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters="^iy\u0268\u0289\u026fu\u026a\u028f\u028ae\u00f8\u0258\u0259\u0275\u0264o\u025b\u0153\u025c\u025e\u028c\u0254\u00e6\u0250a\u0276\u0251\u0252\u1d7b\u0298\u0253\u01c0\u0257\u01c3\u0284\u01c2\u0260\u01c1\u029bpbtd\u0288\u0256c\u025fk\u0261q\u0262\u0294\u0274\u014b\u0272\u0273n\u0271m\u0299r\u0280\u2c71\u027e\u027d\u0278\u03b2fv\u03b8\u00f0sz\u0283\u0292\u0282\u0290\u00e7\u029dx\u0263\u03c7\u0281\u0127\u0295h\u0266\u026c\u026e\u028b\u0279\u027bj\u0270l\u026d\u028e\u029f\u02c8\u02cc\u02d0\u02d1\u028dw\u0265\u029c\u02a2\u02a1\u0255\u0291\u027a\u0267\u02b2\u025a\u02de\u026b",
        punctuations="!'(),-.:;? …",
    ),
    
    # characters=CharactersConfig(
    #     characters_class="TTS.tts.models.vits.VitsCharacters",
    #     pad="<PAD>",
    #     eos="<EOS>",
    #     bos="<BOS>",
    #     blank="<BLNK>",
    #     characters="!¡'(),-.:;¿? abcdefghijklmnopqrstuvwxyzõäöüšž‘’‚“`”„…–^",
    #     punctuations="!¡'(),-.:;¿? ",
    # ),
    cudnn_benchmark=False,
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)
ap.do_trim_silence = True
ap.ref_level_db = 30


# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
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

SPEAKER_ENCODER_CONFIG_PATH = "/home/egert/TTS-train/speakers/output/config_se.json"
SPEAKER_ENCODER_MODEL_PATH = "/home/egert/TTS-train/speakers/output/run-January-13-2023_02+25PM-0000000/checkpoint_36000.pth"


speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")

# config.num_speakers = speaker_manager.num_speakers
config.model_args.num_speakers = speaker_manager.num_speakers
config.model_args.use_speaker_embedding = speaker_manager.num_speakers > 1
print("SPEAKER_COUNT:", speaker_manager.num_speakers)

# init model
model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)

if RESTORE_PATH is not None:
    model.load_checkpoint(config=config, checkpoint_path=RESTORE_PATH)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
#trainer.fit_with_largest_batch_size(starting_batch_size=32)
trainer.fit()

