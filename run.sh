#!/bin/bash

# List of vocoders

vocoder="vocoder_models/en/blizzard2013/hifigan_v2"

# Read vocoder name from text file

for vocoder in $(cat egert/vocoders.txt); do
    # Skip line if empty
    [[ -z $vocoder ]] && continue
    # Skip line if it starts with a comment
    [[ $vocoder == \#* ]] && continue

    echo $vocoder
    # Replace slashes with underscores, strip leading "vocoder_models/"
    vocoder_filename=${vocoder//\//_}
    vocoder_filename=${vocoder_filename#vocoder_models_}

    tts --text "Eesti Energiat aprillist juhtima asuva Andrus Durejko sõnul on energiasektor tervik, kus peab lähtuma võimalikult pikast plaanist ning suuna muutmiseks kulub vähemalt viis kuni kümme aastat." \
        --model_path egert/run-December-19-2022_12+53PM-061ac431/best_model_264018.pth \
        --config_path egert/run-December-19-2022_12+53PM-061ac431/config.json \
        --out_path out/glowtts/$vocoder_filename.wav \
        --vocoder_name $vocoder \
        --use_cuda true \
        
done
    #--vocoder_path path/to/vocoder.pth \
    #--vocoder_config_path path/to/vocoder_config.json
