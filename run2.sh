# Read text from file
text=$(cat egert/text.txt)

tts --text "$text" \
        --model_name tts_models/et/cv/vits \
        --out_path out/test.wav \
        --use_cuda true \
        #--vocoder_name $vocoder \