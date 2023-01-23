#! /bin/bash

trap 'echo Cancelled; exit' INT

src=$1
dst=$2

mkdir -p $dst
mkdir -p $dst/wavs
# cp $src/korp.sisukord $dst/metadata.csv

for file in $src/*.wav; do
    
    name="$(basename $file)"
    sox $file -c 1 -b 16 -r 16000 $dst/wavs/$name ;
    
done
