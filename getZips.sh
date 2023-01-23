#! /bin/bash

# This script will:
# download the zip files listed in the file "zipfiles.txt" to the directory "zips"
# create directories for each zip file in current directory
# unzip each zip file into its own directory

# create the directory "zips" if it doesn't exist
mkdir -p zips

# read the zip files from the file "zipfiles.txt"
while read line
do
    # download the zip file
    wget -P zips $line
    # create a directory for the zip file
    mkdir ${line##*/}
    # unzip the zip file into its own directory
    unzip -d ${line##*/} zips/${line##*/}
done < zipfiles.txt

# rename the directories to remove .zip

for d in ./*.zip ; do
    mv "$d" "${d%.zip}"
done