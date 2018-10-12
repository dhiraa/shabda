#simple script to down load all the datasets used here

Black='\033[0;30m'    
DarkGray='\033[1;30m'
RED='\033[0;31m'    
LightRed='\033[1;31m'
Green='\033[0;32m'     
LightGreen='\033[1;32m'
BrownOrange='\033[0;33m'
Yellow='\033[1;33m'
Blue='\033[0;34m'
LightBlue='\033[1;34m'
Purple='\033[0;35m'
LightPurple='\033[1;35m'
Cyan='\033[0;36m'    
LightCyan='\033[1;36m'
LightGray='\033[0;37m'
White='\033[1;37m'
NC='\033[0m' # No Color

echo -e  "I ${RED} Setup your Kaggle API (https://github.com/Kaggle/kaggle-api) ${NC}"
echo -e "I ${RED} unzip command line tool is used to unzip the files..."
s
echo -e "I ${Yellow} Downloading freesound-audio-tagging... ${NC}"
cd freesound-audio-tagging/input 
kaggle competitions download -c freesound-audio-tagging
unzip audio_test.zip
unzip audio_train.zip
chmod 666 ../input/*.csv
cd ../../





 

