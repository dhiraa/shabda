# Ubuntu Environment Setup

## Java

```
sudo apt-get update
sudo apt-get install openjdk-8-jdk -y
sudo apt-get install unzip
#check java version
java -version
```

## Gradle

Before downloading the gradle pacakge check the official site for the latest version [here](https://gradle.org/releases/)

For tutorials check [here](https://gradle.org/guides/)!

```
mkdir /opt/gradle
wget https://services.gradle.org/distributions/gradle-4.10.2-bin.zip
unzip gradle-4.10.2-bin.zip

or

curl -s https://get.sdkman.io | bash - install sdk
#install gradle 3.5 (or any version 3.0+ or 4.0+)
sdk install gradle 3.5 
#check the installed version 
gradle -v
#switching between versions 
sdk use gradle 4.0
```

Add the binaries to user environment path:

```
vim ~/.bashrc
    # add following to the file
    export PATH=$PATH:/opt/gradle/gradle-4.10.2/bin
source ~/.bashrc
#test the installation
gradle -v
```

## Python Environment

```
conda create -n shabda python=3.6
source activate shabda
cd path/to/shabda/
pip install -e .[tensorflow-cpu] 
#or
pip install -e .[tensorflow-gpu]
pip install -r requirements.txt
```


## Git Configure
https://help.github.com/articles/setting-your-commit-email-address-in-git/

```
git clone https://github.com/dhiraa/shabda

#to push without entering password everytime
git remote rm origin
git remote add origin  https://USERNAME:PASSWORD@github.com/dhiraa/shabda.git

#checkout remote branch
git checkout -b branch_name origin/branch_name

```
