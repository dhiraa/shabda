# Shabda Project Structure

Project has three components
- Python based Model development framework using Tensorflow
- Java/Kotlin framework to imitate Python pre-processing and postprocessing steps
- Android application that uses the model using TFLite and the Jar from above step.  


```

|
|- android : Android Applications
|- bin : Any runnable
|- build : Gradle build output
|- data : Open Datasets 
|- docs : Documentation
|- gradle : Gradle Wrapper executable JAR & configuration properties
|- intellij : 
|- notebooks : 
|- shabda : Python source code
|- src : JVM source code
|- .gitignore :
|- .pylint.rc :
|- .travis.yml : Travis CI build script
|- build.gradle : Gradle build script for configuring the current project 
|- gradlew : Gradle Wrapper script for Unix-based systems
|- gradle.bat : Gradle Wrapper script for Windows
|- LICENSE
|- README.md
|- readthedocs.yml : Readthedocs build script
|- requirements.txt : Python library requirements
|- settings.gradle : Gradle settings script for configuring the Gradle build
|- setup.py


```