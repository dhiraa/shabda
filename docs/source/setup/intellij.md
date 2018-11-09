# IDE Setup

A decent IDE integration is a good start for any development.

Since our motive of supporting different platforms, naturally we end up in dealing with different programing languages.

- **Python** for Tensorflow
- **Kotlin** for Android
- **Java** for Audio libraries
- **Scala** for Web developemnt or Big data, if any.
- **C++** for any performance requirements

## Build Tool

- Gradle has been identifed the build tool for JVM languages. (Android by default uses Gradle)

## Python Support

Intellij Module configuration is provided along with this repo @
```
/path/to/shabda/src/main/python/shabda/python.iml

IntelliJ
File -> Project Structure -> Modules
Select Shabda (2nd one)
Go to Dependencies tab -> Select a python environment eg:shabda
 
```
