LipReader
=========

A Lip Reading program for HackGT using OpenCV

Currently this:

![screen2](https://raw.githubusercontent.com/JoshVorick/LipReader/master/screenshots/webcam.jpg)

becomes this:

![screen1](https://raw.githubusercontent.com/JoshVorick/LipReader/master/screenshots/mouth.jpg)

### Current functionality:
* **Detects face**
* **Trims video to just lips**
* **Detects features on lips**
* **Tracks features across 18 frames**
* **Loads example video of 'f' viseme**

### To be added:
* **Trim tracked features to just features that are present across most frames**
* **Use winner-take-all hash to make matrices of features more manageable**
* **Compare features from live feed against known visemes to guess at what's being said**
* **Load more visemes**

Sorry this readme isn't the best :/
