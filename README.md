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
* **Loads example video of 'f', 'o', 'j', 'm', and 'th' visemes**
* **Trims tracked features to the ones that persist for multiple frames**
* **Brute force comparison to guess which sound is being said**

### To be added:
* **Get higher resolution video for better feature detection**
* **Load more visemes**
* **Get each viseme at a different speed (people talk at different rates)**
* **Turn guessed visemes into guessed words**

