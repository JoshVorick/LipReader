/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *	 notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *	 notice, this list of conditions and the following disclaimer in the
 *	 documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *	 may be used to endorse or promote products derived from this software
 *	 without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

/*#include <cv.h>
#include <face.hpp>
#include <highgui.hpp>
#include <imgproc.hpp>
#include <objdetect.hpp>
*/
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "helperFunctions.cpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
//using namespace cv::face;
using namespace std;

const int numFrameHistory = 5;    // Used to stabalize rectangle around face
const int numFeatureHistory = 18; // Used to guess which sound is being made

int main(int argc, const char *argv[]) {
	// Check for valid command line arguments, print usage
	// if no arguments were given.
	if (argc < 2) {
		cout << "usage: " << argv[0] << " </path/to/haar_cascade> </path/to/csv.ext> </path/to/device id>" << endl;
		cout << "\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection." << endl;
		cout << "\t </path/to/video> -- (OPTIONAL) The path to video to decode. If no argument, will try to use webcam." << endl;
		exit(1);
	}
	// Get the path to your CSV:
	string fn_haar = string(argv[1]);
	string videoFileName;
	int deviceId = 0;
	 if (argc > 2)
		videoFileName = argv[2];

	int im_width = 480;
	int im_height = 560;
	int framesCompleted = 0;

		//Create haar cascade
	CascadeClassifier haar_cascade;
	haar_cascade.load(fn_haar);
	// Get a handle to the Video device:
	//VideoCapture cap(deviceId);
	VideoCapture cap(videoFileName);
	// Check if we can use this device at all:
	if(!cap.isOpened()) {
		cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
		return -1;
	}
	// Rectangle of where lips are relative to face
	Rect lips(120, 395, 240, im_height - 405);

	// Vector of vectors to hold the features for the "f" sound (F1.avi)
	std::vector<std::vector<KeyPoint> > fSound;

	// Holds the current frame from the Video device:
	Mat frame;
	// Vector to hold 'numframeHistory' most recent faces
	// Used to stabilize the computers idea of where the face is
	std::vector<Rect> faceHistory;
	cap >> frame;
	while(frame.data) {
		// Clone the current frame:
		Mat original = frame.clone();
		// Convert the current frame to grayscale:
		Mat gray;
		cvtColor(original, gray, CV_BGR2GRAY);
		// Find the faces in the frame:
		vector< Rect_<int> > faces;
		haar_cascade.detectMultiScale(gray, faces);

		// Get most prominent face from 'faces'
		Rect face(0,0,0,0);
		for (int i = 0; i<faces.size(); i++) {
			if (faces[i].area() > face.area()) {
				face = faces[i];
			}
		}
		// Add to list of faces
		faceHistory.push_back(face);
		// If more than 'numFrameHistory' are stored, remove one
		if (faceHistory.size() > numFrameHistory) {
			faceHistory.erase(faceHistory.begin());
		}
		// Find average bottom right and top left corners of the faces stored
		int brx, bry, tlx, tly;
		brx = bry = tlx = tly = 0;
		for (int i = 0; i < faceHistory.size(); i++) {
			Point br = faceHistory[i].br();
			Point tl = faceHistory[i].tl();
			brx += br.x;
			bry += br.y;
			tlx += tl.x;
			tly += tl.y;
		}
		if (faceHistory.size() > 0) {
			brx /= faceHistory.size();
			bry /= faceHistory.size();
			tlx /= faceHistory.size();
			tly /= faceHistory.size();
		}
		// Create the avg revtangle
		if (brx <= tlx) brx = tlx + 1;
		if (bry <= tly) bry = tly + 1;
		Rect avg_face(tlx, tly, brx - tlx, bry - tly);
		// Standardize the face's size
		Mat face_resized;
		cv::resize(frame(avg_face), face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

		if (face_resized.data) {
			SurfFeatureDetector detector( 200 );
			std::vector<KeyPoint> keyPoints;

			detector.detect( face_resized(lips), keyPoints );
			if (fSound.size() > 0)
				fSound.push_back(alignNewFeatures(fSound[fSound.size()-1], keyPoints));
			else
				fSound.push_back(keyPoints);
		}
		// And display it:
		char key = (char) waitKey(1);
		// Exit this loop on escape:
		if(key == 27)
			break;
		framesCompleted++;
		printf("frame: %i\n", framesCompleted);
		cap >> frame;
	}

	faceHistory.clear();

	// Vector of vectors to story features of past frames
	std::vector<std::vector<KeyPoint> > featureHistory;

	VideoCapture capCam(0);
	capCam >> frame;
	while(frame.data) {
		// Clone the current frame:
		Mat original = frame.clone();
		// Convert the current frame to grayscale:
		Mat gray;
		cvtColor(original, gray, CV_BGR2GRAY);
		// Find the faces in the frame:
		vector< Rect_<int> > faces;
		haar_cascade.detectMultiScale(gray, faces);
		// At this point you have the position of the faces in
		// faces. Now we'll get the faces, make a prediction and
		// annotate it in the video. Cool or what?
		Mat face_resized;

		// Get most prominent face from 'faces'
		Rect face(0,0,0,0);
		for (int i = 0; i<faces.size(); i++) {
			if (faces[i].area() > face.area()) {
				face = faces[i];
			}
		}
		// Add to list of faces
		faceHistory.push_back(face);
		// If more than 'numFrameHistory' are stored, remove one
		if (faceHistory.size() > numFrameHistory) {
			faceHistory.erase(faceHistory.begin());
		}
		// Find average bottom right and top left corners of the faces stored
		int brx, bry, tlx, tly;
		brx = bry = tlx = tly = 0;
		for (int i = 0; i < faceHistory.size(); i++) {
			Point br = faceHistory[i].br();
			Point tl = faceHistory[i].tl();
			brx += br.x;
			bry += br.y;
			tlx += tl.x;
			tly += tl.y;
		}
		if (faceHistory.size() > 0) {
			brx /= faceHistory.size();
			bry /= faceHistory.size();
			tlx /= faceHistory.size();
			tly /= faceHistory.size();
		}
		// Create the avg revtangle
		if (brx <= tlx) brx = tlx + 1;
		if (bry <= tly) bry = tly + 1;
		Rect avg_face(tlx, tly, brx - tlx, bry - tly);
		// Crop the face from the image.
		Mat gface = gray(avg_face);
		//Standardize image size
		cv::resize(frame(avg_face), face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
		// Now perform the prediction, see how easy that is:
		// First of all draw a green rectangle around the detected face:
		rectangle(original, avg_face, CV_RGB(0, 255,0), 1);

		// Show the result:
		imshow("face_recognizer", original);
		if (face_resized.data) {
			//Create feature detector
			SurfFeatureDetector detector( 200 );
			std::vector<KeyPoint> keyPoints;

			detector.detect( face_resized(lips), keyPoints );

			Mat imgKeyPoints;

			// Draw features to image
			drawKeypoints( face_resized(lips), keyPoints, imgKeyPoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
			// Show image
			imshow("features", imgKeyPoints);

			if (featureHistory.size() > 0) {
				featureHistory.push_back(alignNewFeatures(featureHistory[featureHistory.size()-1], keyPoints));
				if (featureHistory.size() > numFeatureHistory) {
					featureHistory.erase(featureHistory.begin());
				}
			}
			else 
				featureHistory.push_back(keyPoints);

			printf("x: %f\ty: %f fH: %i\tfH[0]: %i\n", featureHistory[0][0].pt.x, featureHistory[0][0].pt.y, featureHistory.size(), featureHistory[0].size());
			// compareFeatureVectorVectors(featureHistory, fSound);
		}
		// And display it:
		char key = (char) waitKey(20);
		// Exit this loop on escape:
		if(key == 27)
			break;
		capCam >> frame;
	}
	return 0;
}
