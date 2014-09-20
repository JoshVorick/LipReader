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

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
//using namespace cv::face;
using namespace std;

// Compare key points by location
// Sorts top to bottom and left to right (I think)
bool compKeyPointsLoc(KeyPoint a, KeyPoint b) {
	if (a.pt.x < b.pt.x)
		return true;
	if (a.pt.x == b.pt.x && a.pt.y < b.pt.y)
		return true;
	return false;
}

// Compare key points based on size
// Biggest first, smallest last
bool compKeyPointsSize(KeyPoint a, KeyPoint b) {
	return a.size > b.size;
}

std::vector<float> convertToFloatArray(std::vector<KeyPoint> pts) {
	// Trim down to the 80 biggest features
	if (pts.size() > 20) {
		std::sort(pts.begin(), pts.end(), compKeyPointsSize);
		pts.erase(pts.begin() + 20, pts.end());
	}
	//Sort by location to try and standardize their order
	std::sort(pts.begin(), pts.end(), compKeyPointsLoc);
	
	std::vector<float> arr;

	for(int i=0; i<pts.size(); i++) {
		arr.push_back( pts[i].pt.x );
		arr.push_back( pts[i].pt.y );
		arr.push_back( pts[i].size );
		arr.push_back( pts[i].angle );
	}

	//Make the vector 80 long if it isn't already
	while (arr.size() < 80) {
		arr.push_back(0);
	}
	return arr;
}

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
	int numFrameHistory = 5;
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
	std::vector<std::vector<float> > fSound;

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
			fSound.push_back(convertToFloatArray(keyPoints));
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
		//if (faceHistory.size() > 0)
			//avg_face = faceHistory[0];
		// Crop the face from the image. So simple with OpenCV C++:
		Mat gface = gray(avg_face);
		// Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
		// verify this, by reading through the face recognition tutorial coming with OpenCV.
		// Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
		// input data really depends on the algorithm used.
		//
		// I strongly encourage you to play around with the algorithms. See which work best
		// in your scenario, LBPH should always be a contender for robust face recognition.
		//
		// Since I am showing the Fisherfaces algorithm here, I also show how to resize the
		// face you have just found:
		cv::resize(frame(avg_face), face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
		// Now perform the prediction, see how easy that is:
		// First of all draw a green rectangle around the detected face:
		rectangle(original, avg_face, CV_RGB(0, 255,0), 1);

		// Show the result:
		imshow("face_recognizer", original);
		if (face_resized.data) {
			// Show just the resized face
			// imshow("face", face_resized);
			// Show just the lips
			// imshow("lips", face_resized(lips));

			// Find lips' features
			SurfFeatureDetector detector( 200 );
			std::vector<KeyPoint> keyPoints;

			detector.detect( face_resized(lips), keyPoints );
			std::sort(keyPoints.begin(), keyPoints.end(), compKeyPointsSize);

			Mat imgKeyPoints;

			drawKeypoints( face_resized(lips), keyPoints, imgKeyPoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
			imshow("features", imgKeyPoints);
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
