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

int main(int argc, const char *argv[]) {
	// Check for valid command line arguments, print usage
	// if no arguments were given.
	if (argc != 3) {
		cout << "usage: " << argv[0] << " </path/to/haar_cascade> </path/to/csv.ext> </path/to/device id>" << endl;
		cout << "\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection." << endl;
		cout << "\t <device id> -- The webcam device id to grab frames from." << endl;
		exit(1);
	}
	// Get the path to your CSV:
	string fn_haar = string(argv[1]);
	int deviceId = atoi(argv[2]);

	int im_width = 480;
	int im_height = 560;

		//Create haar cascade
	CascadeClassifier haar_cascade;
	haar_cascade.load(fn_haar);
	// Get a handle to the Video device:
	VideoCapture cap(deviceId);
	// Check if we can use this device at all:
	if(!cap.isOpened()) {
		cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
		return -1;
	}
	// Rectangle of where lips are relative to face
	Rect lips(100, 400, 300, im_height - 410);
	// Holds the current frame from the Video device:
	Mat frame;
	for(;;) {
		cap >> frame;
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
		for(int i = 0; i < faces.size(); i++) {
			// Process face by face:
			Rect face_i = faces[i];
			// Crop the face from the image. So simple with OpenCV C++:
			Mat face = gray(face_i);
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
			cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
			// Now perform the prediction, see how easy that is:
			// First of all draw a green rectangle around the detected face:
			rectangle(original, face_i, CV_RGB(0, 255,0), 1);
		}
		// Show the result:
		imshow("face_recognizer", original);
		if (face_resized.data) {
			// Show just the resized face
			imshow("face", face_resized);
			// Show just the lips
			imshow("lips", face_resized(lips));

			// Find lips' features
			SurfFeatureDetector detector( 100 );
			std::vector<KeyPoint> keyPoints;

			detector.detect( face_resized(lips), keyPoints );

			Mat imgKeyPoints;

			drawKeypoints( face_resized(lips), keyPoints, imgKeyPoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

			imshow("features", imgKeyPoints);
		}
		// And display it:
		char key = (char) waitKey(20);
		// Exit this loop on escape:
		if(key == 27)
			break;
	}
	return 0;
}
