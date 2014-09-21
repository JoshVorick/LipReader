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
const int im_width = 480;
const int im_height = 560;

// Returns the matric of features for the video
std::vector<std::vector<KeyPoint> > getFeaturesFromVideo(string fileName, double threshhold, string haarPath) {
	std::vector<std::vector<KeyPoint> > outF;

	string fn_haar = haarPath;
	string videoFileName;

	int framesCompleted = 0;

	//Create haar cascade
	CascadeClassifier haar_cascade;
	haar_cascade.load(fn_haar);
	// Rectangle of where lips are relative to face
	Rect lips(120, 395, 240, im_height - 405);

	// Holds the current frame from the Video device:
	std::vector<Mat> frames;
	Mat frame;
	// Vector to hold 'numframeHistory' most recent faces
	// Used to stabilize the computers idea of where the face is
	std::vector<Rect> faceHistory;

	// Get the video
	VideoCapture capNone(fileName);
	// Check if we can use this device at all:
	if(!capNone.isOpened()) {
		cout << "Couldn't open " << fileName << endl;
		return outF;
	}
	capNone >> frame;
	while(frame.data) {
		frames.push_back(frame);
		if (frames.size() > FRAME_BLUR)
			frames.erase(frames.begin());

		frame = combineImages(frames);

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
			SurfFeatureDetector detector( FEATURE_THRESHHOLD );
			std::vector<KeyPoint> keyPoints;

			detector.detect( face_resized(lips), keyPoints );
			if (outF.size() > 0)
				outF.push_back(alignNewFeatures(outF[outF.size()-1], keyPoints));
			else
				outF.push_back(keyPoints);
		}
		Mat imgTrimmed;
		std::vector<std::vector<KeyPoint> > temp = trimBadFeatures(outF, threshhold);
		if (temp.size() > 0) {
			drawKeypoints( face_resized(lips), temp[0], imgTrimmed, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
			imshow("trimmed", imgTrimmed);
		}
		char key = (char) waitKey(1);
		// Exit this loop on escapNonee:
		if(key == 27)
			break;

		framesCompleted++;
		printf("frame: %i\n", framesCompleted);

		capNone >> frame;
	}

	faceHistory.clear();
	outF = trimBadFeatures(outF, threshhold);

	return outF;
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

	int im_width = 480;
	int im_height = 560;
	int framesCompleted = 0;

	//Create haar cascade
	CascadeClassifier haar_cascade;
	haar_cascade.load(fn_haar);
	// Rectangle of where lips are relative to face
	Rect lips(120, 395, 240, im_height - 405);

	// Vector of vectors to hold the features for the "f" sound (F1.avi)
	std::vector<std::vector<KeyPoint> > sounds[NUM_SOUNDS];

	// Holds the current frame from the Video device:
	// Array to hold past couple frames
	// This way past frames can be blended so features will
	// persist from frame to frame more reliably
	std::vector<Mat> frames;
	Mat frame;

	// Vector to hold 'numframeHistory' most recent faces
	// Used to stabilize the computers idea of where the face is
	std::vector<Rect> faceHistory;
	

	sounds[NONE] = getFeaturesFromVideo("NONE.avi", 0.75, fn_haar);
	printf("sounds[NONE] size: %i\n", sounds[NONE][0].size());

	sounds[FF] = getFeaturesFromVideo("F1.avi", 0.5, fn_haar);
	printf("sounds[FF] size: %i\n", sounds[FF][0].size());

	sounds[OO] = getFeaturesFromVideo("OO1.avi", 0.4, fn_haar);
	printf("sounds[OO] size: %i\n", sounds[OO][0].size());

	sounds[JJ] = getFeaturesFromVideo("J1.avi", 0., fn_haar);
	printf("sounds[JJ] size: %i\n", sounds[JJ][0].size());

	sounds[MM] = getFeaturesFromVideo("M1.avi", 0., fn_haar);
	printf("sounds[MM] size: %i\n", sounds[MM][0].size());

	sounds[TH] = getFeaturesFromVideo("TH1.avi", 0., fn_haar);
	printf("sounds[TH] size: %i\n", sounds[TH][0].size());

	// Vector of vectors to story features of past frames
	std::vector<std::vector<KeyPoint> > featureHistory;

	// Load webcam image
	VideoCapture capCam("TEST.avi");
	capCam >> frame;
	while(frame.data) {
		frames.push_back(frame);
		if (frames.size() > FRAME_BLUR)
			frames.erase(frames.begin());

		frame = combineImages(frames);

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
			SurfFeatureDetector detector( FEATURE_THRESHHOLD );
			std::vector<KeyPoint> keyPoints;

			detector.detect( face_resized(lips), keyPoints );

			if (featureHistory.size() > 0) {
				featureHistory.push_back(alignNewFeatures(featureHistory[featureHistory.size()-1], keyPoints));
				if (featureHistory.size() > numFeatureHistory) {
					featureHistory.erase(featureHistory.begin());
				}
			}
			else 
				featureHistory.push_back(keyPoints);

			std::vector<std::vector<KeyPoint> > temp = trimBadFeatures(featureHistory, .7);
			// If there aren't enough features for tracking, lower the threshhold
			float threshhold = .65;
			while (temp.size() < 1 || (temp[0].size() < 10 && threshhold > 0)) {
				temp = trimBadFeatures(featureHistory, threshhold);
				threshhold -= .05;
			}

			Mat imgKeyPoints, imgTrimmed, imgSorted;
			// Draw features to images
			drawKeypoints( face_resized(lips), keyPoints, imgKeyPoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
			if (temp.size() > 0) {
				drawKeypoints( face_resized(lips), temp[0], imgTrimmed, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
				imshow("trimmed", imgTrimmed);

				// Calc and print difference
				double diffs[NUM_SOUNDS];
				diffs[NONE] = compareFeatures(sounds[NONE], temp);
				diffs[FF] = compareFeatures(sounds[FF], temp);
				diffs[OO] = compareFeatures(sounds[OO], temp);
				diffs[JJ] = compareFeatures(sounds[NONE], temp);
				diffs[MM] = compareFeatures(sounds[NONE], temp);
				diffs[TH] = compareFeatures(sounds[NONE], temp);

				whichSound(diffs);
			}
			// Show image
			//imshow("features", imgKeyPoints);
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
