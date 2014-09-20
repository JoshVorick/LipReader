#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include <fstream>

using namespace cv;
using namespace std;

// Compare key points by location
// Sorts top to bottom and left to right (I think)
bool compKeyPointsLocY(KeyPoint a, KeyPoint b) {
	if (a.pt.y < b.pt.y)
		return true;
	if (a.pt.y == b.pt.y && a.pt.x < b.pt.x)
		return true;
	return false;
}
// Compare key points by location
// Sorts left to right and top to bottom(I think)
bool compKeyPointsLocX(KeyPoint a, KeyPoint b) {
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

// Compare key points based on angle
// Biggest first, smallest last
bool compKeyPointsAngle(KeyPoint a, KeyPoint b) {
	return a.angle > b.angle;
}
/*
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
*/

// Calculates the distance between two features
float distFeature(KeyPoint a, KeyPoint b) {
	return (a.pt.x - b.pt.x)*(a.pt.x - b.pt.x) + (a.pt.y - b.pt.y)*(a.pt.y - b.pt.y);
}

// This functiion takes in the new features and
// tries to align them with the old ones so that
// features[0][0] and features [1][0] will be consecutive
// frames of the same feature
std::vector<KeyPoint> alignNewFeatures(std::vector<KeyPoint> oldF, std::vector<KeyPoint> newF) {
	std::vector<KeyPoint> outF;
	for (int i=0; i < oldF.size(); i++) {
		float minDist = 1000;
		KeyPoint closest;
		int indexClosest = -1;
		// Find which feature from 'newF' is closest to this feature
		for (int j=0; j < newF.size(); j++) {
			if (distFeature(newF[j], oldF[i]) < minDist) {
				minDist = distFeature(newF[j], oldF[i]);
				closest = newF[j];
				indexClosest = j;
			}
		}
		// If closest is close enough (such that it's probably the same feature)
		// Add it to the output vector and remove it from newF
		if (minDist < 5) {
			outF.push_back(closest);
			newF.erase(newF.begin() + indexClosest);
		} else if (oldF[i].size == 0 && indexClosest > -1) { // If it was a placeholder feature last frame
			outF.push_back(closest);
			newF.erase(newF.begin() + indexClosest);
		} else {
			// Otherwise set this feature to the correct location, but 0 size
			KeyPoint k(oldF[i].pt, 0);
			outF.push_back(k);
		}
	}
	//Throw all the new features (ones that weren't in last frame) into open spots (where size has been 0 for two frames)
	for (int i=0; i<oldF.size(); i++) {
		if (oldF[i].size == 0 && outF[i].size == 0 && newF.size() > 0) {
			outF[i] = newF[0];
			newF.erase(newF.begin());
		}
	}
	for (int i=0; i<newF.size(); i++) {
		outF.push_back(newF[i]);
	}
	return outF;
}
