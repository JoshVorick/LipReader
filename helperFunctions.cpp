#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include <fstream>

using namespace cv;
using namespace std;

const double ANGLE_W = .3;
const double SIZE_W = .3;
const double DIST_W = .4;

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
		if (minDist < 20) {
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

// Trims the features that aren't apparent thoughout most frames
// THe threshhold is the percent of frames features must persist in
std::vector<std::vector<KeyPoint> > trimBadFeatures(std::vector<std::vector<KeyPoint> > arr, double threshhold) {
	std::vector<std::vector<KeyPoint> > outF;

	if (arr.size() < 1)
		return outF;
	// First make it a nice rectangle by adding filler points
	// This makes trimming it easier/neater
	int maxLen = arr[arr.size() - 1].size();
	for (int i=0; i < arr.size(); i++) {
		while (arr[i].size() < maxLen) {
			KeyPoint k(0,0,0);
			arr[i].push_back(k);
		}
	}

	// Now we start firguring which features to keep
	for (int j=0; j < maxLen; j++) {
		// Loop through one feature across all its frames to see if it persists
		int numAppearances = 0;
		KeyPoint kp = arr[0][j];
		for (int i=0; i < arr.size(); i++) {
			if (distFeature(kp, arr[i][j]) < 20 && arr[i][j].size > 0) {
				numAppearances++;
			}
			kp = arr[i][j];
		}
		// If the feature is a good one, add it to the output
		if (numAppearances > arr.size() * threshhold){
			for (int i=0; i < arr.size(); i++) {
				if (outF.size() <= i) {
					std::vector<KeyPoint> kpArr;
					kpArr.push_back(arr[i][j]);
					outF.insert(outF.begin(), kpArr);
				} else {
					outF[i].push_back(arr[i][j]);
				}
			}
		}
	}
	return outF;
}

// Sorts the array while keeping features aligned
// Yes I know its an insertion sort.
// If that bothers you, make a Pull Request
std::vector<std::vector<KeyPoint> > sortFeatures(std::vector<std::vector<KeyPoint> > arr) {
	std::vector<std::vector<KeyPoint> > outF;

	if (arr.size() < 1 || arr[0].size() < 1)
		return outF;

	// Initialize output vector to be right number of frames
	for (int i=0; i < arr.size(); i++) {
		std::vector<KeyPoint> kpArr;
		outF.push_back(kpArr);
	}
	// Find element that should go last, add it, repeat
	while (arr[0].size() > 0) {
		int indexOfMin = 0;
		KeyPoint smallest = arr[0][0];
		// Find which is smallest
		for (int i=0; i < arr[0].size(); i++) {
			if (arr[0][i].pt.x < smallest.pt.x || 
					(arr[0][i].pt.x == smallest.pt.x && arr[0][i].pt.y < smallest.pt.y)) {
				indexOfMin = i;
				smallest = arr[0][i];
			}
		}
		// Insert elements into 'outF' and remove them from 'arr'
		for (int i=0; i < arr.size(); i++) {
			outF[i].push_back(arr[i][indexOfMin]);
			arr[i].erase(arr[i].begin() + indexOfMin);
		}
	}
	return outF;
}

// Returns how dissimilar two points are
// Based on location, size, and angle
int calcDiff(KeyPoint a, KeyPoint b) {
	int dist = distFeature(a, b);
	int ds = abs(a.size - b.size);
	int da = abs(cos(a.angle) - cos(b.angle)); // Using cosine makes it so that 179 and -179 are close together
	return dist + ds + da*10;
}

// Returns % accuracy of feed more or less
// Uses |(theoretical - actual) / theoretical|
double calcPerformance(KeyPoint lib, KeyPoint feed) {
	double dist = calcDiff(lib, feed) / 20;
	double size, angle;
	if (lib.size != 0)
		size = abs(lib.size - feed.size) / lib.size;
	else
		size = dist;
	if (lib.angle != -1)
		angle = abs(lib.angle - feed.angle) / lib.angle;
	else
		angle = dist;
	return ANGLE_W*angle + SIZE_W*size + DIST_W*(1 - dist);
}

// Compare two matrices of features to see how similar they are
double compareFeatures(std::vector<std::vector<KeyPoint> > lib, std::vector<std::vector<KeyPoint> > feed) {
#if 0
	int diff = 0; // Keeps track of how similar the two vectors are
	int numWrong = 0; //Keeps track of features that aren't in both

	// Resize feed to be the same number of frames as lib
	if (feed.size() > lib.size()) {
		feed.erase(feed.begin(), feed.begin() + (feed.size() - lib.size() - 1));
	}

	if (lib.size() < 1 || feed.size() < 1)
		return 10000000;
	// Iterate through one array
	// Find the distance feedetween each feature and its closest counterpart
	for (int i=0; i < lib.size() && i < feed.size(); i++) {
		for (int j=0; j < lib[i].size(); j++) {
			// Find nearest feature from other array
			int distMin = 10000;
			KeyPoint nearest(0,0,0);
			for (int k=0; k < feed[i].size(); k++) {
				if (distFeature(lib[i][j], feed[i][k]) < distMin){
					distMin = distFeature(lib[i][j], feed[i][k]);
					nearest = feed[i][k];
				}
			}
			if (distMin > 30)
				diff += calcDiff(lib[i][j], nearest);
			else
				numWrong++;
		}
	}
	return diff;
#else
	double performance = 0;
	double maxPerformance = 0;

	// Resize feed to be the same number of frames as lib
	if (feed.size() > lib.size()) {
		feed.erase(feed.begin(), feed.begin() + (feed.size() - lib.size() - 1));
	}

	if (lib.size() < 1 || feed.size() < 1)
		return 0;
	// Iterate through one array
	// Find the distance feedetween each feature and its closest counterpart
	for (int i=0; i < lib.size() && i < feed.size(); i++) {
		for (int j=0; j < lib[i].size(); j++) {
			// Find nearest feature from other array
			int distMin = 10000;
			KeyPoint nearest(0,0,0);
			for (int k=0; k < feed[i].size(); k++) {
				if (distFeature(lib[i][j], feed[i][k]) < distMin){
					distMin = distFeature(lib[i][j], feed[i][k]);
					nearest = feed[i][k];
				}
			}
			if (distMin < 20)
				performance += calcPerformance(lib[i][j], nearest);
			maxPerformance++;
		}
	}
	return performance / maxPerformance;
#endif
}

Mat combineImages(std::vector<Mat> images) {
	assert(images.size() >= 1);

#if 1
	double weight = 1. / images.size();
	Mat outImage = weight * images[0];
	for (int i=1; i < images.size(); i++) {
		outImage += weight * images[i];
	}
	return outImage;
#else
	return images[0];
#endif
}
