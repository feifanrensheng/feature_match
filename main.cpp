#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

void KeyPointsToPoints(vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, vector<Point2f>& points1, vector<Point2f>& points2, vector<DMatch>& matches)
{
	points1.clear();
	points2.clear();
	for ( int i = 0; i < keypoints1.size(); i++ )
	{
		points1.push_back( keypoints1[matches[i].queryIdx].pt );
		points2.push_back( keypoints2[matches[i].trainIdx].pt );
	}
}

void match_features(Mat& query, Mat& train, vector<DMatch>& matches)
{
	vector<vector<DMatch>> knn_matches;
	BFMatcher matcher(NORM_HAMMING);
	matcher.knnMatch(query, train, knn_matches, 2);

	//?????Ratio Test????ƥ??ľ??
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		//Ratio Test
		if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
			continue;

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist) min_dist = dist;
	}

	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		//???????Ratio Test?ĵ?ƥ????????
		if (
			knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
			)
			continue;

		//????????
		matches.push_back(knn_matches[r][0]);
	}
}

/*
Mat GetFundamentalMat(const vector<KeyPoint>& imgpts1,
			          const vector<KeyPoint>& imgpts2,
			          vector<KeyPoint>& imgpts1_good,
			          vector<KeyPoint>& imgpts2_good,
			          vector<DMatch>& matches,
			          const Mat& img_1,
			          const Mat& img_2
			          )
{
	//Try to eliminate keypoints based on the fundamental matrix
	//(although this is not the proper way to do this)
	vector<uchar> status(imgpts1.size());

	imgpts1_good.clear();
	imgpts2_good.clear();

	vector<KeyPoint> imgpts1_tmp;
	vector<KeyPoint> imgpts2_tmp;

	if (matches.size() <= 0)
	{
		//points already aligned...
		imgpts1_tmp = imgpts1;
		imgpts2_tmp = imgpts2;

	}
	else
	{
		GetAlignedPointsFromMatch(imgpts1, imgpts2, matches, imgpts1_tmp, imgpts2_tmp);
	}

	Mat F;
	{
		vector<Point2f> pts1, pts2;
		KeyPointsToPoints(imgpts1_tmp, pts1);
		KeyPointsToPoints(imgpts2_tmp, pts2);

		double minVal, maxVal;
		cv::minMaxIdx(pts1, &minVal, &maxVal);
		F = findFundamentalMat(pts1, pts2, FM_RANSAC, 0.006 * maxVal, 0.99, status);
	}

	vector<DMatch> new_matches;
	cout << "F keeping " << countNonZero(status) << " / " << status.size() << endl;
	for (unsigned int i = 0; i < status.size(); i++)
	{
		if (status[i])
		{
			imgpts1_good.push_back(imgpts1_tmp[i]);
			imgpts2_good.push_back(imgpts2_tmp[i]);

			if (matches.size() <= 0)
			{
				new_matches.push_back(DMatch(matches[i].queryIdx,matches[i].trainIdx,matches[i].distance));
			}
			else
			{
				new_matches.push_back(matches[i]);
			}
		}
	}

	cout << matches.size() << " matches before, " << new_matches.size() << " new matches after Fundamental Matrix\n";
	matches = new_matches;

	return F;
}
*/

int main( int argc, char** argv)
{
	if ( argc != 3 )
	{
		cout<<"usage: feature_extraction img1 img2"<<endl;
		return 1;
	}

	Mat img1 = imread( argv[1], CV_LOAD_IMAGE_COLOR );
	Mat img2 = imread( argv[2], CV_LOAD_IMAGE_COLOR );

	//
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	Ptr<ORB> orb = ORB::create( 500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE,31,20 );

	//
	orb->detect(img1, keypoints_1);
	orb->detect(img2, keypoints_2);

	//
	orb->compute(img1, keypoints_1, descriptors_1);
	orb->compute(img2, keypoints_2, descriptors_2);

	//
	BFMatcher matcher1( NORM_HAMMING );
	vector<DMatch> matches1;
	matcher1.match(descriptors_1, descriptors_2, matches1);

	//cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BFMatcher(cv::NORM HAMMING, true));    //use in opencv2.x
	BFMatcher matcher2( NORM_HAMMING, true);
	vector<DMatch> matches2;
	matcher2.match(descriptors_1, descriptors_2, matches2);

	vector<DMatch> matches3;
	match_features(descriptors_1, descriptors_2, matches3);

	vector<Point2f> points1;
	vector<Point2f> points2;
	KeyPointsToPoints(keypoints_1, keypoints_2, points1, points2, matches1);
	

	//compute F
	Mat mask;
	vector<uchar> status(keypoints_1.size());
	Mat F = findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99, status);

	vector<DMatch> new_matches;
	cout << "F keeping " << countNonZero(status) << " / " << status.size() << endl;  
	for (unsigned int i = 0; i < status.size(); i++)
	{
		if (status[i])
		{


			if (matches1.size() <= 0)
			{
				new_matches.push_back(DMatch(matches1[i].queryIdx,matches1[i].trainIdx,matches1[i].distance));
			}
			else
			{
				new_matches.push_back(matches1[i]);
			}
		}
	}

	cout << matches1.size() << " matches before, " << new_matches.size() << " new matches after Fundamental Matrix\n";



	Mat img_match1;
	Mat img_match2;
	Mat img_match3;
	Mat img_match4;

	drawMatches(img1, keypoints_1, img2, keypoints_2, matches1, img_match1 );
	drawMatches(img1, keypoints_1, img2, keypoints_2, matches2, img_match2 );
	drawMatches(img1, keypoints_1, img2, keypoints_2, matches3, img_match3 );
	drawMatches(img1, keypoints_1, img2, keypoints_2, new_matches, img_match4);



	imshow("BFMatcher", img_match1);
	imshow("jiao cha match", img_match2);
	imshow("ratio_matcher", img_match3);
	imshow("after f", img_match4);

	waitKey(0);

	return 0;
}