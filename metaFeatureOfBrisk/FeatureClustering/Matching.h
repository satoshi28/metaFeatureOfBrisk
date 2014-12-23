#ifndef MATCHING_
#define MATCHING_

////////////////////////////////////////////////////////////////////
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "../Pattern.h"
#include "../CONSTANT.h"


/**
 * Store the image data and computed descriptors of target pattern
 */
class Matching
{
public:
	/**
     *
     */
    Matching(cv::Ptr<cv::DescriptorMatcher>   matcher   = cv::DescriptorMatcher::create(matcherName) );
	Matching(bool enableMultipleRatioTest);

    /**
    * 
    */
	void getMatches(const std::vector<Pattern> patterns, std::vector< std::vector<cv::DMatch> >& matches);

	std::vector<std::vector<cv::Mat>> Matching::getHomography();
private:	
	//テンプレート画像からPatternを作成
	void train(const std::vector<cv::Mat> trainDescriptors, std::vector<cv::Ptr<cv::DescriptorMatcher> >& matchers );

	void match(std::vector<cv::KeyPoint> queryKeypoints,cv::Mat queryDescriptors,std::vector<std::vector<cv::KeyPoint>> trainKeypoints,  std::vector<cv::Ptr<cv::DescriptorMatcher> >& matchers, std::vector<cv::DMatch>& matches);

	//幾何学的整合性チェック
	bool geometricConsistencyCheck(std::vector<cv::KeyPoint> queryKeypoints, std::vector<cv::KeyPoint> trainKeypoints, std::vector<cv::DMatch>& match,  cv::Mat& homography);
private:
    
	//画像セットの数
	int dataSetSize;
	bool m_enableMultipleRatioTest;
    cv::Ptr<cv::DescriptorMatcher> m_matcher;

	int imgNumberOfAdjstment;

	std::vector<std::vector<cv::Mat>> AllHomographyes;
};


#endif