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
	//�e���v���[�g�摜����Pattern���쐬
	void train(const std::vector<cv::Mat> trainDescriptors, std::vector<cv::Ptr<cv::DescriptorMatcher> >& matchers );

	void match(std::vector<cv::KeyPoint> queryKeypoints,cv::Mat queryDescriptors,std::vector<std::vector<cv::KeyPoint>> trainKeypoints,  std::vector<cv::Ptr<cv::DescriptorMatcher> >& matchers, std::vector<cv::DMatch>& matches);

	//�􉽊w�I�������`�F�b�N
	bool geometricConsistencyCheck(std::vector<cv::KeyPoint> queryKeypoints, std::vector<cv::KeyPoint> trainKeypoints, std::vector<cv::DMatch>& match,  cv::Mat& homography);
private:
    
	//�摜�Z�b�g�̐�
	int dataSetSize;
	bool m_enableMultipleRatioTest;
    cv::Ptr<cv::DescriptorMatcher> m_matcher;

	int imgNumberOfAdjstment;

	std::vector<std::vector<cv::Mat>> AllHomographyes;
};


#endif