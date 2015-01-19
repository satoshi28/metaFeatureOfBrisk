#include "stdafx.h"
#include "ExtractFeatures.hpp"

ExtractFeatures::ExtractFeatures(cv::SurfFeatureDetector detector, 
    cv::Ptr<cv::DescriptorExtractor> extractor
	)
    : m_detector(detector)
    , m_extractor(extractor)
{
	
}

ExtractFeatures::~ExtractFeatures()
{
}

bool ExtractFeatures::getFeatures(std::vector<cv::Mat>& images,
									  std::vector<Pattern>& patterns)
{
	bool extractFlag = false;

	//�t�H���_�ɂ���摜�̖������ǂݍ���
	for(int i = 0; i < images.size(); i++)
	{	
		//�O���C�X�P�[����
		cv::Mat grayImg;										
		getGray(images[i], grayImg);

		//�����ʂ̒��o
		Pattern _pattern;
		extractFlag = extractFeatures(grayImg, _pattern.keypoints, _pattern.descriptors);

		//Pattern�\���̌Q�ɒǉ�
		_pattern.image = images[i];
		patterns.push_back( _pattern );

		if (extractFlag == false)
			return false;
	}
	return true;
}

void ExtractFeatures::getGray(const cv::Mat& image, cv::Mat& gray)
{
    if (image.channels()  == 3)
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, CV_BGRA2GRAY);
    else if (image.channels() == 1)
        gray = image;
}

bool ExtractFeatures::extractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    assert(!image.empty());
    assert(image.channels() == 1);

    m_detector.detect(image, keypoints);
    if (keypoints.empty())
        return false;

    m_extractor->compute(image, keypoints, descriptors);
    if (keypoints.empty())
        return false;

    return true;
}