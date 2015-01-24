#include "stdafx.h"
#include "Matching.h"

Matching::Matching()
{
}


void Matching::getMatches(const std::vector<Pattern> patterns, std::vector< std::vector<cv::DMatch> >& clusterMatches)
{
	dataSetSize = patterns.size();	//�摜�Z�b�g�̐�
	imgNumberOfAdjstment = 0;		//matches�̃}�b�`���O�����摜ID���C�����邽�߂̕ϐ�

	//
	//���ׂẲ摜���}�b�`���O����
	for(int i = 0; i < dataSetSize; i++)
	{
		//�����p�ꎞ�ϐ��錾
		std::vector<cv::KeyPoint> queryKeypoints;				//�}�b�`���O��������_
		cv::Mat queryDescriptors;								//�}�b�`���O���������
		std::vector<std::vector<cv::KeyPoint>> trainKeypoints;	//�}�b�`���O���������_
		std::vector<cv::Mat> trainDescriptors;					//�}�b�`���O����������
		std::vector<cv::DMatch> matches;						//�}�b�`���O�y�A

		//macher��p��
		std::vector< cv::Ptr<cv::DescriptorMatcher> > matchers( dataSetSize -1 );
		for(int k = 0; k < dataSetSize -1; k++)
		{
			matchers[k] = cv::DescriptorMatcher::create(matcherName);
		}

		//�f�[�^�̃R�s�[
		queryKeypoints = patterns[i].keypoints;
		queryDescriptors = patterns[i].descriptors;

		//�}�b�`���O���������ʂ����ׂăR�s�[
		for(int j = 0; j < dataSetSize; j++)
		{
			//�N�G���摜�ȊO�̌P�������_�C�����ʂɊi�[
			if(i != j)
			{
				trainDescriptors.push_back( patterns[j].descriptors );
				trainKeypoints.push_back(patterns[j].keypoints);
			}
		}

		//�P���f�[�^��matchers�ɒǉ�
		match(queryKeypoints, queryDescriptors, trainDescriptors, trainKeypoints, matches);

		//�}�b�`���O�y�A���摜���Ƃɕ������Ă���vector�z��Ɋi�[
		clusterMatches.push_back(matches);

		//matches�̉摜�ԍ��C���p
		imgNumberOfAdjstment++;
	}
}


void Matching::train(const cv::Mat trainDescriptors, cv::Ptr<cv::DescriptorMatcher>& matcher)
{
	std::vector<cv::Mat> descriptors(1);

	// API of cv::DescriptorMatcher is somewhat tricky
	// First we clear old train data:
	matcher->clear();

	// Then we add vector of descriptors (each descriptors matrix describe one image). 
	// This allows us to perform search across multiple images:

	descriptors[0]= trainDescriptors.clone();
	matcher->add(descriptors);

	// After adding train data perform actual train:
	matcher->train();
}



void Matching::match(const std::vector<cv::KeyPoint> queryKeypoints,const cv::Mat queryDescriptors,
	const std::vector<cv::Mat> trainDescriptors,const std::vector<std::vector<cv::KeyPoint>> trainKeypoints, std::vector<cv::DMatch>& matches)
{
	matches.clear();		//������
	int imgNumber = 0;		//�}�b�`���O����Ă���摜��ID

	//�ŋߖT�_�̒T��
	for(int i = 0; i < dataSetSize -1 ; i++)
	{
		cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(matcherName);
		std::vector< std::vector<cv::DMatch>>  knnMatches;

		//k�ߖT����������.
		train(trainDescriptors[i], matcher);
		matcher->knnMatch(queryDescriptors, knnMatches, 2);

		//ratio test��ʉ߂����}�b�`���O�y�A
		std::vector<cv::DMatch> correctMatches;

		//ratio test
		for(int j = 0; j < knnMatches.size(); j++)
		{
			if(knnMatches[j].empty() == false)
			{
				 cv::DMatch& bestMatch = knnMatches[j][0];
				 cv::DMatch& betterMatch = knnMatches[j][1];

				float distanceRatio = bestMatch.distance / betterMatch.distance;

				//�����̔䂪0.8�ȉ��̓��������ۑ�
				if(distanceRatio < minRatio)
				{
					if(i == imgNumberOfAdjstment)
					{
						imgNumber = imgNumberOfAdjstment + 1;
					}
					bestMatch.imgIdx = imgNumber;
					correctMatches.push_back(bestMatch);
				}
			}
		}
		//�􉽊w�I�������`�F�b�N
		bool passFlag = geometricConsistencyCheck(queryKeypoints, trainKeypoints[i], correctMatches);

		//�􉽊w�I�������`�F�b�N�ɒʉ߂������̂̂ݓo�^����
		if(passFlag == true){
			//�v�f�̈ڂ��ւ�
			for(int k = 0; k < correctMatches.size(); k++)
			{
				matches.push_back(correctMatches[k]);
			}

		}

		//������
		knnMatches.clear();
		correctMatches.clear();
		imgNumber++;
	}
}

bool Matching::geometricConsistencyCheck(std::vector<cv::KeyPoint> queryKeypoints, std::vector<cv::KeyPoint> trainKeypoints, std::vector<cv::DMatch>& matches)
{
	if(matches.size() < 8)
	{
		matches.clear();
		return false;
	}
	std::vector<cv::Point2f>  queryPoints, trainPoints; 
	for(int i = 0; i < matches.size(); i++)
	{
		queryPoints.push_back(queryKeypoints[matches[i].queryIdx].pt);
		trainPoints.push_back(trainKeypoints[matches[i].trainIdx].pt);
	}

	//�􉽊w�I�������`�F�b�N
	std::vector<unsigned char> inliersMask(queryPoints.size() );

	//�􉽊w�I�������`�F�b�N�ɂ���ē�����l�𒊏o
	cv::Mat homography = cv::findHomography( queryPoints, trainPoints, CV_FM_RANSAC, 10, inliersMask);
	/*
	//Homography�s�񂪐�����������
	bool isGoodHomography = niceHomography(homography);
	if(isGoodHomography == false)
		return false;
		*/
	std::vector<cv::DMatch> inliers;
	for(size_t i =0 ; i < inliersMask.size(); i++)
	{
		if(inliersMask[i])
			inliers.push_back(matches[i]);
	}

	matches.swap(inliers);
	return true;
}

bool Matching::niceHomography(const cv::Mat H)
{
	const double det = H.at<double>(0,0) * H.at<double>(1,1) - H.at<double>(1,0) * H.at<double>(0,1);
	if (det < 0)
	  return false;
	
	const double N1 = sqrt( H.at<double>(0,0) * H.at<double>(0,0) + H.at<double>(1,0) * H.at<double>(1,0) );
	if (N1 > 4 || N1 < 0.1)
	  return false;
	
	const double N2 = sqrt( H.at<double>(0,1) * H.at<double>(0,1) + H.at<double>(1,1) * H.at<double>(1,1) );
	if (N2 > 4 || N2 < 0.1)
	  return false;

	const double N3 = sqrt( H.at<double>(2,0) * H.at<double>(2,0) + H.at<double>(2,1) * H.at<double>(2,1) );
	if (N3 > 0.002)
	  return false;
	
	return true;
}