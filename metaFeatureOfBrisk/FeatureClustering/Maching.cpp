#include "stdafx.h"
#include "Matching.h"

Matching::Matching(cv::Ptr<cv::DescriptorMatcher> matcher)
	: m_matcher(matcher)
{
}

Matching::Matching(bool flag)
{
	m_enableMultipleRatioTest = flag;
}

void Matching::getMatches(const std::vector<Pattern> patterns, std::vector< std::vector<cv::DMatch> >& clusterMatches)
{
	dataSetSize = patterns.size();
	imgNumberOfAdjstment = 0;

	//
	//���ׂẲ摜���}�b�`���O����
	for(int i = 0; i < dataSetSize; i++)
	{
		//�����p�ϐ��錾
		std::vector<cv::KeyPoint> queryKeypoints;
		cv::Mat queryDescriptors;						//�N�G���f�B�X�N���v�^
		std::vector<std::vector<cv::KeyPoint>> trainKeypoints;
		std::vector<cv::Mat> trainDescriptors;			//�P���f�B�X�N���v�^
		std::vector<cv::DMatch> matches;				//�}�b�`���O�œ���ꂽ����


		//macher��p��
		std::vector< cv::Ptr<cv::DescriptorMatcher> > matchers( dataSetSize -1 );
		for(int k = 0; k < dataSetSize -1; k++)
		{
			matchers[k] = cv::DescriptorMatcher::create(matcherName);
		}


		//������
		queryDescriptors = patterns[i].descriptors;		//�N�G���摜
		queryKeypoints = patterns[i].keypoints;

		for(int j = 0; j < dataSetSize; j++)
		{
			//�N�G���摜�ȊO���P���f�B�X�N���v�^�Ɋi�[
			if(i != j)
			{
				trainDescriptors.push_back( patterns[j].descriptors );
				trainKeypoints.push_back(patterns[j].keypoints);
			}
		}

		//train�f�[�^��matchers�ɒǉ�
		train(trainDescriptors, matchers);

		// Get matches
		match( queryKeypoints, queryDescriptors,trainKeypoints,  matchers, matches);

		//���ʂ��i�[
		clusterMatches.push_back(matches);

		for(int j =0; j < dataSetSize; j++)
		{
			//�N�G���摜�ȊO���P���f�B�X�N���v�^�Ɋi�[
			if(i != j)
			{
				cv::Mat img1,img2,result;
				std::vector<cv::DMatch> match;
				img1 = patterns[i].image.clone();
				img2 = patterns[j].image.clone();

				for(int k = 0; k < matches.size();k++)
				{
					if(j == matches[k].imgIdx)
					{
						match.push_back(matches[k]);
					}
				}

				cv::drawMatches(img1,patterns[i].keypoints,img2 ,patterns[j].keypoints, match, result);

				static int count = 0;
				std::stringstream ss;
				ss << count;
				std::string name = "matching";
				name +=  ss.str();
				name += ".jpg";
				cv::imwrite(name,result);
				count++;

			}
		}

		//matches�̉摜�ԍ��C���p
		imgNumberOfAdjstment++;
	}
}


void Matching::train(const std::vector<cv::Mat> trainDescriptors, std::vector<cv::Ptr<cv::DescriptorMatcher> >& matchers)
{

	std::vector<cv::Mat> descriptors(1);

	for(int i = 0; i < trainDescriptors.size(); i++)
	{
		// API of cv::DescriptorMatcher is somewhat tricky
		// First we clear old train data:
		matchers[i]->clear();

		// Then we add vector of descriptors (each descriptors matrix describe one image). 
		// This allows us to perform search across multiple images:

		descriptors[0]= trainDescriptors[i].clone();
		matchers[i]->add(descriptors);

		// After adding train data perform actual train:
		matchers[i]->train();
	}
}



void Matching::match(std::vector<cv::KeyPoint> queryKeypoints,cv::Mat queryDescriptors,
				std::vector<std::vector<cv::KeyPoint>> trainKeypoints,
				std::vector<cv::Ptr<cv::DescriptorMatcher> >& matchers, std::vector<cv::DMatch>& matches)
{

	if(m_enableMultipleRatioTest == true)
	{
		matches.clear();

		//�}�b�`���O���i�[
		std::vector< std::vector<cv::DMatch>> patternMatches(dataSetSize -1);
		std::vector<cv::DMatch> tmpMatches;


		//�ŋߖT�_�̒T��
		for(int i = 0; i < dataSetSize -1 ; i++)
		{
			//knn�}�b�`���O
			std::vector< std::vector<cv::DMatch>>  knnMatches;

			// query��matcher�ɕۑ�����Ă�������ʂ�knn�\���̂�p���čŋߖT�_����������.
			matchers[i]->knnMatch(queryDescriptors, knnMatches, 1);

			//_matches��m_knnMathes�̗v�f���R�s�[
			for(int l = 0; l < knnMatches.size(); l++)
			{
				if(knnMatches[l].empty() == false)						//�}�b�`���O�����݂��邩
					patternMatches[i].push_back(knnMatches[l][0]);
			}

			//������
			knnMatches.clear();

		}

		//�N�G�������_�̐��܂Ń��[�v
		for (size_t j = 0; j <patternMatches[0].size(); j++)
		{
			float worstdistance = 0;		//�ł������}�b�`���O�̃��[�N���b�h����(0�����ɂȂ邱�Ƃ͂Ȃ�)
			int worstId=-1;					//�ł������}�b�`���O��ID

			//�ł������}�b�`���O�̒T��
			for(size_t k = 0; k < dataSetSize - 1 ; k++)
			{
				if(worstdistance <= patternMatches[k][j].distance)
				{
					worstdistance =  patternMatches[k][j].distance;
					worstId = k;
				}
			}

			for(size_t i=0; i< dataSetSize - 1 ; i++)
			{

				if(worstId != i)
				{
					cv::DMatch& currentMatch   =patternMatches[i][j];
					float distanceRatio = currentMatch.distance / worstdistance;

					// Pass only matches where distance ratio between 
					// nearest matches is greater than 1.5 (distinct criteria)
					if (distanceRatio < minRatio)
					{
						//
						currentMatch.imgIdx = i;
						//������l���i�[
					    tmpMatches.push_back(currentMatch);
					}
				}
			}
		}
		//�􉽊w�I�������`�F�b�N�̉�����
		std::vector< std::vector<cv::DMatch>> correctMatches(dataSetSize -1);	//multipleRatioTest�Œʉ߃y�A���摜���Ƃɕۑ�

		for(int i = 0; i < dataSetSize -1; i++)
		{
			for(int j =0; j < tmpMatches.size(); j++)
			{
				if( i== tmpMatches[j].imgIdx)
					correctMatches[i].push_back(tmpMatches[j]);
			}
		}

		for(int i = 0; i < dataSetSize -1; i++)
		{
			//�􉽊w�I�������`�F�b�N
			bool passFlag = geometricConsistencyCheck(queryKeypoints, trainKeypoints[i], correctMatches[i]);
			//�􉽊w�I�������`�F�b�N�ɒʉ߂������̂̂ݓo�^����
			if(passFlag == true){
				//�v�f�̈ڂ��ւ�
				for(int k = 0; k < correctMatches[i].size(); k++)
				{
					matches.push_back(correctMatches[i][k]);
				}
			}
		}


	}else
	{
		matches.clear();

		int imgNumber = 0;

		//�ŋߖT�_�̒T��
		for(int i = 0; i < dataSetSize -1 ; i++)
		{
			//knn�}�b�`���O
			std::vector< std::vector<cv::DMatch>>  knnMatches;

			// query��matcher�ɕۑ�����Ă�������ʂ�knn�\���̂�p���čŋߖT�_����������.
			matchers[i]->knnMatch(queryDescriptors, knnMatches, 2);

			//
			std::vector<cv::DMatch> correctMatches;


			//ratio test
			for(int j = 0; j < knnMatches.size(); j++)
			{
				if(knnMatches[j].empty() == false)
				{
					 cv::DMatch& bestMatch = knnMatches[j][0];
					 cv::DMatch& betterMatch = knnMatches[j][1];

					float distanceRatio = bestMatch.distance / betterMatch.distance;

					//�����̔䂪1.5�ȉ��̓��������ۑ�
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
}

bool Matching::geometricConsistencyCheck(std::vector<cv::KeyPoint> queryKeypoints, std::vector<cv::KeyPoint> trainKeypoints, std::vector<cv::DMatch>& match)
{
	if(match.size() < 8)
	{
		match.clear();
		return false;
	}
	std::vector<cv::Point2f>  queryPoints, trainPoints; 
	for(int i = 0; i < match.size(); i++)
	{
		queryPoints.push_back(queryKeypoints[match[i].queryIdx].pt);
		trainPoints.push_back(trainKeypoints[match[i].trainIdx].pt);
	}

	//�􉽊w�I�������`�F�b�N
	std::vector<unsigned char> inliersMask(queryPoints.size() );

	//�􉽊w�I�������`�F�b�N�ɂ���ē�����l�𒊏o
	cv::findHomography( queryPoints, trainPoints, CV_FM_RANSAC, 10, inliersMask);

	std::vector<cv::DMatch> inliers;
	for(size_t i =0 ; i < inliersMask.size(); i++)
	{
		if(inliersMask[i])
			inliers.push_back(match[i]);
	}

	match.swap(inliers);
	return true;
}