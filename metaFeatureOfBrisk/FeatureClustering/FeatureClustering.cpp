
#include "stdafx.h"
#include "FeatureClustering.h"

FeatureClustering::FeatureClustering(int buget, bool enableMultipleRatioTest)
	: m_budget(buget)
	, m_enableMultipleRatioTest(enableMultipleRatioTest)
{

}

void FeatureClustering::clusterFeatures(std::vector<cv::Mat> images, Pattern& metaFeatures)
{
	ExtractFeatures extract;
	Matching matching(m_enableMultipleRatioTest);

	std::vector< std::vector<cv::DMatch> > clusterMatches;

	if(m_enableMultipleRatioTest == true)
	{
		cv::Mat falseImage = cv::imread("C:\\Users\\satoshi\\Documents\\Image\\lenna.png",1);
		images.push_back(falseImage);
	}

	//�����p
	// �����ʂ�Pattern�ɕۑ�
	extract.getFeatures(images,patterns);
	// ���ׂẲ摜���m���}�b�`���O����
	matching.getMatches(patterns, clusterMatches);
	
	std::vector<ClusterOfFeature> clusters;
	//�}�b�`���O���ʂ���N���X�^�����O�����ʂ��쐬����
	clusterDescriptors(clusterMatches, clusters);

	std::cout << "OK" << std::endl;
	for(int i = 0; i < patterns.size(); i++)
	{
		showResult(patterns[i], clusters[i]);

		cv::waitKey(0);
	}
	//�N���X�^�����O�����ʂ��烁�^�����ʂ��쐬����
	featureBudgeting(clusters, metaFeatures);

	for(int i = 0; i < patterns.size(); i++)
	{
		showMetaFeatures(patterns[i].image,clusters[i], metaFeatures);

		cv::waitKey(0);
	}
	//�㏈��
	patterns.clear();

}



void FeatureClustering::clusterDescriptors( std::vector<std::vector<cv::DMatch>> clusterMatches, std::vector<ClusterOfFeature>& clusters)
{

	int rank = 1;			//������Ԃ��痈�������_�̐�
	int metaNum = 0;
	int singleNum = 0;
	int id;
	std::vector<int> matchList;


	if(m_enableMultipleRatioTest == true)
	{

		for(int i = 0; i < clusterMatches.size(); i++)
		{
			int cols = patterns[i].descriptors.cols;

			//�����p�ϐ�
			ClusterOfFeature cluster;
			metaNum = 0;
			cluster.metaDescriptors = cv::Mat::zeros(1, cols,  CV_8U);
			cluster.singleDescriptors = cv::Mat::zeros(1, cols,  CV_8U);
			matchList.clear();

			//�摜�ԂŃ}�b�`���O�����摜��metaDescriptors�ɕۑ�
			for(int j = 1; j < clusterMatches[i].size(); j++)
			{
				id = clusterMatches[i][j -1].queryIdx;

				if(id == clusterMatches[i][j].queryIdx)
				{
					rank += 1;
				}
				else
				{
					int queryIdx = clusterMatches[i][j-1].queryIdx;		//query�̃C���f�b�N�X
					matchList.push_back(queryIdx);						//�}�b�`���O���X�g��query�̔ԍ���ۑ�

					metaNum += 1;
					cluster.metaDescriptors.resize(metaNum, 0);
					//�ۑ����� 
					//�����ʂ̍s�ɒǉ�
					cluster.metaDescriptors.row(metaNum-1) += patterns[i].descriptors.row(queryIdx);


					#if _DEBUG
					//�����_��ۑ�
					cluster.metaKeypoints.push_back(patterns[i].keypoints.at(queryIdx) );
					#endif

					//�����ʂ̃����N��ۑ�
					cluster.rankingList.push_back(rank);

					rank = 1;
				}

			}
			singleNum = 0;
			//�P�̂̓����ʂ�singleDescriptors�ɕۑ�
			for(int k = 0; k < patterns[i].descriptors.rows; k++)
			{
				for(int m = 0; m < matchList.size(); m++)
				{
					if(matchList[m] == k)
					{
						break;
					}
					if(m == matchList.size()-1)
					{
						singleNum += 1;
						cluster.singleDescriptors.resize(singleNum, 0);
						//�����ʂ̍s�ɒǉ�
						cluster.singleDescriptors.row(singleNum-1) += patterns[i].descriptors.row(k);

						#if _DEBUG
						//�����_��ۑ�
						cluster.singleKeypoints.push_back(patterns[i].keypoints.at(k) );
						#endif
					}
				}
			}
			
			//cluster�̍ő�T�C�Y��ۑ�
			cluster.maxClusterSize = clusterMatches.size();
			//�ۑ�
			clusters.push_back(cluster);

			
		}
	//ratio test�ɂ��}�b�`���O����
	}else
	{
		for(int i = 0; i < clusterMatches.size(); i++)
		{
			int cols = patterns[i].descriptors.cols;
			rank = 0;
			//�����p�ϐ�
			ClusterOfFeature cluster;
			metaNum = 0;
			cluster.metaDescriptors = cv::Mat::zeros(1, cols,  CV_8U);
			cluster.singleDescriptors = cv::Mat::zeros(1, cols,  CV_8U);
			matchList.clear();

			for(int j = 1; j < clusterMatches[i].size(); j++)
			{
				id = clusterMatches[i][j - 1].queryIdx;

				//query�ԍ�����v��������ʂ�T��
				for(int k = 0; k < clusterMatches[i].size(); k++)
				{
					if(id == clusterMatches[i][k].queryIdx)
						rank += 1;
				}

				int queryIdx = clusterMatches[i][j].queryIdx;		//query�̃C���f�b�N�X
				matchList.push_back(queryIdx);						//�}�b�`���O���X�g��query�̔ԍ���ۑ�
				metaNum += 1;

				cluster.metaDescriptors.resize(metaNum, 0);
				//�ۑ����� 
				//�����ʂ̍s�ɒǉ�
				cluster.metaDescriptors.row(metaNum-1) += patterns[i].descriptors.row(queryIdx);
		
				#if _DEBUG
				//�����_��ۑ�
				cluster.metaKeypoints.push_back(patterns[i].keypoints.at(queryIdx) );
				#endif

				//�����ʂ̃����N��ۑ�
				cluster.rankingList.push_back(rank);

				rank = 0;
			}
			//������
			singleNum = 0;
			//�P�̂̓����ʂ�singleDescriptors�ɕۑ�
			for(int k = 0; k < patterns[i].descriptors.rows; k++)
			{
				for(int m = 0; m < matchList.size(); m++)
				{
					if(matchList[m] == k)
					{
						break;
					}
					if(m == matchList.size()-1)
					{
						singleNum += 1;
						cluster.singleDescriptors.resize(singleNum, 0);
						//�����ʂ̍s�ɒǉ�
						cluster.singleDescriptors.row(singleNum-1) += patterns[i].descriptors.row(k);

						#if _DEBUG
						//�����_��ۑ�
						cluster.singleKeypoints.push_back(patterns[i].keypoints.at(k) );
						#endif
					}
				}
			}
			//cluster�̍ő�T�C�Y��ۑ�
			cluster.maxClusterSize = clusterMatches.size();
			//�ۑ�
			clusters.push_back(cluster);
		}
	}


}

void FeatureClustering::featureBudgeting(std::vector<ClusterOfFeature> clusters, Pattern& metaFeature)
{
	//�����ʂ̎�����
	int cols = clusters[0].metaDescriptors.cols;

	//�����p�ϐ�
	std::vector< std::pair<int, int> > imageRankingList;	//�e�摜�̃����L���O(rank, index)
	std::vector<cv::Mat> rankedDescriptors;					//rank�Ɋ�Â��ĕ��ёւ����ecluster�̓����ʂ�ۑ�


	//--------------------step 1 ------------------------------------//

	//�摜�̃����N�t��
	for(int i = 0; i < clusters.size(); i++)
	{
		int rank = clusters[i].rankingList.size();					//�摜�̃����N
		std::pair<int , int> list;

		for(int j = 0; j < clusters[i].rankingList.size(); j++)
		{
			rank += clusters[i].rankingList[j];
		}

		list.first = rank;
		list.second = i;

		//�摜�̃����N��ۑ�
		imageRankingList.push_back(list);
	}
	//�摜�̃����L���O�Ɋ�Â��č~���ɕ��ёւ�
	std::sort(imageRankingList.begin(), imageRankingList.end(),std::greater<std::pair<int, int>>() );

	//-----------------step 2 --------------------------------------------//

	//�������A�ecluster��metaDescriptors��cluster�T�C�Y(�}�b�`���O������)�Ɋ�Â��č~���ɕ��ёւ�
	for(int i = 0; i < clusters.size(); i++)
	{
		std::vector<std::pair<int, int>> index;						//���ёւ��p�����ϐ�pair(rank, �����ʂ�index)
		int num = imageRankingList[i].second;						//�摜�����L���O
																	//clusters[num]�͍ł��摜�����L���O���������cluster
		for(int j = 0; j < clusters[num].rankingList.size(); j++)
		{
			std::pair<int, int> pair;								//pair(�����ʂ�rank, �����ʂ�index)
			pair.first = clusters[num].rankingList[j];
			pair.second = j;

			index.push_back(pair);
		}
		//���ёւ�
		std::sort(index.begin(), index.end(), std::greater<std::pair<int, int>>());

		//�ecluster��metaDescriptors��cluster�T�C�Y(�}�b�`���O������)�Ɋ�Â��č~���ɕ��ёւ�
		cv::Mat descriptors = cv::Mat::zeros(index.size(), cols,  CV_8U);
		for(int k = 0; k < index.size(); k++)
		{
			descriptors.row(k) += clusters[num].metaDescriptors.row(index[k].second);

		}
		//�摜�����L���O���ɃN���X�^�̓����ʂ�ۑ�
		rankedDescriptors.push_back(descriptors);
	}

	//--------------------- step 3 ---------------------------------------// 
	//�摜�����L���O�������摜�́A�ł������}�b�`���O���������ʂ�D�悵�Ċ��蓖�Ă鏈��
	bool isBeFilled = false;
	metaFeature.descriptors = cv::Mat::zeros(m_budget, cols,  CV_8U);		//������
	isBeFilled = createMetaFeature(rankedDescriptors, metaFeature.descriptors);
	
	if(isBeFilled == false)
	{
		addSingleFeatures(clusters, imageRankingList, metaFeature.descriptors);
	}
}

bool FeatureClustering::createMetaFeature(std::vector<cv::Mat> rankedDescriptors, cv::Mat& metaDescriptors)
{
	bool isBeFilled = false;									//�����ʂ̊��蓖�Ă��\�Z�܂ŒB������
	std::vector<int> descSize;									//�edescriptors�̎c������ʐ�
	int descSum = 0 ;												//�����ʂ̐��̍��v

	//������
	for(int i = 0; i < rankedDescriptors.size(); i++)
	{
		descSize.push_back(0);
		descSum +=rankedDescriptors[i].rows;
	}

	int total = 0;												//���蓖�Ă���

	//���^�����ʂ̊��蓖��
	while (isBeFilled==false)
	{
		for(int i = 0; i < rankedDescriptors.size(); i++)
		{
			int max = total + 30;
			for(total; total < max; total++)
			{
				if(descSize[i] < rankedDescriptors[i].rows )
				{
					if(total >= m_budget)
					{
						isBeFilled = true;
						return isBeFilled;
						break;
					}else
					{
						metaDescriptors.row(total) += rankedDescriptors[i].row(descSize[i]);
						descSize[i] += 1;
					}
				}else
				{
					break;
				}
			}

			if(total >= m_budget)
			{
				isBeFilled = true;
				return isBeFilled;
				break;
			}

			int sum = std::accumulate(descSize.begin(),descSize.end(), 0);
			if(sum == descSum)
			{
				metaDescriptors.resize(sum, 0);
				isBeFilled = false;
				return isBeFilled;
				break;
			}
		}
	}

	//���^�����ʂ��\�Z�ɒB���Ȃ������ꍇ
	return isBeFilled;
}

void FeatureClustering::addSingleFeatures(std::vector<ClusterOfFeature> clusters,std::vector<std::pair<int, int>> rankingIndex, cv::Mat& metaDescriptors)
{
	std::vector<int> descSize;									//�edescriptors�̎c������ʐ�
	int startRow = metaDescriptors.rows;
	bool isBeFilled = false;									//�����ʂ̊��蓖�Ă��\�Z�܂ŒB������
	int total = metaDescriptors.rows + 1;	
	int descSum = 0;

	metaDescriptors.resize(m_budget, 0);

	//������
	for(int i = 0; i < clusters.size(); i++)
	{
		descSize.push_back(0);
		descSum +=clusters[i].singleDescriptors.rows;
	}

	while (isBeFilled==false)
	{
		for(int i = 0; i < clusters.size(); i++)
		{
			int num = rankingIndex[i].second;						//�摜�����L���O
		
			int max = total + 30;
			for(total; total < max; total++)
			{
				if(descSize[num] < clusters[num].singleDescriptors.rows )
				{
					if(total >= m_budget)
					{
						isBeFilled = true;
						break;
					}else
					{
						
						metaDescriptors.row(total) += clusters[num].singleDescriptors.row(descSize[num]);
						std::cout << clusters[num].singleDescriptors.row(descSize[num]) << std::endl;
						descSize[num] += 1;
					}
				}else
				{
					break;
				}
			}

			if(total >= m_budget)
			{
				isBeFilled = true;
				break;
			}

			int sum = std::accumulate(descSize.begin(),descSize.end(), 0);
			if(sum == descSum)
			{
				metaDescriptors.resize(sum, 0);
				isBeFilled = true;
				break;
			}
		}
	}
}

void FeatureClustering::showResult(Pattern pattern,ClusterOfFeature cluster)
{

	cv::Mat clusteringResult;
	clusteringResult = pattern.image.clone();

	std::cout << cluster.metaDescriptors.size() << std::endl;
	std::cout << cluster.singleDescriptors.size() << std::endl;

	std::cout << "---------------"<< std::endl;

	for(int i = 0; i < cluster.metaKeypoints.size(); i++)
	{//white
		switch (cluster.rankingList[i])
		{
		case 1://blue
			cv::circle(clusteringResult, cluster.metaKeypoints[i].pt , 1, cv::Scalar(255,0,0),2, CV_FILLED);
			break;
		case 2://green
			cv::circle(clusteringResult, cluster.metaKeypoints[i].pt , 1, cv::Scalar(0,255,0),2, CV_FILLED);
			break;
		case 3://red
			cv::circle(clusteringResult, cluster.metaKeypoints[i].pt , 1, cv::Scalar(0,0,255),2, CV_FILLED);
			break;
		case 4:
			cv::circle(clusteringResult, cluster.metaKeypoints[i].pt , 1, cv::Scalar(255,255,0),2, CV_FILLED);
			break;
		case 5:
			cv::circle(clusteringResult, cluster.metaKeypoints[i].pt , 1, cv::Scalar(250,0,255),2, CV_FILLED);
			break;
		default:
			cv::circle(clusteringResult, cluster.metaKeypoints[i].pt , 1, cv::Scalar(255,255,255),2, CV_FILLED);
			break;

		}
		//�����_���摜�ɕ`��
		std::ostringstream stream;
		stream <<  cluster.rankingList[i];
		std::string rank = stream.str();

		cv::putText(clusteringResult, rank, cluster.metaKeypoints[i].pt, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255,255,255), 1, CV_AA);
	}
	/*
	for(int i = 0; i < cluster.singleKeypoints.size(); i++)
	{//black
		cv::circle(clusteringResult, cluster.singleKeypoints[i].pt , 1, cv::Scalar(0,0,0),2, CV_FILLED);
	}
	*/
	cv::imshow("clusterinResult",clusteringResult);
	/*
	static int count = 0;
	std::stringstream ss;
	ss << count;
	std::string result = "result";
	result +=  ss.str();
	result += ".jpg";
	cv::imwrite(result,clusteringResult);
	count++;
	*/
}

void  FeatureClustering::showMetaFeatures(cv::Mat image,ClusterOfFeature cluster, Pattern meta)
{
	cv::Mat metaResult;
	metaResult = image.clone();
	std::vector<cv::DMatch> matches;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;

	cv::Ptr<cv::FeatureDetector>     detector  = cv::FeatureDetector::create(detectorName);
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::FeatureDetector::create(extractorName);

	cv::Mat gray;

	if (image.channels()  == 3)
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, CV_BGRA2GRAY);
    else if (image.channels() == 1)
        gray = image;

	detector->detect(gray, keypoints);
	extractor->compute(gray,keypoints, descriptors);

	cv::Ptr<cv::DescriptorMatcher>   matcher   = cv::DescriptorMatcher::create(matcherName);
	std::vector<std::vector<cv::DMatch>> knnMatches;
	matcher->knnMatch(descriptors, meta.descriptors, knnMatches, 2);

	for(int j = 0; j < knnMatches.size(); j++)
	{
		if(knnMatches[j].empty() == false)
		{
			const cv::DMatch& bestMatch = knnMatches[j][0];
			const cv::DMatch& betterMatch = knnMatches[j][1];

			float distanceRatio = bestMatch.distance / betterMatch.distance;

			//�����̔䂪1.5�ȉ��̓��������ۑ�
			if(distanceRatio < 0.8f)
			{
				matches.push_back(bestMatch);
			}
		}
	}

	for(int i = 0; i < matches.size(); i++)
	{
		//if(matches[i].distance == 0)
		cv::circle(metaResult, keypoints[matches[i].queryIdx].pt , 1, cv::Scalar(0,0,255),2, CV_FILLED);
	}
	
	cv::imshow("metaResult",metaResult);

	/*
	static int count = 0;
	std::stringstream ss;
	ss << count;
	std::string result = "result";
	result +=  ss.str();
	result += ".jpg";
	cv::imwrite(result,metaResult);
	count++;
	*/
}