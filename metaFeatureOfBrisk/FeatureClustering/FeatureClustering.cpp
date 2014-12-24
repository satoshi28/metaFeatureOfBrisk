
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
		cv::Mat falseImage = cv::imread("C:\\Users\\satoshi\\Documents\\Image\\falseImage.JPG",1);
		images.push_back(falseImage);
	}

	//�����p
	// �����ʂ�Pattern�ɕۑ�
	extract.getFeatures(images,patterns);
	// ���ׂẲ摜���m���}�b�`���O����
	matching.getMatches(patterns, clusterMatches);

	homographyes = matching.getHomography();

	std::vector<ClusterOfFeature> clusters;
	//�}�b�`���O���ʂ���N���X�^�����O�����ʂ��쐬����
	clusterDescriptors(clusterMatches, clusters);
	//
	std::cout << "OK" << std::endl;
	
	/*
	for(int i = 0; i < patterns.size(); i++)
	{
		showResult(patterns[i], clusters[i]);
		cv::waitKey(0);
	}
	*/
	//�N���X�^�����O�����ʂ��烁�^�����ʂ��쐬����
	featureBudgeting(clusters, metaFeatures);
	
	//showMetaFeatures(patterns, metaFeatures);
	//�㏈��
	patterns.clear();

}



void FeatureClustering::clusterDescriptors( std::vector<std::vector<cv::DMatch>> clusterMatches, std::vector<ClusterOfFeature>& clusters)
{

	int rank = 0;			//������Ԃ��痈�������_�̐�
	int id;
	std::vector<int> matchList;


	if(m_enableMultipleRatioTest == true)
	{

		for(int i = 0; i < clusterMatches.size(); i++)
		{
			int cols = patterns[i].descriptors.cols;

			//�����p�ϐ�
			ClusterOfFeature cluster;
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

					//�ۑ����� 
					//�����ʂ̍s�ɒǉ�
					cluster.metaDescriptors.push_back(patterns[i].descriptors.row(queryIdx) );

					//�����_��ۑ�
					cluster.metaKeypoints.push_back(patterns[i].keypoints.at(queryIdx) );

					//�����ʂ̃����N��ۑ�
					cluster.rankingList.push_back(rank);

					rank = 1;
				}

			}
			//�P�̂̓����ʂ�singleDescriptors�ɕۑ�
			for(int k = 0; k < patterns[i].descriptors.rows; k++)
			{
				for(int m = 0; m < matchList.size(); m++)
				{
					if(matchList[m] == k)	//�}�b�`���O���X�g�ɍڂ��Ă���Δ�����
					{
						break;
					}
					if(m == matchList.size()-1)
					{
						//�����ʂ̍s�ɒǉ�
						cluster.singleDescriptors.push_back( patterns[i].descriptors.row(k) );

						//�����_��ۑ�
						cluster.singleKeypoints.push_back(patterns[i].keypoints.at(k) );
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
			matchList.clear();

			for(int j = 0; j < clusterMatches[i].size(); j++)
			{
				id = clusterMatches[i][j].queryIdx;

				//query�ԍ�����v��������ʂ�T��
				for(int k = 0; k < clusterMatches[i].size(); k++)
				{//�K��rank>1(���g���Q�Ƃ��Ă��邩��)
					if(id == clusterMatches[i][k].queryIdx)
					{
						rank += 1;
					}
				}

				bool isBeFind = false;
				int queryIdx = clusterMatches[i][j].queryIdx;		//query�̃C���f�b�N�X

				for(int k=0; k < matchList.size(); k++)
				{
					if(matchList[k] == queryIdx)
						isBeFind = true;
				}
				//���łɌ�������query��������ǉ����Ȃ�
				if(isBeFind == false)
				{
					matchList.push_back(queryIdx);						//�}�b�`���O���X�g��query�̔ԍ���ۑ�
					
					//�����ʂ̍s�ɒǉ�
					cluster.metaDescriptors.push_back(patterns[i].descriptors.row(queryIdx) );
		
					//�����_��ۑ�
					cluster.metaKeypoints.push_back(patterns[i].keypoints.at(queryIdx) );

					//�����ʂ̃����N��ۑ�
					cluster.rankingList.push_back(rank);
				}
				rank = 0;
			}

			//std::sort(matchList.begin(), matchList.end());


			//�P�̂̓����ʂ�singleDescriptors�ɕۑ�
			for(int k = 0; k < patterns[i].descriptors.rows; k++)
			{
				for(int m = 0; m < matchList.size(); m++)
				{
					if(matchList[m] == k)
					{
						break;
					}
					if(m == matchList.size()-1)	//�}�b�`���X�g�̍Ō�܂Ō�����Ȃ������ꍇ�A�V���O�������ʂɒǉ�
					{
						//�����ʂ̍s�ɒǉ�
						cluster.singleDescriptors.push_back(patterns[i].descriptors.row(k) );

						//�����_��ۑ�
						cluster.singleKeypoints.push_back(patterns[i].keypoints.at(k) );
					}
				}

				//�}�b�`���O�������̂��Ȃ������ꍇ
				if(matchList.size() == 0)
				{
					//�����ʂ̍s�ɒǉ�
					cluster.singleDescriptors.push_back(patterns[i].descriptors.row(k) );

					//�����_��ۑ�
					cluster.singleKeypoints.push_back(patterns[i].keypoints.at(k) );
				}

			}
			//�V���O�������ʂ̓����_���ɕ��בւ�			
			//random_shuffle(cluster.singleKeypoints.begin(), cluster.singleKeypoints.end());

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
	std::vector< std::vector<cv::KeyPoint>> rankedKeypoints;					//rank�Ɋ�Â��ĕ��ёւ����ecluster�̓����_��ۑ�
	std::vector<int> imgNumbers;

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

	/* 
	* �ō������N�̃N���X�^(Imax)�̉摜�ɁC�e�N���X�^�̓����_�𓊉e
	* �����ʒu�ɗ��������_��cluster�T�C�Y�����Ƃ̓����_�̃N���X�^�T�C�Y�ɉ��Z
	* inliner�Ƃ��Ď�菜���ꂽ�����_��Imax�ɐV�K�ɍ쐬�C�N���X�^�T�C�Y�͂��̂܂�
	* step2�͐�ɂ���Ƃ�
	* �N���X�^�T�C�Y���������̂��烁�^�����ʂɊ��蓖�Ă�
	*/
	clusterToMetaFeature(clusters, homographyes, imageRankingList[0].second, metaFeature);

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
		cv::Mat descriptors;
		std::vector<cv::KeyPoint> keypoints;

		for(int k = 0; k < index.size(); k++)
		{
			descriptors.push_back( clusters[num].metaDescriptors.row(index[k].second) );
			keypoints.push_back( clusters[num].metaKeypoints[index[k].second] );
			
		}
		//�摜�����L���O���ɃN���X�^�̓����ʂ�ۑ�
		rankedDescriptors.push_back(descriptors);
		rankedKeypoints.push_back(keypoints);
		imgNumbers.push_back(num);
	}

	//--------------------- step 3 ---------------------------------------// 
	//�摜�����L���O�������摜�́A�ł������}�b�`���O���������ʂ�D�悵�Ċ��蓖�Ă鏈��
	bool isBeFilled = false;
	isBeFilled = createMetaFeature(rankedDescriptors,rankedKeypoints,imgNumbers, metaFeature);
	
	if(isBeFilled == false)
	{
		addSingleFeatures(clusters, imageRankingList, metaFeature);
	}
}

bool FeatureClustering::createMetaFeature(std::vector<cv::Mat> rankedDescriptors,std::vector< std::vector<cv::KeyPoint>> rankedKeypoints,std::vector<int> imgNumbers, Pattern& metaFeature)
{
	bool isBeFilled = false;									//�����ʂ̊��蓖�Ă��\�Z�܂ŒB������
	std::vector<int> descSize;									//�edescriptors�̎c������ʐ�
	int descSum = 0 ;												//�����ʂ̐��̍��v
	std::pair<bool, int>			paramOfKeypoint;
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
						paramOfKeypoint.first = true;
						paramOfKeypoint.second = imgNumbers[i];

						metaFeature.paramOfKeypoints.push_back(paramOfKeypoint);						//keypoint�̃p�����[�^��ۑ�

						metaFeature.descriptors.push_back( rankedDescriptors[i].row(descSize[i]) );
						metaFeature.keypoints.push_back( rankedKeypoints[i][ descSize[i] ]);

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
				metaFeature.descriptors.resize(sum, 0);
				isBeFilled = false;
				return isBeFilled;
				break;
			}
		}
	}

	//���^�����ʂ��\�Z�ɒB���Ȃ������ꍇ
	return isBeFilled;
}

void FeatureClustering::addSingleFeatures(std::vector<ClusterOfFeature> clusters,std::vector<std::pair<int, int>> rankingIndex, Pattern& metaFeature)
{
	std::vector<int> descSize;									//�edescriptors�̎c������ʐ�
	int startRow = metaFeature.descriptors.rows;
	bool isBeFilled = false;									//�����ʂ̊��蓖�Ă��\�Z�܂ŒB������
	int total = metaFeature.descriptors.rows;	
	int descSum = 0;
	std::pair<bool, int>			paramOfKeypoint;

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
						paramOfKeypoint.first = false;
						paramOfKeypoint.second = num;

						metaFeature.paramOfKeypoints.push_back(paramOfKeypoint);						//keypoint�̃p�����[�^��ۑ�
						
						metaFeature.descriptors.push_back( clusters[num].singleDescriptors.row(descSize[num]) );
						metaFeature.keypoints.push_back(clusters[num].singleKeypoints[ descSize[num] ] );
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
				metaFeature.descriptors.resize(sum, 0);
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
	
	for(int i = 0; i < cluster.singleKeypoints.size(); i++)
	{//black
		cv::circle(clusteringResult, cluster.singleKeypoints[i].pt , 1, cv::Scalar(0,0,0),2, CV_FILLED);
	}
	
	cv::imshow("clusterinResult",clusteringResult);
	/*
	static int count = 0;
	std::stringstream ss;
	ss << count;
	std::string result = "matchingResult";
	result +=  ss.str();
	result += ".png";
	cv::imwrite(result,clusteringResult);
	count++;
	*/
	
}

void  FeatureClustering::showMetaFeatures(std::vector<Pattern> patterns,Pattern metaFeature)
{
	for(int i =0; i< patterns.size();i++)
	{
		cv::Mat metaResult;
		metaResult = patterns[i].image.clone();
		int metaFeaturesSize =0;


		for(int j = 0; j < metaFeature.paramOfKeypoints.size(); j++)
		{
			if(metaFeature.paramOfKeypoints[j].second == i)
			{
				if(metaFeature.paramOfKeypoints[j].first == true)//�����̉摜�Ō���������
					cv::circle(metaResult,metaFeature.keypoints[j].pt , 4, cv::Scalar(0,0,255),-1, CV_AA);
				else													//�V���O���̓���
					cv::circle(metaResult,metaFeature.keypoints[j].pt , 4, cv::Scalar(0,0,0),-1, CV_AA);

				metaFeaturesSize++;

			}
		}
		
		cv::imshow("metaResult",metaResult);
		std::cout << metaFeaturesSize << std::endl;
		cv::waitKey(0);
		/*
		static int count = 0;
		std::stringstream ss;
		ss << count;
		std::string result = "metaResult";
		result +=  ss.str();
		result += ".png";
		cv::imwrite(result,metaResult);
		count++;
		*/
	}
}

void FeatureClustering::clusterToMetaFeature(std::vector<ClusterOfFeature> clusters,std::vector<std::vector<cv::Mat>> allHomographyes, int basisImgNum, Pattern& metaFeature)
{
	//��̃N���X�^
	ClusterOfFeature basisCluster;
	basisCluster = clusters[basisImgNum];

	
/*
	//���̑��̃N���X�^
	std::vector<ClusterOfFeature> trainClusters;
	for(int i = 0; i < clusters.size(); i++)
	{
		if(i == basisImgNum) continue;

		trainClusters.push_back(clusters[i]);
	}
	//�ϊ���̃N���X�^
	std::vector<ClusterOfFeature> transformedClusters;
	std::vector<cv::Mat> homographyes;

	//macher��p��
	std::vector< cv::Ptr<cv::DescriptorMatcher> > matchers( trainClusters.size());
	for(int k = 0; k <trainClusters.size(); k++)
	{
		matchers[k] = cv::DescriptorMatcher::create(matcherName);

		std::vector<cv::Mat> descriptors(1);
		descriptors[0] = trainClusters[k].metaDescriptors.clone();
		matchers[k]->add(descriptors);

		matchers[k]->train();
	}

	//�ŋߖT�_�̒T��
	for(int i = 0; i < trainClusters.size() ; i++)
	{
		//knn�}�b�`���O
		std::vector< std::vector<cv::DMatch>>  knnMatches;

		// query��matcher�ɕۑ�����Ă�������ʂ�knn�\���̂�p���čŋߖT�_����������.
		matchers[i]->knnMatch(basisCluster.metaDescriptors, knnMatches, 2);
	
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

				//�����̔䂪0.8�ȉ��̓��������ۑ�
				if(distanceRatio < minRatio)
				{
					correctMatches.push_back(bestMatch);
				}
			}
		}

		cv::Mat homography;
		//�􉽊w�I�������`�F�b�N
		if(correctMatches.size() < 8)
		{
			correctMatches.clear();
			//�z���O���t�B�s�񂪐���ł��Ȃ������ꍇ�͒P�ʍs����i�[
			cv::Mat eye = cv::Mat::zeros(3, 3, CV_64FC1); // �P�ʍs��𐶐�
			homographyes.push_back(eye);
			continue;
		}
		
		std::vector<cv::Point2f>  queryPoints, trainPoints; 
		for(int j = 0; j < correctMatches.size(); j++)
		{
			queryPoints.push_back(trainClusters[i].metaKeypoints[correctMatches[j].trainIdx].pt);
			trainPoints.push_back(basisCluster.metaKeypoints[correctMatches[j].queryIdx].pt);
		}

		//�􉽊w�I�������`�F�b�N
		std::vector<unsigned char> inliersMask(queryPoints.size() );

		//�􉽊w�I�������`�F�b�N�ɂ���ē�����l�𒊏o
		homography = cv::findHomography( queryPoints, trainPoints, CV_FM_RANSAC, 3, inliersMask);

		std::vector<cv::DMatch> inliers;
		for(size_t j =0 ; j < inliersMask.size(); j++)
		{
			if(inliersMask[j])
				inliers.push_back(correctMatches[j]);
		}

		correctMatches.swap(inliers);
		homographyes.push_back(homography);



		cv::Mat result;
		if(i < basisImgNum)	//
		{
			cv::drawMatches(patterns[basisImgNum].image, basisCluster.metaKeypoints,patterns[i].image , clusters[i].metaKeypoints,correctMatches, result);
			
		}
		else if(i > basisImgNum)
		{
			cv::drawMatches(patterns[basisImgNum].image, basisCluster.metaKeypoints,patterns[i-1].image , clusters[i-1].metaKeypoints,correctMatches, result);
		}				//������
		knnMatches.clear();
		correctMatches.clear();

	}

	for(int i = 0; i < trainClusters.size() ; i++)
	{
		//�[���s�񂩔���
		cv::Mat tmp1,tmp2;
		cv::reduce(homographyes[i], tmp1, 1, CV_REDUCE_SUM);
		cv::reduce(tmp1, tmp2, 0, CV_REDUCE_SUM);
		if(tmp2.at<double>(0,0) == 0.0)
			continue;

		// �ϊ�
		double mm[3][3];
		for(int j = 0; j < 3; j++)
		{
			for(int k = 0; k < 3; k++)
			{
				mm[j][k] = homographyes[i].at<double>(j,k);
			}
		}

		ClusterOfFeature transformedCluster;
		transformedCluster = trainClusters[i];

		for(int j = 0; j < transformedCluster.metaKeypoints.size(); j++)
		{
			double x = transformedCluster.metaKeypoints[j].pt.x;
			double y = transformedCluster.metaKeypoints[j].pt.y;

			double t_x = (mm[0][0]*x + mm[0][1]*y + mm[0][2])/(mm[2][0]*x + mm[2][1] + mm[2][2]);
			double t_y = (mm[1][0]*x + mm[1][1]*y + mm[1][2])/(mm[2][0]*x + mm[2][1] + mm[2][2]);

			transformedCluster.metaKeypoints[j].pt.x = t_x;
			transformedCluster.metaKeypoints[j].pt.y = t_y;

		}

		for(int j = 0; j < transformedCluster.singleKeypoints.size(); j++)
		{
			double x = transformedCluster.singleKeypoints[j].pt.x;
			double y = transformedCluster.singleKeypoints[j].pt.y;

			double t_x = (mm[0][0]*x + mm[0][1]*y + mm[0][2])/(mm[2][0]*x + mm[2][1] + mm[2][2]);
			double t_y = (mm[1][0]*x + mm[1][1]*y + mm[1][2])/(mm[2][0]*x + mm[2][1] + mm[2][2]);

			transformedCluster.singleKeypoints[j].pt.x = t_x;
			transformedCluster.singleKeypoints[j].pt.y = t_y;

		}

		cv::Mat resultImg = patterns[basisImgNum].image.clone();

		for(int j = 0; j < basisCluster.metaKeypoints.size(); j++)
		{//white
			cv::circle(resultImg, basisCluster.metaKeypoints[j].pt , 2, cv::Scalar(0,0,255),2, CV_FILLED);
		}
		
		for(int j = 0; j < transformedCluster.metaKeypoints.size(); j++)
		{//white
			cv::circle(resultImg, transformedCluster.metaKeypoints[j].pt , 1, cv::Scalar(255,0,0),2, CV_FILLED);
		}

		cv::imshow("result", resultImg);
		cv::waitKey(0);

	}
*/
	
		
	for(int i = 0; i < clusters.size(); i++)
	{
		//��̏ꍇ�͔�΂�
		if(i == basisImgNum) continue;

		//�}�b�`���O�̍ې��肵���z���O���t�B�̎擾
		cv::Mat homography;
		if(i < basisImgNum)	//
		{
			homography = homographyes[i][basisImgNum - 1];
		}
		else if(i > basisImgNum)
		{
			homography = homographyes[i][basisImgNum];
		}

		//�[���s�񂩔���
		cv::Mat tmp1,tmp2;
		cv::reduce(homography, tmp1, 1, CV_REDUCE_SUM);
		cv::reduce(tmp1, tmp2, 0, CV_REDUCE_SUM);
		if(tmp2.at<double>(0,0) == 0.0)
			continue;



		std::cout << homography << std::endl;

		double mm[3][3];
		for(int j = 0; j < 3; j++)
		{
			for(int k = 0; k < 3; k++)
			{
				mm[j][k] = homography.at<double>(j,k);
				std::cout << mm[j][k] << std::endl;
			}
		}

		ClusterOfFeature transformedCluster;
		transformedCluster = clusters[i];
		//���W�ϊ�����
		for(int j = 0; j < transformedCluster.metaKeypoints.size(); j++)
		{
			double x = transformedCluster.metaKeypoints[j].pt.x;
			double y = transformedCluster.metaKeypoints[j].pt.y;

			double t_x = (mm[0][0]*x + mm[0][1]*y + mm[0][2])/(mm[2][0]*x + mm[2][1] + mm[2][2]);
			double t_y = (mm[1][0]*x + mm[1][1]*y + mm[1][2])/(mm[2][0]*x + mm[2][1] + mm[2][2]);

			transformedCluster.metaKeypoints[j].pt.x = t_x;
			transformedCluster.metaKeypoints[j].pt.y = t_y;

		}
		/*
		for(int j = 0; j < transformedCluster.singleKeypoints.size(); j++)
		{
			double x = transformedCluster.singleKeypoints[j].pt.x;
			double y = transformedCluster.singleKeypoints[j].pt.y;

			double t_x = (mm[0][0]*x + mm[0][1]*y + mm[0][2])/(mm[2][0]*x + mm[2][1] + mm[2][2]);
			double t_y = (mm[1][0]*x + mm[1][1]*y + mm[1][2])/(mm[2][0]*x + mm[2][1] + mm[2][2]);

			transformedCluster.singleKeypoints[j].pt.x = t_x;
			transformedCluster.singleKeypoints[j].pt.y = t_y;

		}*/

		//
		ClusterOfFeature refinedCluster;
		//���[�N���b�h�����v�Z
		for(int j = 0; j < transformedCluster.metaKeypoints.size(); j++)
		{
			double minDist = 10;
			int minDistanceKeypointNum = -1;
			double x1 = transformedCluster.metaKeypoints[j].pt.x;
			double y1 = transformedCluster.metaKeypoints[j].pt.y;

			for(int k = 0; k < basisCluster.metaKeypoints.size(); k++)
			{
				double x2 = basisCluster.metaKeypoints[k].pt.x;
				double y2 = basisCluster.metaKeypoints[k].pt.y;

				double dist = sqrt( (x1 -x2)*(x1 -x2) + (y1 - y2)*(y1 - y2) );

				if(dist < minDist)
				{
					minDist = dist;
					minDistanceKeypointNum = k;
				}
			}

			//臒l�ȓ��ōŏ��̃��[�N���b�h���������_����{�ƂȂ�N���X�^�ɒu��
			if(minDist < 6.0)
			{
				basisCluster.rankingList[minDistanceKeypointNum] += transformedCluster.rankingList[j];

				//descriptor�𕽋ω�
				//
			}else //����������ǉ�
			{/*
				basisCluster.metaKeypoints.push_back(transformedCluster.metaKeypoints[j]);
				basisCluster.metaDescriptors.push_back(transformedCluster.metaDescriptors.row(j) );
				basisCluster.rankingList.push_back(transformedCluster.rankingList[j]);
				*/
			}
		}

		cv::Mat resultImg = patterns[basisImgNum].image.clone();

		for(int i = 0; i < basisCluster.metaKeypoints.size(); i++)
		{//white
			cv::circle(resultImg, basisCluster.metaKeypoints[i].pt , 2, cv::Scalar(0,0,255),2, CV_FILLED);
		}
		
		for(int i = 0; i < transformedCluster.metaKeypoints.size(); i++)
		{//white
			cv::circle(resultImg, transformedCluster.metaKeypoints[i].pt , 1, cv::Scalar(255,0,0),2, CV_FILLED);
		}
		
		//cv::imshow("result", resultImg);
		//cv::waitKey(0);

/*		static int count = 0;
		std::stringstream ss;
		ss << count;
		std::string result = "matchingPoint";
		result +=  ss.str();
		result += ".jpg";
		cv::imwrite(result,resultImg);
		count++;
	*/	

	}

	std::vector<std::pair<int, int>> index;						//���ёւ��p�����ϐ�pair(rank, �����ʂ�index)
															//clusters[num]�͍ł��摜�����L���O���������cluster
	for(int i = 0; i < basisCluster.rankingList.size(); i++)
	{
		std::pair<int, int> pair;								//pair(�����ʂ�rank, �����ʂ�index)
		pair.first = basisCluster.rankingList[i];
		pair.second = i;

		index.push_back(pair);
	}
	//���ёւ�
	std::sort(index.begin(), index.end(), std::greater<std::pair<int, int>>());

	//�ecluster��metaDescriptors��cluster�T�C�Y(�}�b�`���O������)�Ɋ�Â��č~���ɕ��ёւ�

	for(int j =0; j < 200; j++)
	{
		if(j >= (int)basisCluster.metaKeypoints.size() )
		{
			break;
		}

		int num = index[j].second;
		metaFeature.descriptors.push_back(basisCluster.metaDescriptors.row(num) );
		metaFeature.keypoints.push_back(basisCluster.metaKeypoints[num] );


	}
	
	cv::Mat image = patterns[basisImgNum].image.clone();
	for(int i = 0; i < metaFeature.keypoints.size(); i++)
	{//white
		cv::circle(image, metaFeature.keypoints[i].pt , 1, cv::Scalar(255,0,0),2, CV_FILLED);
	}
		
	cv::imshow("result", image);
	cv::waitKey(0);

}