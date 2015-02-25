#include "stdafx.h"
#include "FeatureClustering.h"

FeatureClustering::FeatureClustering(int budget)
{
	m_budget = budget;
}

void FeatureClustering::clusterFeatures(std::vector<cv::Mat> images, Pattern& metaFeatures)
{
	ExtractFeatures extract;
	Matching matching;

	std::vector< std::vector<cv::DMatch> > clusterMatches;
	

	//�����p
	// �����ʂ�Pattern�ɕۑ�
	extract.getFeatures(images,patterns);
	// ���ׂẲ摜���m���}�b�`���O����
	matching.getMatches(patterns, clusterMatches);

	std::vector<ClusterOfFeature> clusters;
	//�}�b�`���O���ʂ���N���X�^�����O�����ʂ��쐬����
	clusterDescriptors(clusterMatches, clusters);
	
	
	for(int i = 0; i < patterns.size(); i++)
	{
		showResult(patterns[i], clusters[i]);
		cv::waitKey(0);
	}
	

	//�摜�������N�t��
	std::vector< std::pair<int, int> > imageRankingList;
	rankImages(clusters, imageRankingList);


	//�N���X�^�����O�����ʂ��烁�^�����ʂ��쐬����
	featureBudgeting(clusters, imageRankingList, metaFeatures);
	
	//showMetaFeatures(patterns, metaFeatures);

	if(metaFeatures.descriptors.rows != m_budget)
		std::cout << metaFeatures.descriptors.rows << std::endl;

	//�㏈��
	patterns.clear();

}


//�}�b�`���O���X�g����}�b�`���O���������ʓ��m�Ɋ�Â��Ĉ�̓����ʂɂ܂Ƃ߂鏈��
void FeatureClustering::clusterDescriptors( std::vector<std::vector<cv::DMatch>> clusterMatches,
											std::vector<ClusterOfFeature>& clusters)
{
	for(int i = 0; i < clusterMatches.size(); i++)
	{
		//�����p�ϐ�
		ClusterOfFeature cluster;		//�}�b�`���O���ʂ����ɂ܂Ƃ߂�������
		std::vector<int> matchList;		//�o�^�ς݃��X�g
		matchList.clear();

		//�}�b�`���O���������ʂ��ЂƂɂ܂Ƃ߃��^�����ʂƂ��ĕۑ����鏈��
		for(int j = 0; j < clusterMatches[i].size(); j++)
		{
			int queryId = clusterMatches[i][j].queryIdx;	//�}�b�`���O�����������ʂ�ID�i�N�G���ԍ��j
			int rank = 0;									//������Ԃ��痈�������_�̐�
			cv::Mat descriptors;							//����queryID�����}�b�`���O�����ʂ��i�[
			bool isFounded = false;

			//�d�����ēo�^���Ȃ��悤��queryId���o�^�ς݃��X�g�ɍڂ��Ă��邩�m�F�B����ꍇ�͔�΂�
			for (int k = 0; k < matchList.size(); k++)
			{
				if (matchList[k] == queryId){
					isFounded = true;
					break;
				}
			}
			if(isFounded == true)
				continue;
			
			for(int k = 0; k < clusterMatches[i].size(); k++)	//query�ԍ�����v��������ʂ�T��
			{
				if (queryId == clusterMatches[i][k].queryIdx)
				{
					rank += 1;									//�K��rank>1(���g���Q�Ƃ��Ă��邩��)
				}
			}
			
			matchList.push_back(queryId);						//�}�b�`���O���X�g��query�̔ԍ���ۑ�
			
			//�����ʂ̍s�ɒǉ�
			cluster.metaDescriptors.push_back( patterns[i].descriptors.row(queryId) );
			cluster.metaKeypoints.push_back(patterns[i].keypoints.at(queryId) );
			cluster.rankingList.push_back(rank);

		}
		std::sort(matchList.begin(), matchList.end());

		//�P�̂̓����ʂ�singleDescriptors�ɕۑ�
		for(int k = 0; k < patterns[i].descriptors.rows; k++)
		{
			//�}�b�`���O���X�g��ID���o�^����Ă��邩����
			std::vector<int>::iterator cIter = std::find(matchList.begin(), matchList.end(), k);
			if (cIter != matchList.end())
				continue;

			//�}�b�`���O���X�g�ɂȂ���Γ����ʂ̍s�ɒǉ�
			cluster.singleDescriptors.push_back(patterns[i].descriptors.row(k) );
			cluster.singleKeypoints.push_back(patterns[i].keypoints.at(k) );

		}
		//cluster�̍ő�T�C�Y��ۑ�
		cluster.maxClusterSize = clusterMatches.size();

		//�ۑ�
		clusters.push_back(cluster);
	}



}

void FeatureClustering::featureBudgeting(std::vector<ClusterOfFeature> clusters,
										 std::vector< std::pair<int, int> > imageRankingList,
										 Pattern& metaFeature)
{
	//�����ʂ̎�����
	int cols = clusters[0].metaDescriptors.cols;

	//�����p�ϐ�
	std::vector<cv::Mat> rankedDescriptors;						//rank�Ɋ�Â��ĕ��ёւ����ecluster�̓����ʂ�ۑ�
	std::vector< std::vector<cv::KeyPoint>> rankedKeypoints;	//rank�Ɋ�Â��ĕ��ёւ����ecluster�̓����_��ۑ�
	std::vector<int> imgNumbers;

	//-----------------step 1 --------------------------------------------//

	//�������A�ecluster��metaDescriptors��cluster�T�C�Y(�}�b�`���O������)�Ɋ�Â��č~���ɕ��ёւ�
	for(int i = 0; i < clusters.size(); i++)
	{
		std::vector<std::pair<int, int>> index;						//���^�����ʂ̃����L���O���X�g(rank, �����ʂ�index)
		int num = imageRankingList[i].second;						//�摜�̃����L���O
																	//clusters[num]�͍ł��摜�����L���O���������cluster
		//�e�����ʂ��}�b�`���O�������Ɋ�Â��ĕ]��
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

	//--------------------- step 2 ---------------------------------------// 
	//�摜�����L���O�������摜�́A�ł������}�b�`���O���������ʂ�D�悵�Ċ��蓖�Ă鏈��
	bool isBeFilled = false;
	isBeFilled = createMetaFeature(rankedDescriptors,rankedKeypoints,imgNumbers, metaFeature);
	
	//���^�����ʂŗ\�Z�����܂�Ȃ�������P�̂̓����ʂ����蓖�Ă�
	if(isBeFilled == false)
		addSingleFeatures(clusters, imageRankingList, metaFeature);

}

void FeatureClustering::rankImages(std::vector<ClusterOfFeature> clusters,
								   std::vector< std::pair<int, int> >& imageRankingList)
{
	imageRankingList.clear();

	//�摜�����L���O�̍쐬(clusters�̃T�C�Y�͉摜�̖���)
	for (int i = 0; i < clusters.size(); i++)
	{
		int rank = clusters[i].rankingList.size();												//�摜�̃����L���O
		std::pair<int, int> list;									//�����L���O�̃��X�g

		//�e�摜�ɑ΂��ă��^�����ʂ��\����������ʂ̐��Ɋ�Â��ē��[
		for (int j = 0; j < clusters[i].rankingList.size(); j++)
			rank += clusters[i].rankingList[j];

		//�摜�̃����N��ۑ�
		list.first = rank;
		list.second = i;
		imageRankingList.push_back(list);
	}

	//�摜�̃����L���O�Ɋ�Â��ă��X�g���~���ɕ��ёւ�
	std::sort(imageRankingList.begin(), imageRankingList.end(), std::greater<std::pair<int, int>>());

}

bool FeatureClustering::createMetaFeature(std::vector<cv::Mat> rankedDescriptors,
										  std::vector< std::vector<cv::KeyPoint>> rankedKeypoints,
										  std::vector<int> imgNumbers,
										  Pattern& metaFeature)
{
	bool isBeFilled = false;									//�����ʂ̊��蓖�Ă��\�Z�܂ŒB������
	std::vector<int> descSize;									//�edescriptors�̎c������ʐ�
	int descSum = 0 ;											//�����ʂ̐��̍��v
	std::pair<bool, int> paramOfKeypoint;						//���^�����ʂ��P�̂̓����ʂ��������ϐ�(�f�o�b�N�p)
	
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
			int max = total + 30;		//30�_�����܂邲�Ƃɉ摜��ς���
			for(total; total < max; total++)
			{
				if(descSize[i] < rankedDescriptors[i].rows )
				{
					if(total >= m_budget)	//�\�Z�����܂������̏���
					{
						isBeFilled = true;
						return isBeFilled;
						break;
					}else
					{
						//keypoint�̃p�����[�^��ۑ�
						paramOfKeypoint.first = true;
						paramOfKeypoint.second = imgNumbers[i];
						metaFeature.paramOfKeypoints.push_back(paramOfKeypoint);

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
			}

			int sum = std::accumulate(descSize.begin(),descSize.end(), 0);
			if(sum == descSum)
			{
				metaFeature.descriptors.resize(sum, 0);
				isBeFilled = false;
				return isBeFilled;
			}
		}
	}

	//���^�����ʂ��\�Z�ɒB���Ȃ������ꍇ
	return isBeFilled;
}

void FeatureClustering::addSingleFeatures(std::vector<ClusterOfFeature> clusters,
										  std::vector<std::pair<int, int>> rankingIndex,
										  Pattern& metaFeature)
{
	std::vector<int> descSize;									//�edescriptors�̎c������ʐ�
	bool isBeFilled = false;									//�����ʂ̊��蓖�Ă��\�Z�܂ŒB������
	int total = metaFeature.descriptors.rows;					//���^�����ʐ��̍��v
	int descSum = 0;											//�����ʂ̐��̍��v
	std::pair<bool, int> paramOfKeypoint;						//���^�����ʂ��P�̂̓����ʂ��������ϐ�(�f�o�b�N�p)

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
			int max = total + 30;									//30�_�����܂邲�Ƃɉ摜��ς���
			for(total; total < max; total++)
			{
				if(descSize[num] < clusters[num].singleDescriptors.rows )
				{
					if(total >= m_budget)							//�\�Z�����܂������̏���
					{
						isBeFilled = true;
						break;
					}else
					{
						//keypoint�̃p�����[�^��ۑ�
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

	for(int i = 0; i < cluster.singleKeypoints.size(); i++)
	{//black
		cv::circle(clusteringResult, cluster.singleKeypoints[i].pt , 1, cv::Scalar(0,0,0),2, CV_FILLED);
	}

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
	

	
	cv::imshow("clusterinResult",clusteringResult);
	
	static int count = 0;
	std::stringstream ss;
	ss << count;
	std::string result = "matchingResult";
	result +=  ss.str();
	result += ".png";
	cv::imwrite(result,clusteringResult);
	count++;
	
	
}

void  FeatureClustering::showMetaFeatures(std::vector<Pattern> patterns, Pattern metaFeature)
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
					cv::circle(metaResult,metaFeature.keypoints[j].pt , 3, cv::Scalar(0,153,255),-1, CV_AA);
				else													//�V���O���̓���
					cv::circle(metaResult,metaFeature.keypoints[j].pt , 3, cv::Scalar(0,0,0),-1, CV_AA);

				metaFeaturesSize++;

			}
		}
		
		cv::imshow("metaResult",metaResult);
		std::cout << metaFeaturesSize << std::endl;
		cv::waitKey(0);
		
		static int count = 0;
		std::stringstream ss;
		ss << count;
		std::string result = "metaResult";
		result +=  ss.str();
		result += ".png";
		cv::imwrite(result,metaResult);
		count++;
		
	}
}
