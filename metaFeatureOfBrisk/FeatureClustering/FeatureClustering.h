#ifndef FEATURE_DESCRIPTOR_CLUSTERING
#define FEATURE_DESCRIPTOR_CLUSTERING

////////////////////////////////////////////////////////////////////

#include <functional>
#include <numeric>

#include "ExtractFeatures.hpp"
#include "Matching.h"
#include "ClusterOfFeature.h"


/**
 * �����̉摜���烁�^�����ʂ��쐬����N���X
 */
class FeatureClustering
{
public:
	FeatureClustering(int m_budget = budget);

	/**
	* @brief �O���[�v�����ꂽ�摜�Q���烁�^�����ʂ��쐬����
	* @param[in] images			�摜�Q
	* @param[out] metaFeatures	���^������
	*/
	void clusterFeatures(std::vector<cv::Mat> images, Pattern& metaFeatures);

private:
	/**
	* @brief �����̉摜�ԂŌ������������ʂ���̃��^�����ʂɂ܂Ƃ߂�
	* @param[in] matches		�}�b�`���O����
	* @param[out] clusters		�}�b�`���O���ʂ����̓����ɂ܂Ƃ߂��N���X�^
	*/
	void clusterDescriptors(std::vector<std::vector<cv::DMatch>> matches, std::vector<ClusterOfFeature>& clusters);

	/**
	* @brief �摜�������N�t������
	* @param[in] clusters		�}�b�`���O���ʂ����̓����ɂ܂Ƃ߂��N���X�^
	* @param[out] imageRankingList �摜�̃����N�t��������킵�����X�g(first==rank, second==index)
	*/
	void rankImages(std::vector<ClusterOfFeature> clusters, std::vector< std::pair<int, int> >& imageRankingList);

	/**
	* @brief �}�b�`���O���������ʂ��܂Ƃ߂��N���X�^���烁�^�����ʂ��쐬���鏈��
	* @param[in] clusters		�N���X�^
	* @param[out] metaFeature	���^������
	*/
	void featureBudgeting(std::vector<ClusterOfFeature> clusters, std::vector< std::pair<int, int> > imageRankingList, Pattern& metaFeature);

	/**
	* @brief �����L���O�������摜���珇�ɍł������}�b�`���O���������ʂ�D�悵�Ċ��蓖�Ă鏈��
	* @param[in] rankedDescriptors	�N���X�^�̃����N���������ɕ��ёւ���������
	* @param[in] rankedKeypoints	�N���X�^�̃����N���������ɕ��ёւ��������_
	* @param[in] imgNumbers			�摜ID���X�g
	* @param[out] metaFeature		���^������
	* return �������ۂ�
	*/
	bool createMetaFeature(std::vector<cv::Mat> rankedDescriptors,std::vector< std::vector<cv::KeyPoint>> rankedKeypoints,std::vector<int> imgNumbers, Pattern& metaFeature);

	/**
	* @brief �P�̂̓����ʂ��烁�^�����ʂ𖄂߂鏈��
	* @param[in] clusters		�}�b�`���O���ʂ����̓����ɂ܂Ƃ߂��N���X�^
	* @param[in] rankingIndex	�摜�����L���O
	* @param[out] metaFeature	���^������
	*/
	void addSingleFeatures(std::vector<ClusterOfFeature> clusters,std::vector<std::pair<int, int>> rankingIndex,Pattern& metaFeature);

	/**
	* @brief �P�̂̓����ʂ��烁�^�����ʂ𖄂߂鏈��
	* @param[in] homography		�z���O���t�B
	* @param[in] src_keypoints	�ϊ����̓����_
	* @param[out] dst_keypoints	�ϊ���̓����_
	*/
	bool adjustKeypoints(const cv::Mat_<double>& H, std::vector<cv::KeyPoint> src_keypoints, std::vector<cv::KeyPoint>& dst_keypoints);

	/* �����ʂ�`�� */
	void showResult(Pattern pattern, ClusterOfFeature cluster);

	void showMetaFeatures(std::vector<Pattern> patterns,Pattern metaFeature);

	cv::Mat getModeDescriptors(cv::Mat trainDescriptors);
private:
	int m_budget;					//���^�����ʂ̗\�Z
	std::vector<Pattern> patterns;	//�e�摜�̓���
	std::vector< std::vector<cv::Mat> > AllHomography;

};


#endif