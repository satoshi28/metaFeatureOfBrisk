#ifndef FEATURE_DESCRIPTOR_CLUSTERING
#define FEATURE_DESCRIPTOR_CLUSTERING

////////////////////////////////////////////////////////////////////

#include <functional>
#include <numeric>

#include "ExtractFeatures.hpp"
#include "Matching.h"
#include "ClusterOfFeature.h"


/**
 * Store the image data and computed descriptors of target pattern
 */
class FeatureClustering
{
public:
	FeatureClustering(int m_budget = budget, bool m_enableMultipleRatioTest = enableMultipleRatioTest);

	/*
	* �O���[�v�����ꂽ�摜�Q���烁�^�����ʂ𒊏o����
	*/
	void clusterFeatures(std::vector<cv::Mat> images, Pattern& metaFeatures);

private:

	/* �����̉摜�ԂŌ������������ʂ����meta�����ʂɂ܂Ƃ߂� */
	void clusterDescriptors(std::vector<std::vector<cv::DMatch>> matches, std::vector<ClusterOfFeature>& clusters);

	/* �}�b�`���O���������ʂ��܂Ƃ߂�cluster���烁�^�����ʂ��쐬���鏈�� */
	void featureBudgeting(std::vector<ClusterOfFeature> clusters, Pattern& metaFeature);

	bool createMetaFeature(std::vector<cv::Mat> rankedDescriptors, cv::Mat& metaDescriptors);

	void addSingleFeatures(std::vector<ClusterOfFeature> clusters,std::vector<std::pair<int, int>> rankingIndex, cv::Mat& metaDescriptors);
#if _DEBUG
	/* �����ʂ�`�� */
	void showResult(Pattern pattern, ClusterOfFeature cluster);

	void showMetaFeatures(cv::Mat image,ClusterOfFeature cluster, Pattern meta);
#endif

private:
	int m_budget;
	bool m_enableMultipleRatioTest;
	std::vector<Pattern> patterns;

};


#endif