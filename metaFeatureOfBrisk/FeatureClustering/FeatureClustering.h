#ifndef FEATURE_DESCRIPTOR_CLUSTERING
#define FEATURE_DESCRIPTOR_CLUSTERING

////////////////////////////////////////////////////////////////////

#include <functional>
#include <numeric>
#include <random>

#include "ExtractFeatures.hpp"
#include "Matching.h"
#include "ClusterOfFeature.h"


/**
 * 複数の画像からメタ特徴量を作成するクラス
 */
class FeatureClustering
{
public:
	FeatureClustering(int m_budget);

	/**
	* @brief グループ化された画像群からメタ特徴量を作成する
	* @param[in] images			画像群
	* @param[out] metaFeatures	メタ特徴量
	*/
	void clusterFeatures(std::vector<cv::Mat> images, Pattern& metaFeatures);

private:
	/**
	* @brief 複数の画像間で見つかった特徴量を一つのメタ特徴量にまとめる
	* @param[in] matches		マッチング結果
	* @param[out] clusters		マッチング結果から一つの特徴にまとめたクラスタ
	*/
	void clusterDescriptors(std::vector<std::vector<cv::DMatch>> matches, std::vector<ClusterOfFeature>& clusters);

	/**
	* @brief 画像をランク付けする
	* @param[in] clusters		マッチング結果から一つの特徴にまとめたクラスタ
	* @param[out] imageRankingList 画像のランク付けをあらわしたリスト(first==rank, second==index)
	*/
	void rankImages(std::vector<ClusterOfFeature> clusters, std::vector< std::pair<int, int> >& imageRankingList);

	/**
	* @brief マッチングした特徴量をまとめたクラスタからメタ特徴量を作成する処理
	* @param[in] clusters		クラスタ
	* @param[out] metaFeature	メタ特徴量
	*/
	void featureBudgeting(std::vector<ClusterOfFeature> clusters, std::vector< std::pair<int, int> > imageRankingList, Pattern& metaFeature);

	/**
	* @brief ランキングが高い画像から順に最も多くマッチングした特徴量を優先して割り当てる処理
	* @param[in] rankedDescriptors	クラスタのランクが高い順に並び替えた特徴量
	* @param[in] rankedKeypoints	クラスタのランクが高い順に並び替えた特徴点
	* @param[in] imgNumbers			画像IDリスト
	* @param[out] metaFeature		メタ特徴量
	* return 成功か否か
	*/
	bool createMetaFeature(std::vector<cv::Mat> rankedDescriptors,std::vector< std::vector<cv::KeyPoint>> rankedKeypoints,std::vector<int> imgNumbers, Pattern& metaFeature);

	/**
	* @brief 単体の特徴量からメタ特徴量を埋める処理
	* @param[in] clusters		マッチング結果から一つの特徴にまとめたクラスタ
	* @param[in] rankingIndex	画像ランキング
	* @param[out] metaFeature	メタ特徴量
	*/
	void addSingleFeatures(std::vector<ClusterOfFeature> clusters,std::vector<std::pair<int, int>> rankingIndex,Pattern& metaFeature);

	/* 特徴量を描画 */
	void showResult(Pattern pattern, ClusterOfFeature cluster);

	void showMetaFeatures(std::vector<Pattern> patterns,Pattern metaFeature);

private:
	int m_budget;					//メタ特徴量の予算
	std::vector<Pattern> patterns;	//各画像の特徴

};


#endif