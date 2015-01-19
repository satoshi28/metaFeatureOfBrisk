#ifndef MATCHING_
#define MATCHING_

////////////////////////////////////////////////////////////////////
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "../Pattern.h"
#include "../CONSTANT.h"


/**
 * 特徴量同士をマッチングするクラス
 */
class Matching
{
public:
	/**
    * @brief 初期化
	* @note バイナリーコード型の特徴量の場合Flannは使用不可
     */
    Matching();

    /**
    * @brief 入力されたPattern構造体群の特徴量同士でマッチング処理しマッチングしたペアを返す
	* @param[in] patterns 特徴点，特徴量のデータを持つPattern構造体の集合
	* @param[out] matches マッチングペア(画像1→画像2,画像1→画像3...のマッチングペアをそれぞれのvectorに格納し，
	*																ほかの画像のvector配列もを一つにまとめたvector配列) 
    */
	void getMatches(const std::vector<Pattern> patterns, std::vector< std::vector<cv::DMatch> >& matches);

	/**
	* @brief Homographyの取得
	* return すべてのホモグラフィを返す
	*/
	std::vector<std::vector<cv::Mat>> Matching::getHomography();
private:	
	/**
	* @brief matcherに訓練特徴量を学習させる
	* @param[in] trainDescriptors マッチングされるすべての特徴量
	* @param[out] matcher マッチングアルゴリズム
	*/
	void train(const cv::Mat trainDescriptors, cv::Ptr<cv::DescriptorMatcher>& matcher);

	/**
	* @brief 特徴量のマッチング処理
	* @param[in] queryKeypoints マッチングするすべての特徴点
	* @param[in] queryDescriptors マッチングするすべての特徴量
	* @param[in] trainDescriptors マッチングされるすべての特徴量
	* @param[in] trainKeypoints マッチングされるすべての特徴点
	* @param[out] matches マッチングペア
	* @note trainDescriptorsとmatchersのサイズは同じにしなければならない
	*/
	void Matching::match(const std::vector<cv::KeyPoint> queryKeypoints, const cv::Mat queryDescriptors,
		const std::vector<cv::Mat> trainDescriptors, const std::vector<std::vector<cv::KeyPoint>> trainKeypoints, std::vector<cv::DMatch>& matches);

	/**
	* @brief 特徴点を利用し幾何学的整合性チェックを行い外れ値を除去する
	* @param[in] queryKeypoints マッチングする特徴点
	* @param[in] trainKeypoints マッチングされるすべての特徴点
	* @param[out] matches 外れ値を除外されたマッチングペア
	* return Homographyが推定でき外れ値を除外できたらtrueを返す
	* @note Ransacアルゴリズムを使用
	*/
	bool geometricConsistencyCheck(std::vector<cv::KeyPoint> queryKeypoints, 
		std::vector<cv::KeyPoint> trainKeypoints, std::vector<cv::DMatch>& matches, cv::Mat& homography);

	/**
	* @brief 推定したHomography行列が正しいものかを判定する
	* @param[in] H Homography行列
	* return 拘束条件を満たしていたらtrueを返す
	* @note 
	*/
	bool niceHomography(const cv::Mat H);
private:
	int dataSetSize;			//画像セットの数
	int imgNumberOfAdjstment;	//マッチングペアの画像IDを修正するための変数
	std::vector<std::vector<cv::Mat>> AllHomographyes;	//GeometoricConsistencyCheckで得られたすべてのhomography
};


#endif