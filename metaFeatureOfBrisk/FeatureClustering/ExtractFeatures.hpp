#ifndef EXTRACT_FEATURES
#define EXTRACT_FEATURES

////////////////////////////////////////////////////////////////////

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "../Pattern.h"
#include "../CONSTANT.h"

/**
 * 特徴点，特徴量を取得するクラス
 */
class ExtractFeatures
{
public:
	/**
     * Initialize a pattern detector with specified feature detector and descriptor extraction
     */
    ExtractFeatures
        (
        cv::SurfFeatureDetector detector = cv::SurfFeatureDetector(400), 
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::FeatureDetector::create(extractorName)
        );

	~ExtractFeatures();

	/**
	* @brief 入力された画像データから特徴点，特徴量を抽出しPattern構造体に保存する
	* @param[in] images 画像データ群
	* return 成功したか否か
	*/
	bool getFeatures(std::vector<cv::Mat>& images, std::vector<Pattern>& patterns);

private:
	
	/**
	* @brief グレースケール画像にする
	* @param[in] image
	* return グレースケール後の画像
	* @note Supported input images types - 1 channel (no conversion is done), 3 channels (assuming BGR) and 4 channels (assuming BGRA).
	*/
	void getGray(const cv::Mat& image, cv::Mat& grayImg);

	/**
	* @brief 画像データから特徴点，特徴量を抽出する
	* @param[in] image
	* @param[out] keypoints
	* @param[out] descriptors
	* return 成功か否か
	* @note 画像は1chanelのみに対応　特徴点，特徴量が一つも得られなかった場合にfalseを返す
	*/
	bool extractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

private:
	cv::SurfFeatureDetector  m_detector;			//特徴点検出アルゴリズム
    cv::Ptr<cv::DescriptorExtractor> m_extractor;	//特徴量抽出アルゴリズム

};




#endif