#ifndef EXTRACT_FEATURES
#define EXTRACT_FEATURES

////////////////////////////////////////////////////////////////////

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "../Pattern.h"
#include "../CONSTANT.h"

/**
 * Store the kepoints and descriptors of image
 */
class ExtractFeatures
{
public:
	/**
     * Initialize a pattern detector with specified feature detector, descriptor extraction and matching algorithm
     */
    ExtractFeatures
        (
        cv::Ptr<cv::FeatureDetector>     detector  = cv::FeatureDetector::create(detectorName), 
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::FeatureDetector::create(extractorName)
        );

	~ExtractFeatures();

	/**
    * 画像を受け取り,特徴量を抽出する 
    * 抽出した特徴点,特徴量はPattern構造体として保存する
    */
	bool getFeatures(std::vector<cv::Mat>& images, std::vector<Pattern>& patterns);

private:
	/**
    * 入力された画像からグレイスケール画像を取得する
	* Supported input images types - 1 channel (no conversion is done), 3 channels (assuming BGR) and 4 channels (assuming BGRA).
    */
	void getGray(const cv::Mat& image, cv::Mat& grayImg);

	/**
    * 入力された画像から特徴点,特徴量を抽出する
    */
	bool extractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

private:
	cv::Ptr<cv::FeatureDetector>     m_detector;
    cv::Ptr<cv::DescriptorExtractor> m_extractor;
    cv::Ptr<cv::DescriptorMatcher>   m_matcher;

};




#endif