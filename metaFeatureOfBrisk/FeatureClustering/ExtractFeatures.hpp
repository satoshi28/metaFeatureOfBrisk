#ifndef EXTRACT_FEATURES
#define EXTRACT_FEATURES

////////////////////////////////////////////////////////////////////

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "../Pattern.h"
#include "../CONSTANT.h"

/**
 * �����_�C�����ʂ��擾����N���X
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
	* @brief ���͂��ꂽ�摜�f�[�^��������_�C�����ʂ𒊏o��Pattern�\���̂ɕۑ�����
	* @param[in] images �摜�f�[�^�Q
	* return �����������ۂ�
	*/
	bool getFeatures(std::vector<cv::Mat>& images, std::vector<Pattern>& patterns);

private:
	
	/**
	* @brief �O���[�X�P�[���摜�ɂ���
	* @param[in] image
	* return �O���[�X�P�[����̉摜
	* @note Supported input images types - 1 channel (no conversion is done), 3 channels (assuming BGR) and 4 channels (assuming BGRA).
	*/
	void getGray(const cv::Mat& image, cv::Mat& grayImg);

	/**
	* @brief �摜�f�[�^��������_�C�����ʂ𒊏o����
	* @param[in] image
	* @param[out] keypoints
	* @param[out] descriptors
	* return �������ۂ�
	* @note �摜��1chanel�݂̂ɑΉ��@�����_�C�����ʂ���������Ȃ������ꍇ��false��Ԃ�
	*/
	bool extractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

private:
	cv::SurfFeatureDetector  m_detector;			//�����_���o�A���S���Y��
    cv::Ptr<cv::DescriptorExtractor> m_extractor;	//�����ʒ��o�A���S���Y��

};




#endif