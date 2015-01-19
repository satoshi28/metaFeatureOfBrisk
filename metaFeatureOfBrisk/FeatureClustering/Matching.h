#ifndef MATCHING_
#define MATCHING_

////////////////////////////////////////////////////////////////////
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "../Pattern.h"
#include "../CONSTANT.h"


/**
 * �����ʓ��m���}�b�`���O����N���X
 */
class Matching
{
public:
	/**
    * @brief ������
	* @note �o�C�i���[�R�[�h�^�̓����ʂ̏ꍇFlann�͎g�p�s��
     */
    Matching();

    /**
    * @brief ���͂��ꂽPattern�\���̌Q�̓����ʓ��m�Ń}�b�`���O�������}�b�`���O�����y�A��Ԃ�
	* @param[in] patterns �����_�C�����ʂ̃f�[�^������Pattern�\���̂̏W��
	* @param[out] matches �}�b�`���O�y�A(�摜1���摜2,�摜1���摜3...�̃}�b�`���O�y�A�����ꂼ���vector�Ɋi�[���C
	*																�ق��̉摜��vector�z�������ɂ܂Ƃ߂�vector�z��) 
    */
	void getMatches(const std::vector<Pattern> patterns, std::vector< std::vector<cv::DMatch> >& matches);

	/**
	* @brief Homography�̎擾
	* return ���ׂẴz���O���t�B��Ԃ�
	*/
	std::vector<std::vector<cv::Mat>> Matching::getHomography();
private:	
	/**
	* @brief matcher�ɌP�������ʂ��w�K������
	* @param[in] trainDescriptors �}�b�`���O����邷�ׂĂ̓�����
	* @param[out] matcher �}�b�`���O�A���S���Y��
	*/
	void train(const cv::Mat trainDescriptors, cv::Ptr<cv::DescriptorMatcher>& matcher);

	/**
	* @brief �����ʂ̃}�b�`���O����
	* @param[in] queryKeypoints �}�b�`���O���邷�ׂĂ̓����_
	* @param[in] queryDescriptors �}�b�`���O���邷�ׂĂ̓�����
	* @param[in] trainDescriptors �}�b�`���O����邷�ׂĂ̓�����
	* @param[in] trainKeypoints �}�b�`���O����邷�ׂĂ̓����_
	* @param[out] matches �}�b�`���O�y�A
	* @note trainDescriptors��matchers�̃T�C�Y�͓����ɂ��Ȃ���΂Ȃ�Ȃ�
	*/
	void Matching::match(const std::vector<cv::KeyPoint> queryKeypoints, const cv::Mat queryDescriptors,
		const std::vector<cv::Mat> trainDescriptors, const std::vector<std::vector<cv::KeyPoint>> trainKeypoints, std::vector<cv::DMatch>& matches);

	/**
	* @brief �����_�𗘗p���􉽊w�I�������`�F�b�N���s���O��l����������
	* @param[in] queryKeypoints �}�b�`���O��������_
	* @param[in] trainKeypoints �}�b�`���O����邷�ׂĂ̓����_
	* @param[out] matches �O��l�����O���ꂽ�}�b�`���O�y�A
	* return Homography������ł��O��l�����O�ł�����true��Ԃ�
	* @note Ransac�A���S���Y�����g�p
	*/
	bool geometricConsistencyCheck(std::vector<cv::KeyPoint> queryKeypoints, 
		std::vector<cv::KeyPoint> trainKeypoints, std::vector<cv::DMatch>& matches, cv::Mat& homography);

	/**
	* @brief ���肵��Homography�s�񂪐��������̂��𔻒肷��
	* @param[in] H Homography�s��
	* return �S�������𖞂����Ă�����true��Ԃ�
	* @note 
	*/
	bool niceHomography(const cv::Mat H);
private:
	int dataSetSize;			//�摜�Z�b�g�̐�
	int imgNumberOfAdjstment;	//�}�b�`���O�y�A�̉摜ID���C�����邽�߂̕ϐ�
	std::vector<std::vector<cv::Mat>> AllHomographyes;	//GeometoricConsistencyCheck�œ���ꂽ���ׂĂ�homography
};


#endif