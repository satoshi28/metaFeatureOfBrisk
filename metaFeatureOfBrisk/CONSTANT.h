#ifndef _CONSTANT
#define _CONSTANT

const std::string detectorName = "SURF";		//�����_���o�A���S���Y��
const std::string extractorName = "BRISK";	//�����ʒ��o�A���S���Y��
const std::string matcherName = "BruteForce-Hamming";		//�}�b�`���O�A���S���Y��

const int budget = 200;					//���^�����ʂ̃T�C�Y
const bool enableMultipleRatioTest = false;	//�����䔻��@��p���邩
const float minRatio = 0.8f;			// To avoid NaN's when best match has zero distance we will use inversed ratio. 
#endif