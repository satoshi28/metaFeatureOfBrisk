// metaFeatures.cpp : ���C�� �v���W�F�N�g �t�@�C���ł��B

#include "stdafx.h"
#include "GroupPath.h"
#include "FeatureClustering\FeatureClustering.h"

bool readImages(std::vector<std::string> filenames, std::vector<cv::Mat>& images);

int main(array<System::String ^> ^args)
{
	/* class */
	GroupPath groupPath;
	FeatureClustering clustering;
	/* �ϐ��錾 */
	String^ folder = "C:\\Users\\satoshi\\Documents\\Image\\db";
	std::vector< std::vector<std::string> > sortFile;
	std::vector<std::vector<Pattern>> matches;
	std::vector<Pattern> metaFeatures;

	/* �������� */
	groupPath.getPath(folder, sortFile);

	for(int i = 0; i < sortFile.size(); i++)
	{
		std::vector<cv::Mat> images;
		Pattern metaFeature;

		//�摜�̓ǂݍ���
		readImages(sortFile[i], images);
		//���^�����ʂ̍쐬
		clustering.clusterFeatures(images, metaFeature);
		//std::cout << metaFeature.descriptors << std::endl;
		std::cout << "-----------------" << std::endl;
	}
	
	cv::waitKey(0);
	return 0;
}

bool readImages(std::vector<std::string> filenames, std::vector<cv::Mat>& images)
{
	for(int i = 0; i < filenames.size(); i++)
	{

		cv::Mat image;
		std::string a = filenames[i];
		
		std::cout << a << std::endl;

		image = cv::imread(a,1);			//�摜�̓ǂݍ���
		if (image.empty())
		{
			std::cout << "Input image cannot be read" << std::endl;
			return false;
		}
		//�摜��ǉ�
		images.push_back(image);

	}
}

