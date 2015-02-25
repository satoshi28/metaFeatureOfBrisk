// metaFeatures.cpp : ���C�� �v���W�F�N�g �t�@�C���ł��B

#include "stdafx.h"
#include "GroupPath.h"
#include "FeatureClustering\FeatureClustering.h"
#include "ConnectingDB\ConnectingDB.h"
#include "ConnectingDB\exifGps.h"

bool readImages(std::vector<std::string> filenames, std::vector<cv::Mat>& images);

int main(int argc, char* argv[])
{
	if(argc < 2)
	{
		std::cout << "please input argumet" << std::cout;
		return 1;
	}
	int budget = atoi(argv[1]);
	std::string fileName = argv[2];

	/* class */
	GroupPath groupPath;
	FeatureClustering clustering(budget);
	ConnectingDB db;
	exifGps gps;
	/* �ϐ��錾 */
	String^ folder = "C:\\Users\\satoshi\\Documents\\Image\\ZuBuD\\database\\";
	std::vector< std::vector<std::string> > sortFile;
	std::vector<std::vector<Pattern>> matches;

	/* �������� */
	groupPath.getPath(folder, sortFile);

	std::vector<Pattern> metaFeatures( sortFile.size() );
	
	std::cout << "start " << budget << " fileName=" << fileName << std::endl;

	for(int i = 0; i < sortFile.size(); i++)
	{
		std::vector<cv::Mat> images;
		
		//�摜�̓ǂݍ���
		readImages(sortFile[i], images);
		//�摜��gps���W�̓ǂݍ���
		gps.GetGps( sortFile[i][0].c_str(), metaFeatures[i].gps );

		//���^�����ʂ̍쐬
		clustering.clusterFeatures(images, metaFeatures[i]);

		//�摜�̓o�^
		cv::Mat img;
		cv::resize(images[0], img,cv::Size(320,240));
		metaFeatures[i].image = img;

		//std::cout << metaFeature.descriptors << std::endl;
		//std::cout << "-----------------" << std::endl;
	}
	
	db.updateDB(metaFeatures, fileName);
	std::cout << "end" << std::endl;

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

