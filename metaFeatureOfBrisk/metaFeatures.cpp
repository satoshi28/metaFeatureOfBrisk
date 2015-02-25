// metaFeatures.cpp : メイン プロジェクト ファイルです。

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
	/* 変数宣言 */
	String^ folder = "C:\\Users\\satoshi\\Documents\\Image\\ZuBuD\\database\\";
	std::vector< std::vector<std::string> > sortFile;
	std::vector<std::vector<Pattern>> matches;

	/* 処理部分 */
	groupPath.getPath(folder, sortFile);

	std::vector<Pattern> metaFeatures( sortFile.size() );
	
	std::cout << "start " << budget << " fileName=" << fileName << std::endl;

	for(int i = 0; i < sortFile.size(); i++)
	{
		std::vector<cv::Mat> images;
		
		//画像の読み込み
		readImages(sortFile[i], images);
		//画像のgps座標の読み込み
		gps.GetGps( sortFile[i][0].c_str(), metaFeatures[i].gps );

		//メタ特徴量の作成
		clustering.clusterFeatures(images, metaFeatures[i]);

		//画像の登録
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
		image = cv::imread(a,1);			//画像の読み込み
		if (image.empty())
		{
			std::cout << "Input image cannot be read" << std::endl;
			return false;
		}
		//画像を追加
		images.push_back(image);

	}
}

