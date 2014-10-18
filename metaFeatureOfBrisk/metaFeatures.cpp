// metaFeatures.cpp : メイン プロジェクト ファイルです。

#include "stdafx.h"
#include "GroupPath.h"
#include "FeatureClustering\FeatureClustering.h"
#include "ConnectingDB\ConnectingDB.h"
#include "ConnectingDB\exifGps.h"

bool readImages(std::vector<std::string> filenames, std::vector<cv::Mat>& images);

int main(array<System::String ^> ^args)
{

	/* class */
	GroupPath groupPath;
	FeatureClustering clustering;
	ConnectingDB db;
	exifGps gps;
	/* 変数宣言 */
	String^ folder = "C:\\Users\\satoshi\\Documents\\Image\\ZuBuD\\database";
	std::vector< std::vector<std::string> > sortFile;
	std::vector<std::vector<Pattern>> matches;

	/* 処理部分 */
	groupPath.getPath(folder, sortFile);

	std::vector<Pattern> metaFeatures( sortFile.size() );

	for(int i = 0; i < sortFile.size(); i++)
	{
		std::vector<cv::Mat> images;
		
		//画像の読み込み
		readImages(sortFile[i], images);
		//画像のgps座標の読み込み
		gps.GetGps( sortFile[i][0].c_str(), metaFeatures[i].gps );

		//メタ特徴量の作成
		clustering.clusterFeatures(images, metaFeatures[i]);

		//std::cout << metaFeature.descriptors << std::endl;
		std::cout << "-----------------" << std::endl;
	}

	db.updateDB(metaFeatures);
	
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

