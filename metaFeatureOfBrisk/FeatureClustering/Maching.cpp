#include "stdafx.h"
#include "Matching.h"

Matching::Matching()
{
}


void Matching::getMatches(const std::vector<Pattern> patterns, std::vector< std::vector<cv::DMatch> >& clusterMatches)
{
	dataSetSize = patterns.size();	//画像セットの数
	imgNumberOfAdjstment = 0;		//matchesのマッチングした画像IDを修正するための変数

	//
	//すべての画像をマッチングする
	for(int i = 0; i < dataSetSize; i++)
	{
		//処理用一時変数宣言
		std::vector<cv::KeyPoint> queryKeypoints;				//マッチングする特徴点
		cv::Mat queryDescriptors;								//マッチングする特徴量
		std::vector<std::vector<cv::KeyPoint>> trainKeypoints;	//マッチングされる特徴点
		std::vector<cv::Mat> trainDescriptors;					//マッチングされる特徴量
		std::vector<cv::DMatch> matches;						//マッチングペア

		//macherを用意
		std::vector< cv::Ptr<cv::DescriptorMatcher> > matchers( dataSetSize -1 );
		for(int k = 0; k < dataSetSize -1; k++)
		{
			matchers[k] = cv::DescriptorMatcher::create(matcherName);
		}

		//データのコピー
		queryKeypoints = patterns[i].keypoints;
		queryDescriptors = patterns[i].descriptors;

		//マッチングされる特徴量をすべてコピー
		for(int j = 0; j < dataSetSize; j++)
		{
			//クエリ画像以外の訓練特徴点，特徴量に格納
			if(i != j)
			{
				trainDescriptors.push_back( patterns[j].descriptors );
				trainKeypoints.push_back(patterns[j].keypoints);
			}
		}

		//訓練データをmatchersに追加
		match(queryKeypoints, queryDescriptors, trainDescriptors, trainKeypoints, matches);

		//マッチングペアを画像ごとに分けられているvector配列に格納
		clusterMatches.push_back(matches);

		//matchesの画像番号修正用
		imgNumberOfAdjstment++;
	}
}


void Matching::train(const cv::Mat trainDescriptors, cv::Ptr<cv::DescriptorMatcher>& matcher)
{
	std::vector<cv::Mat> descriptors(1);

	// API of cv::DescriptorMatcher is somewhat tricky
	// First we clear old train data:
	matcher->clear();

	// Then we add vector of descriptors (each descriptors matrix describe one image). 
	// This allows us to perform search across multiple images:

	descriptors[0]= trainDescriptors.clone();
	matcher->add(descriptors);

	// After adding train data perform actual train:
	matcher->train();
}



void Matching::match(const std::vector<cv::KeyPoint> queryKeypoints,const cv::Mat queryDescriptors,
	const std::vector<cv::Mat> trainDescriptors,const std::vector<std::vector<cv::KeyPoint>> trainKeypoints, std::vector<cv::DMatch>& matches)
{
	matches.clear();		//初期化
	int imgNumber = 0;		//マッチングされている画像のID

	//最近傍点の探索
	for(int i = 0; i < dataSetSize -1 ; i++)
	{
		cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(matcherName);
		std::vector< std::vector<cv::DMatch>>  knnMatches;

		//k近傍を検索する.
		train(trainDescriptors[i], matcher);
		matcher->knnMatch(queryDescriptors, knnMatches, 2);

		//ratio testを通過したマッチングペア
		std::vector<cv::DMatch> correctMatches;

		//ratio test
		for(int j = 0; j < knnMatches.size(); j++)
		{
			if(knnMatches[j].empty() == false)
			{
				 cv::DMatch& bestMatch = knnMatches[j][0];
				 cv::DMatch& betterMatch = knnMatches[j][1];

				float distanceRatio = bestMatch.distance / betterMatch.distance;

				//距離の比が0.8以下の特徴だけ保存
				if(distanceRatio < minRatio)
				{
					if(i == imgNumberOfAdjstment)
					{
						imgNumber = imgNumberOfAdjstment + 1;
					}
					bestMatch.imgIdx = imgNumber;
					correctMatches.push_back(bestMatch);
				}
			}
		}
		//幾何学的整合性チェック
		bool passFlag = geometricConsistencyCheck(queryKeypoints, trainKeypoints[i], correctMatches);

		//幾何学的整合性チェックに通過したもののみ登録する
		if(passFlag == true){
			//要素の移し替え
			for(int k = 0; k < correctMatches.size(); k++)
			{
				matches.push_back(correctMatches[k]);
			}

		}

		//初期化
		knnMatches.clear();
		correctMatches.clear();
		imgNumber++;
	}
}

bool Matching::geometricConsistencyCheck(std::vector<cv::KeyPoint> queryKeypoints, std::vector<cv::KeyPoint> trainKeypoints, std::vector<cv::DMatch>& matches)
{
	if(matches.size() < 8)
	{
		matches.clear();
		return false;
	}
	std::vector<cv::Point2f>  queryPoints, trainPoints; 
	for(int i = 0; i < matches.size(); i++)
	{
		queryPoints.push_back(queryKeypoints[matches[i].queryIdx].pt);
		trainPoints.push_back(trainKeypoints[matches[i].trainIdx].pt);
	}

	//幾何学的整合性チェック
	std::vector<unsigned char> inliersMask(queryPoints.size() );

	//幾何学的整合性チェックによって当たり値を抽出
	cv::Mat homography = cv::findHomography( queryPoints, trainPoints, CV_FM_RANSAC, 10, inliersMask);
	/*
	//Homography行列が正しいか検証
	bool isGoodHomography = niceHomography(homography);
	if(isGoodHomography == false)
		return false;
		*/
	std::vector<cv::DMatch> inliers;
	for(size_t i =0 ; i < inliersMask.size(); i++)
	{
		if(inliersMask[i])
			inliers.push_back(matches[i]);
	}

	matches.swap(inliers);
	return true;
}

bool Matching::niceHomography(const cv::Mat H)
{
	const double det = H.at<double>(0,0) * H.at<double>(1,1) - H.at<double>(1,0) * H.at<double>(0,1);
	if (det < 0)
	  return false;
	
	const double N1 = sqrt( H.at<double>(0,0) * H.at<double>(0,0) + H.at<double>(1,0) * H.at<double>(1,0) );
	if (N1 > 4 || N1 < 0.1)
	  return false;
	
	const double N2 = sqrt( H.at<double>(0,1) * H.at<double>(0,1) + H.at<double>(1,1) * H.at<double>(1,1) );
	if (N2 > 4 || N2 < 0.1)
	  return false;

	const double N3 = sqrt( H.at<double>(2,0) * H.at<double>(2,0) + H.at<double>(2,1) * H.at<double>(2,1) );
	if (N3 > 0.002)
	  return false;
	
	return true;
}