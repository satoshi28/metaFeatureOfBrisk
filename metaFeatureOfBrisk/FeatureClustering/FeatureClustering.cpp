
#include "stdafx.h"
#include "FeatureClustering.h"

FeatureClustering::FeatureClustering(int buget, bool enableMultipleRatioTest)
	: m_budget(buget)
	, m_enableMultipleRatioTest(enableMultipleRatioTest)
{

}

void FeatureClustering::clusterFeatures(std::vector<cv::Mat> images, Pattern& metaFeatures)
{
	ExtractFeatures extract;
	Matching matching(m_enableMultipleRatioTest);

	std::vector< std::vector<cv::DMatch> > clusterMatches;

	if(m_enableMultipleRatioTest == true)
	{
		cv::Mat falseImage = cv::imread("C:\\Users\\satoshi\\Documents\\Image\\lenna.png",1);
		images.push_back(falseImage);
	}

	//処理用
	// 特徴量をPatternに保存
	extract.getFeatures(images,patterns);
	// すべての画像同士をマッチングする
	matching.getMatches(patterns, clusterMatches);
	
	std::vector<ClusterOfFeature> clusters;
	//マッチング結果からクラスタリング特徴量を作成する
	clusterDescriptors(clusterMatches, clusters);

	std::cout << "OK" << std::endl;
	for(int i = 0; i < patterns.size(); i++)
	{
		showResult(patterns[i], clusters[i]);

		cv::waitKey(0);
	}
	//クラスタリング特徴量からメタ特徴量を作成する
	featureBudgeting(clusters, metaFeatures);

	for(int i = 0; i < patterns.size(); i++)
	{
		showMetaFeatures(patterns[i].image,clusters[i], metaFeatures);

		cv::waitKey(0);
	}
	//後処理
	patterns.clear();

}



void FeatureClustering::clusterDescriptors( std::vector<std::vector<cv::DMatch>> clusterMatches, std::vector<ClusterOfFeature>& clusters)
{

	int rank = 1;			//同じ空間から来た特徴点の数
	int metaNum = 0;
	int singleNum = 0;
	int id;
	std::vector<int> matchList;


	if(m_enableMultipleRatioTest == true)
	{

		for(int i = 0; i < clusterMatches.size(); i++)
		{
			int cols = patterns[i].descriptors.cols;

			//処理用変数
			ClusterOfFeature cluster;
			metaNum = 0;
			cluster.metaDescriptors = cv::Mat::zeros(1, cols,  CV_8U);
			cluster.singleDescriptors = cv::Mat::zeros(1, cols,  CV_8U);
			matchList.clear();

			//画像間でマッチングした画像をmetaDescriptorsに保存
			for(int j = 1; j < clusterMatches[i].size(); j++)
			{
				id = clusterMatches[i][j -1].queryIdx;

				if(id == clusterMatches[i][j].queryIdx)
				{
					rank += 1;
				}
				else
				{
					int queryIdx = clusterMatches[i][j-1].queryIdx;		//queryのインデックス
					matchList.push_back(queryIdx);						//マッチングリストにqueryの番号を保存

					metaNum += 1;
					cluster.metaDescriptors.resize(metaNum, 0);
					//保存処理 
					//特徴量の行に追加
					cluster.metaDescriptors.row(metaNum-1) += patterns[i].descriptors.row(queryIdx);


					#if _DEBUG
					//特徴点を保存
					cluster.metaKeypoints.push_back(patterns[i].keypoints.at(queryIdx) );
					#endif

					//特徴量のランクを保存
					cluster.rankingList.push_back(rank);

					rank = 1;
				}

			}
			singleNum = 0;
			//単体の特徴量をsingleDescriptorsに保存
			for(int k = 0; k < patterns[i].descriptors.rows; k++)
			{
				for(int m = 0; m < matchList.size(); m++)
				{
					if(matchList[m] == k)
					{
						break;
					}
					if(m == matchList.size()-1)
					{
						singleNum += 1;
						cluster.singleDescriptors.resize(singleNum, 0);
						//特徴量の行に追加
						cluster.singleDescriptors.row(singleNum-1) += patterns[i].descriptors.row(k);

						#if _DEBUG
						//特徴点を保存
						cluster.singleKeypoints.push_back(patterns[i].keypoints.at(k) );
						#endif
					}
				}
			}
			
			//clusterの最大サイズを保存
			cluster.maxClusterSize = clusterMatches.size();
			//保存
			clusters.push_back(cluster);

			
		}
	//ratio testによるマッチング判定
	}else
	{
		for(int i = 0; i < clusterMatches.size(); i++)
		{
			int cols = patterns[i].descriptors.cols;
			rank = 0;
			//処理用変数
			ClusterOfFeature cluster;
			metaNum = 0;
			cluster.metaDescriptors = cv::Mat::zeros(1, cols,  CV_8U);
			cluster.singleDescriptors = cv::Mat::zeros(1, cols,  CV_8U);
			matchList.clear();

			for(int j = 1; j < clusterMatches[i].size(); j++)
			{
				id = clusterMatches[i][j - 1].queryIdx;

				//query番号が一致する特徴量を探す
				for(int k = 0; k < clusterMatches[i].size(); k++)
				{
					if(id == clusterMatches[i][k].queryIdx)
						rank += 1;
				}

				int queryIdx = clusterMatches[i][j].queryIdx;		//queryのインデックス
				matchList.push_back(queryIdx);						//マッチングリストにqueryの番号を保存
				metaNum += 1;

				cluster.metaDescriptors.resize(metaNum, 0);
				//保存処理 
				//特徴量の行に追加
				cluster.metaDescriptors.row(metaNum-1) += patterns[i].descriptors.row(queryIdx);
		
				#if _DEBUG
				//特徴点を保存
				cluster.metaKeypoints.push_back(patterns[i].keypoints.at(queryIdx) );
				#endif

				//特徴量のランクを保存
				cluster.rankingList.push_back(rank);

				rank = 0;
			}
			//初期化
			singleNum = 0;
			//単体の特徴量をsingleDescriptorsに保存
			for(int k = 0; k < patterns[i].descriptors.rows; k++)
			{
				for(int m = 0; m < matchList.size(); m++)
				{
					if(matchList[m] == k)
					{
						break;
					}
					if(m == matchList.size()-1)
					{
						singleNum += 1;
						cluster.singleDescriptors.resize(singleNum, 0);
						//特徴量の行に追加
						cluster.singleDescriptors.row(singleNum-1) += patterns[i].descriptors.row(k);

						#if _DEBUG
						//特徴点を保存
						cluster.singleKeypoints.push_back(patterns[i].keypoints.at(k) );
						#endif
					}
				}
			}
			//clusterの最大サイズを保存
			cluster.maxClusterSize = clusterMatches.size();
			//保存
			clusters.push_back(cluster);
		}
	}


}

void FeatureClustering::featureBudgeting(std::vector<ClusterOfFeature> clusters, Pattern& metaFeature)
{
	//特徴量の次元数
	int cols = clusters[0].metaDescriptors.cols;

	//処理用変数
	std::vector< std::pair<int, int> > imageRankingList;	//各画像のランキング(rank, index)
	std::vector<cv::Mat> rankedDescriptors;					//rankに基づいて並び替えた各clusterの特徴量を保存


	//--------------------step 1 ------------------------------------//

	//画像のランク付け
	for(int i = 0; i < clusters.size(); i++)
	{
		int rank = clusters[i].rankingList.size();					//画像のランク
		std::pair<int , int> list;

		for(int j = 0; j < clusters[i].rankingList.size(); j++)
		{
			rank += clusters[i].rankingList[j];
		}

		list.first = rank;
		list.second = i;

		//画像のランクを保存
		imageRankingList.push_back(list);
	}
	//画像のランキングに基づいて降順に並び替え
	std::sort(imageRankingList.begin(), imageRankingList.end(),std::greater<std::pair<int, int>>() );

	//-----------------step 2 --------------------------------------------//

	//下準備、各clusterのmetaDescriptorsをclusterサイズ(マッチングした数)に基づいて降順に並び替え
	for(int i = 0; i < clusters.size(); i++)
	{
		std::vector<std::pair<int, int>> index;						//並び替え用処理変数pair(rank, 特徴量のindex)
		int num = imageRankingList[i].second;						//画像ランキング
																	//clusters[num]は最も画像ランキングが高いやつのcluster
		for(int j = 0; j < clusters[num].rankingList.size(); j++)
		{
			std::pair<int, int> pair;								//pair(特徴量のrank, 特徴量のindex)
			pair.first = clusters[num].rankingList[j];
			pair.second = j;

			index.push_back(pair);
		}
		//並び替え
		std::sort(index.begin(), index.end(), std::greater<std::pair<int, int>>());

		//各clusterのmetaDescriptorsをclusterサイズ(マッチングした数)に基づいて降順に並び替え
		cv::Mat descriptors = cv::Mat::zeros(index.size(), cols,  CV_8U);
		for(int k = 0; k < index.size(); k++)
		{
			descriptors.row(k) += clusters[num].metaDescriptors.row(index[k].second);

		}
		//画像ランキング順にクラスタの特徴量を保存
		rankedDescriptors.push_back(descriptors);
	}

	//--------------------- step 3 ---------------------------------------// 
	//画像ランキングが高い画像の、最も多くマッチングした特徴量を優先して割り当てる処理
	bool isBeFilled = false;
	metaFeature.descriptors = cv::Mat::zeros(m_budget, cols,  CV_8U);		//初期化
	isBeFilled = createMetaFeature(rankedDescriptors, metaFeature.descriptors);
	
	if(isBeFilled == false)
	{
		addSingleFeatures(clusters, imageRankingList, metaFeature.descriptors);
	}
}

bool FeatureClustering::createMetaFeature(std::vector<cv::Mat> rankedDescriptors, cv::Mat& metaDescriptors)
{
	bool isBeFilled = false;									//特徴量の割り当てが予算まで達したか
	std::vector<int> descSize;									//各descriptorsの残り特徴量数
	int descSum = 0 ;												//特徴量の数の合計

	//初期化
	for(int i = 0; i < rankedDescriptors.size(); i++)
	{
		descSize.push_back(0);
		descSum +=rankedDescriptors[i].rows;
	}

	int total = 0;												//割り当てた数

	//メタ特徴量の割り当て
	while (isBeFilled==false)
	{
		for(int i = 0; i < rankedDescriptors.size(); i++)
		{
			int max = total + 30;
			for(total; total < max; total++)
			{
				if(descSize[i] < rankedDescriptors[i].rows )
				{
					if(total >= m_budget)
					{
						isBeFilled = true;
						return isBeFilled;
						break;
					}else
					{
						metaDescriptors.row(total) += rankedDescriptors[i].row(descSize[i]);
						descSize[i] += 1;
					}
				}else
				{
					break;
				}
			}

			if(total >= m_budget)
			{
				isBeFilled = true;
				return isBeFilled;
				break;
			}

			int sum = std::accumulate(descSize.begin(),descSize.end(), 0);
			if(sum == descSum)
			{
				metaDescriptors.resize(sum, 0);
				isBeFilled = false;
				return isBeFilled;
				break;
			}
		}
	}

	//メタ特徴量が予算に達しなかった場合
	return isBeFilled;
}

void FeatureClustering::addSingleFeatures(std::vector<ClusterOfFeature> clusters,std::vector<std::pair<int, int>> rankingIndex, cv::Mat& metaDescriptors)
{
	std::vector<int> descSize;									//各descriptorsの残り特徴量数
	int startRow = metaDescriptors.rows;
	bool isBeFilled = false;									//特徴量の割り当てが予算まで達したか
	int total = metaDescriptors.rows + 1;	
	int descSum = 0;

	metaDescriptors.resize(m_budget, 0);

	//初期化
	for(int i = 0; i < clusters.size(); i++)
	{
		descSize.push_back(0);
		descSum +=clusters[i].singleDescriptors.rows;
	}

	while (isBeFilled==false)
	{
		for(int i = 0; i < clusters.size(); i++)
		{
			int num = rankingIndex[i].second;						//画像ランキング
		
			int max = total + 30;
			for(total; total < max; total++)
			{
				if(descSize[num] < clusters[num].singleDescriptors.rows )
				{
					if(total >= m_budget)
					{
						isBeFilled = true;
						break;
					}else
					{
						
						metaDescriptors.row(total) += clusters[num].singleDescriptors.row(descSize[num]);
						std::cout << clusters[num].singleDescriptors.row(descSize[num]) << std::endl;
						descSize[num] += 1;
					}
				}else
				{
					break;
				}
			}

			if(total >= m_budget)
			{
				isBeFilled = true;
				break;
			}

			int sum = std::accumulate(descSize.begin(),descSize.end(), 0);
			if(sum == descSum)
			{
				metaDescriptors.resize(sum, 0);
				isBeFilled = true;
				break;
			}
		}
	}
}

void FeatureClustering::showResult(Pattern pattern,ClusterOfFeature cluster)
{

	cv::Mat clusteringResult;
	clusteringResult = pattern.image.clone();

	std::cout << cluster.metaDescriptors.size() << std::endl;
	std::cout << cluster.singleDescriptors.size() << std::endl;

	std::cout << "---------------"<< std::endl;

	for(int i = 0; i < cluster.metaKeypoints.size(); i++)
	{//white
		switch (cluster.rankingList[i])
		{
		case 1://blue
			cv::circle(clusteringResult, cluster.metaKeypoints[i].pt , 1, cv::Scalar(255,0,0),2, CV_FILLED);
			break;
		case 2://green
			cv::circle(clusteringResult, cluster.metaKeypoints[i].pt , 1, cv::Scalar(0,255,0),2, CV_FILLED);
			break;
		case 3://red
			cv::circle(clusteringResult, cluster.metaKeypoints[i].pt , 1, cv::Scalar(0,0,255),2, CV_FILLED);
			break;
		case 4:
			cv::circle(clusteringResult, cluster.metaKeypoints[i].pt , 1, cv::Scalar(255,255,0),2, CV_FILLED);
			break;
		case 5:
			cv::circle(clusteringResult, cluster.metaKeypoints[i].pt , 1, cv::Scalar(250,0,255),2, CV_FILLED);
			break;
		default:
			cv::circle(clusteringResult, cluster.metaKeypoints[i].pt , 1, cv::Scalar(255,255,255),2, CV_FILLED);
			break;

		}
		//特徴点を画像に描画
		std::ostringstream stream;
		stream <<  cluster.rankingList[i];
		std::string rank = stream.str();

		cv::putText(clusteringResult, rank, cluster.metaKeypoints[i].pt, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255,255,255), 1, CV_AA);
	}
	/*
	for(int i = 0; i < cluster.singleKeypoints.size(); i++)
	{//black
		cv::circle(clusteringResult, cluster.singleKeypoints[i].pt , 1, cv::Scalar(0,0,0),2, CV_FILLED);
	}
	*/
	cv::imshow("clusterinResult",clusteringResult);
	/*
	static int count = 0;
	std::stringstream ss;
	ss << count;
	std::string result = "result";
	result +=  ss.str();
	result += ".jpg";
	cv::imwrite(result,clusteringResult);
	count++;
	*/
}

void  FeatureClustering::showMetaFeatures(cv::Mat image,ClusterOfFeature cluster, Pattern meta)
{
	cv::Mat metaResult;
	metaResult = image.clone();
	std::vector<cv::DMatch> matches;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;

	cv::Ptr<cv::FeatureDetector>     detector  = cv::FeatureDetector::create(detectorName);
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::FeatureDetector::create(extractorName);

	cv::Mat gray;

	if (image.channels()  == 3)
        cv::cvtColor(image, gray, CV_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, CV_BGRA2GRAY);
    else if (image.channels() == 1)
        gray = image;

	detector->detect(gray, keypoints);
	extractor->compute(gray,keypoints, descriptors);

	cv::Ptr<cv::DescriptorMatcher>   matcher   = cv::DescriptorMatcher::create(matcherName);
	std::vector<std::vector<cv::DMatch>> knnMatches;
	matcher->knnMatch(descriptors, meta.descriptors, knnMatches, 2);

	for(int j = 0; j < knnMatches.size(); j++)
	{
		if(knnMatches[j].empty() == false)
		{
			const cv::DMatch& bestMatch = knnMatches[j][0];
			const cv::DMatch& betterMatch = knnMatches[j][1];

			float distanceRatio = bestMatch.distance / betterMatch.distance;

			//距離の比が1.5以下の特徴だけ保存
			if(distanceRatio < 0.8f)
			{
				matches.push_back(bestMatch);
			}
		}
	}

	for(int i = 0; i < matches.size(); i++)
	{
		//if(matches[i].distance == 0)
		cv::circle(metaResult, keypoints[matches[i].queryIdx].pt , 1, cv::Scalar(0,0,255),2, CV_FILLED);
	}
	
	cv::imshow("metaResult",metaResult);

	/*
	static int count = 0;
	std::stringstream ss;
	ss << count;
	std::string result = "result";
	result +=  ss.str();
	result += ".jpg";
	cv::imwrite(result,metaResult);
	count++;
	*/
}