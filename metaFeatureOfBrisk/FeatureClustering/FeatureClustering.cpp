
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
	//
	std::cout << "OK" << std::endl;
	/*
	for(int i = 0; i < patterns.size(); i++)
	{
		showResult(patterns[i], clusters[i]);
		cv::waitKey(0);
	}*/
	//クラスタリング特徴量からメタ特徴量を作成する
	featureBudgeting(clusters, metaFeatures);
	
	//showMetaFeatures(patterns, metaFeatures);
	//後処理
	patterns.clear();

}



void FeatureClustering::clusterDescriptors( std::vector<std::vector<cv::DMatch>> clusterMatches, std::vector<ClusterOfFeature>& clusters)
{

	int rank = 1;			//同じ空間から来た特徴点の数
	int id;
	std::vector<int> matchList;


	if(m_enableMultipleRatioTest == true)
	{

		for(int i = 0; i < clusterMatches.size(); i++)
		{
			int cols = patterns[i].descriptors.cols;

			//処理用変数
			ClusterOfFeature cluster;
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

					//保存処理 
					//特徴量の行に追加
					cluster.metaDescriptors.push_back(patterns[i].descriptors.row(queryIdx) );


					#if _DEBUG
					//特徴点を保存
					cluster.metaKeypoints.push_back(patterns[i].keypoints.at(queryIdx) );
					#endif

					//特徴量のランクを保存
					cluster.rankingList.push_back(rank);

					rank = 1;
				}

			}
			//単体の特徴量をsingleDescriptorsに保存
			for(int k = 0; k < patterns[i].descriptors.rows; k++)
			{
				for(int m = 0; m < matchList.size(); m++)
				{
					if(matchList[m] == k)	//マッチングリストに載っていれば抜ける
					{
						break;
					}
					if(m == matchList.size()-1)
					{
						//特徴量の行に追加
						cluster.singleDescriptors.push_back( patterns[i].descriptors.row(k) );

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
			matchList.clear();

			for(int j = 1; j < clusterMatches[i].size(); j++)
			{
				id = clusterMatches[i][j - 1].queryIdx;

				//query番号が一致する特徴量を探す
				for(int k = 0; k < clusterMatches[i].size(); k++)
				{
					if(id == clusterMatches[i][k].queryIdx)
					{
						rank += 1;
					}
				}

				int queryIdx = clusterMatches[i][j].queryIdx;		//queryのインデックス
				matchList.push_back(queryIdx);						//マッチングリストにqueryの番号を保存

				//特徴量の行に追加
				cluster.metaDescriptors.push_back(patterns[i].descriptors.row(queryIdx) );
		
				#if _DEBUG
				//特徴点を保存
				cluster.metaKeypoints.push_back(patterns[i].keypoints.at(queryIdx) );
				#endif

				//特徴量のランクを保存
				cluster.rankingList.push_back(rank);

				rank = 0;
			}

			//単体の特徴量をsingleDescriptorsに保存
			for(int k = 0; k < patterns[i].descriptors.rows; k++)
			{
				for(int m = 0; m < matchList.size(); m++)
				{
					if(matchList[m] == k)
					{
						break;
					}
					if(m == matchList.size()-1)	//マッチリストの最後まで見つからなかった場合、シングル特徴量に追加
					{
						//特徴量の行に追加
						cluster.singleDescriptors.push_back(patterns[i].descriptors.row(k) );

						#if _DEBUG
						//特徴点を保存
						cluster.singleKeypoints.push_back(patterns[i].keypoints.at(k) );
						#endif
					}
				}

				//マッチングしたものがなかった場合
				if(matchList.size() == 0)
				{
					//特徴量の行に追加
					cluster.singleDescriptors.push_back(patterns[i].descriptors.row(k) );

					#if _DEBUG
					//特徴点を保存
					cluster.singleKeypoints.push_back(patterns[i].keypoints.at(k) );
					#endif
				}

			}
			//シングル特徴量はランダムに並べ替え			
			random_shuffle(cluster.singleKeypoints.begin(), cluster.singleKeypoints.end());

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
	std::vector< std::vector<cv::KeyPoint>> rankedKeypoints;					//rankに基づいて並び替えた各clusterの特徴点を保存
	std::vector<int> imgNumbers;

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
		cv::Mat descriptors;
		std::vector<cv::KeyPoint> keypoints;

		for(int k = 0; k < index.size(); k++)
		{
			descriptors.push_back( clusters[num].metaDescriptors.row(index[k].second) );
			keypoints.push_back( clusters[num].metaKeypoints[index[k].second] );
			
		}
		//画像ランキング順にクラスタの特徴量を保存
		rankedDescriptors.push_back(descriptors);
		rankedKeypoints.push_back(keypoints);
		imgNumbers.push_back(num);
	}

	//--------------------- step 3 ---------------------------------------// 
	//画像ランキングが高い画像の、最も多くマッチングした特徴量を優先して割り当てる処理
	bool isBeFilled = false;
	isBeFilled = createMetaFeature(rankedDescriptors,rankedKeypoints,imgNumbers, metaFeature);
	
	if(isBeFilled == false)
	{
		addSingleFeatures(clusters, imageRankingList, metaFeature);
	}
}

bool FeatureClustering::createMetaFeature(std::vector<cv::Mat> rankedDescriptors,std::vector< std::vector<cv::KeyPoint>> rankedKeypoints,std::vector<int> imgNumbers, Pattern& metaFeature)
{
	bool isBeFilled = false;									//特徴量の割り当てが予算まで達したか
	std::vector<int> descSize;									//各descriptorsの残り特徴量数
	int descSum = 0 ;												//特徴量の数の合計
	std::pair<bool, int>			paramOfKeypoint;
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
						paramOfKeypoint.first = true;
						paramOfKeypoint.second = imgNumbers[i];

						metaFeature.paramOfKeypoints.push_back(paramOfKeypoint);						//keypointのパラメータを保存

						metaFeature.descriptors.push_back( rankedDescriptors[i].row(descSize[i]) );
						metaFeature.keypoints.push_back( rankedKeypoints[i][ descSize[i] ]);

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
				metaFeature.descriptors.resize(sum, 0);
				isBeFilled = false;
				return isBeFilled;
				break;
			}
		}
	}

	//メタ特徴量が予算に達しなかった場合
	return isBeFilled;
}

void FeatureClustering::addSingleFeatures(std::vector<ClusterOfFeature> clusters,std::vector<std::pair<int, int>> rankingIndex, Pattern& metaFeature)
{
	std::vector<int> descSize;									//各descriptorsの残り特徴量数
	int startRow = metaFeature.descriptors.rows;
	bool isBeFilled = false;									//特徴量の割り当てが予算まで達したか
	int total = metaFeature.descriptors.rows;	
	int descSum = 0;
	std::pair<bool, int>			paramOfKeypoint;

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
						paramOfKeypoint.first = false;
						paramOfKeypoint.second = num;

						metaFeature.paramOfKeypoints.push_back(paramOfKeypoint);						//keypointのパラメータを保存
						
						metaFeature.descriptors.push_back( clusters[num].singleDescriptors.row(descSize[num]) );
						metaFeature.keypoints.push_back(clusters[num].singleKeypoints[ descSize[num] ] );
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
				metaFeature.descriptors.resize(sum, 0);
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

void  FeatureClustering::showMetaFeatures(std::vector<Pattern> patterns,Pattern metaFeature)
{
	for(int i =0; i< patterns.size();i++)
	{
		cv::Mat metaResult;
		metaResult = patterns[i].image.clone();

		for(int j = 0; j < metaFeature.paramOfKeypoints.size(); j++)
		{
			if(metaFeature.paramOfKeypoints[j].second == i)
			{
				if(metaFeature.paramOfKeypoints[j].first == true)//複数の画像で見つかったか
					cv::circle(metaResult,metaFeature.keypoints[j].pt , 2, cv::Scalar(0,0,255),2, CV_FILLED);
				else													//シングルの特徴
					cv::circle(metaResult,metaFeature.keypoints[j].pt , 1, cv::Scalar(0,0,0),2, CV_FILLED);

			}
		}
		
		cv::imshow("metaResult",metaResult);
		cv::waitKey(0);
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
}