
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
		cv::Mat falseImage = cv::imread("C:\\Users\\satoshi\\Documents\\Image\\falseImage.JPG",1);
		images.push_back(falseImage);
	}

	//処理用
	// 特徴量をPatternに保存
	extract.getFeatures(images,patterns);
	// すべての画像同士をマッチングする
	matching.getMatches(patterns, clusterMatches);

	homographyes = matching.getHomography();

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
	}
	*/
	//クラスタリング特徴量からメタ特徴量を作成する
	featureBudgeting(clusters, metaFeatures);
	
	//showMetaFeatures(patterns, metaFeatures);
	//後処理
	patterns.clear();

}



void FeatureClustering::clusterDescriptors( std::vector<std::vector<cv::DMatch>> clusterMatches, std::vector<ClusterOfFeature>& clusters)
{

	int rank = 0;			//同じ空間から来た特徴点の数
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

					//特徴点を保存
					cluster.metaKeypoints.push_back(patterns[i].keypoints.at(queryIdx) );

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

						//特徴点を保存
						cluster.singleKeypoints.push_back(patterns[i].keypoints.at(k) );
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

			for(int j = 0; j < clusterMatches[i].size(); j++)
			{
				id = clusterMatches[i][j].queryIdx;

				//query番号が一致する特徴量を探す
				for(int k = 0; k < clusterMatches[i].size(); k++)
				{//必ずrank>1(自身を参照しているから)
					if(id == clusterMatches[i][k].queryIdx)
					{
						rank += 1;
					}
				}

				bool isBeFind = false;
				int queryIdx = clusterMatches[i][j].queryIdx;		//queryのインデックス

				for(int k=0; k < matchList.size(); k++)
				{
					if(matchList[k] == queryIdx)
						isBeFind = true;
				}
				//すでに見つかったqueryだったら追加しない
				if(isBeFind == false)
				{
					matchList.push_back(queryIdx);						//マッチングリストにqueryの番号を保存
					
					//特徴量の行に追加
					cluster.metaDescriptors.push_back(patterns[i].descriptors.row(queryIdx) );
		
					//特徴点を保存
					cluster.metaKeypoints.push_back(patterns[i].keypoints.at(queryIdx) );

					//特徴量のランクを保存
					cluster.rankingList.push_back(rank);
				}
				rank = 0;
			}

			//std::sort(matchList.begin(), matchList.end());


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

						//特徴点を保存
						cluster.singleKeypoints.push_back(patterns[i].keypoints.at(k) );
					}
				}

				//マッチングしたものがなかった場合
				if(matchList.size() == 0)
				{
					//特徴量の行に追加
					cluster.singleDescriptors.push_back(patterns[i].descriptors.row(k) );

					//特徴点を保存
					cluster.singleKeypoints.push_back(patterns[i].keypoints.at(k) );
				}

			}
			//シングル特徴量はランダムに並べ替え			
			//random_shuffle(cluster.singleKeypoints.begin(), cluster.singleKeypoints.end());

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

	/* 
	* 最高ランクのクラスタ(Imax)の画像に，各クラスタの特徴点を投影
	* 同じ位置に来た特徴点のclusterサイズをもとの特徴点のクラスタサイズに加算
	* inlinerとして取り除かれた特徴点はImaxに新規に作成，クラスタサイズはそのまま
	* step2は先にやっとく
	* クラスタサイズが高いものからメタ特徴量に割り当てる
	*/
	clusterToMetaFeature(clusters, homographyes, imageRankingList[0].second, metaFeature);

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
	
	for(int i = 0; i < cluster.singleKeypoints.size(); i++)
	{//black
		cv::circle(clusteringResult, cluster.singleKeypoints[i].pt , 1, cv::Scalar(0,0,0),2, CV_FILLED);
	}
	
	cv::imshow("clusterinResult",clusteringResult);
	/*
	static int count = 0;
	std::stringstream ss;
	ss << count;
	std::string result = "matchingResult";
	result +=  ss.str();
	result += ".png";
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
		int metaFeaturesSize =0;


		for(int j = 0; j < metaFeature.paramOfKeypoints.size(); j++)
		{
			if(metaFeature.paramOfKeypoints[j].second == i)
			{
				if(metaFeature.paramOfKeypoints[j].first == true)//複数の画像で見つかったか
					cv::circle(metaResult,metaFeature.keypoints[j].pt , 4, cv::Scalar(0,0,255),-1, CV_AA);
				else													//シングルの特徴
					cv::circle(metaResult,metaFeature.keypoints[j].pt , 4, cv::Scalar(0,0,0),-1, CV_AA);

				metaFeaturesSize++;

			}
		}
		
		cv::imshow("metaResult",metaResult);
		std::cout << metaFeaturesSize << std::endl;
		cv::waitKey(0);
		/*
		static int count = 0;
		std::stringstream ss;
		ss << count;
		std::string result = "metaResult";
		result +=  ss.str();
		result += ".png";
		cv::imwrite(result,metaResult);
		count++;
		*/
	}
}

void FeatureClustering::clusterToMetaFeature(std::vector<ClusterOfFeature> clusters,std::vector<std::vector<cv::Mat>> allHomographyes, int basisImgNum, Pattern& metaFeature)
{
	//基準のクラスタ
	ClusterOfFeature basisCluster;
	basisCluster = clusters[basisImgNum];

	
/*
	//その他のクラスタ
	std::vector<ClusterOfFeature> trainClusters;
	for(int i = 0; i < clusters.size(); i++)
	{
		if(i == basisImgNum) continue;

		trainClusters.push_back(clusters[i]);
	}
	//変換後のクラスタ
	std::vector<ClusterOfFeature> transformedClusters;
	std::vector<cv::Mat> homographyes;

	//macherを用意
	std::vector< cv::Ptr<cv::DescriptorMatcher> > matchers( trainClusters.size());
	for(int k = 0; k <trainClusters.size(); k++)
	{
		matchers[k] = cv::DescriptorMatcher::create(matcherName);

		std::vector<cv::Mat> descriptors(1);
		descriptors[0] = trainClusters[k].metaDescriptors.clone();
		matchers[k]->add(descriptors);

		matchers[k]->train();
	}

	//最近傍点の探索
	for(int i = 0; i < trainClusters.size() ; i++)
	{
		//knnマッチング
		std::vector< std::vector<cv::DMatch>>  knnMatches;

		// queryとmatcherに保存されている特徴量をknn構造体を用いて最近傍点を検索する.
		matchers[i]->knnMatch(basisCluster.metaDescriptors, knnMatches, 2);
	
		//
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
					correctMatches.push_back(bestMatch);
				}
			}
		}

		cv::Mat homography;
		//幾何学的整合性チェック
		if(correctMatches.size() < 8)
		{
			correctMatches.clear();
			//ホモグラフィ行列が推定できなかった場合は単位行列を格納
			cv::Mat eye = cv::Mat::zeros(3, 3, CV_64FC1); // 単位行列を生成
			homographyes.push_back(eye);
			continue;
		}
		
		std::vector<cv::Point2f>  queryPoints, trainPoints; 
		for(int j = 0; j < correctMatches.size(); j++)
		{
			queryPoints.push_back(trainClusters[i].metaKeypoints[correctMatches[j].trainIdx].pt);
			trainPoints.push_back(basisCluster.metaKeypoints[correctMatches[j].queryIdx].pt);
		}

		//幾何学的整合性チェック
		std::vector<unsigned char> inliersMask(queryPoints.size() );

		//幾何学的整合性チェックによって当たり値を抽出
		homography = cv::findHomography( queryPoints, trainPoints, CV_FM_RANSAC, 3, inliersMask);

		std::vector<cv::DMatch> inliers;
		for(size_t j =0 ; j < inliersMask.size(); j++)
		{
			if(inliersMask[j])
				inliers.push_back(correctMatches[j]);
		}

		correctMatches.swap(inliers);
		homographyes.push_back(homography);



		cv::Mat result;
		if(i < basisImgNum)	//
		{
			cv::drawMatches(patterns[basisImgNum].image, basisCluster.metaKeypoints,patterns[i].image , clusters[i].metaKeypoints,correctMatches, result);
			
		}
		else if(i > basisImgNum)
		{
			cv::drawMatches(patterns[basisImgNum].image, basisCluster.metaKeypoints,patterns[i-1].image , clusters[i-1].metaKeypoints,correctMatches, result);
		}				//初期化
		knnMatches.clear();
		correctMatches.clear();

	}

	for(int i = 0; i < trainClusters.size() ; i++)
	{
		//ゼロ行列か判定
		cv::Mat tmp1,tmp2;
		cv::reduce(homographyes[i], tmp1, 1, CV_REDUCE_SUM);
		cv::reduce(tmp1, tmp2, 0, CV_REDUCE_SUM);
		if(tmp2.at<double>(0,0) == 0.0)
			continue;

		// 変換
		double mm[3][3];
		for(int j = 0; j < 3; j++)
		{
			for(int k = 0; k < 3; k++)
			{
				mm[j][k] = homographyes[i].at<double>(j,k);
			}
		}

		ClusterOfFeature transformedCluster;
		transformedCluster = trainClusters[i];

		for(int j = 0; j < transformedCluster.metaKeypoints.size(); j++)
		{
			double x = transformedCluster.metaKeypoints[j].pt.x;
			double y = transformedCluster.metaKeypoints[j].pt.y;

			double t_x = (mm[0][0]*x + mm[0][1]*y + mm[0][2])/(mm[2][0]*x + mm[2][1] + mm[2][2]);
			double t_y = (mm[1][0]*x + mm[1][1]*y + mm[1][2])/(mm[2][0]*x + mm[2][1] + mm[2][2]);

			transformedCluster.metaKeypoints[j].pt.x = t_x;
			transformedCluster.metaKeypoints[j].pt.y = t_y;

		}

		for(int j = 0; j < transformedCluster.singleKeypoints.size(); j++)
		{
			double x = transformedCluster.singleKeypoints[j].pt.x;
			double y = transformedCluster.singleKeypoints[j].pt.y;

			double t_x = (mm[0][0]*x + mm[0][1]*y + mm[0][2])/(mm[2][0]*x + mm[2][1] + mm[2][2]);
			double t_y = (mm[1][0]*x + mm[1][1]*y + mm[1][2])/(mm[2][0]*x + mm[2][1] + mm[2][2]);

			transformedCluster.singleKeypoints[j].pt.x = t_x;
			transformedCluster.singleKeypoints[j].pt.y = t_y;

		}

		cv::Mat resultImg = patterns[basisImgNum].image.clone();

		for(int j = 0; j < basisCluster.metaKeypoints.size(); j++)
		{//white
			cv::circle(resultImg, basisCluster.metaKeypoints[j].pt , 2, cv::Scalar(0,0,255),2, CV_FILLED);
		}
		
		for(int j = 0; j < transformedCluster.metaKeypoints.size(); j++)
		{//white
			cv::circle(resultImg, transformedCluster.metaKeypoints[j].pt , 1, cv::Scalar(255,0,0),2, CV_FILLED);
		}

		cv::imshow("result", resultImg);
		cv::waitKey(0);

	}
*/
	
		
	for(int i = 0; i < clusters.size(); i++)
	{
		//基準の場合は飛ばす
		if(i == basisImgNum) continue;

		//マッチングの際推定したホモグラフィの取得
		cv::Mat homography;
		if(i < basisImgNum)	//
		{
			homography = homographyes[i][basisImgNum - 1];
		}
		else if(i > basisImgNum)
		{
			homography = homographyes[i][basisImgNum];
		}

		//ゼロ行列か判定
		cv::Mat tmp1,tmp2;
		cv::reduce(homography, tmp1, 1, CV_REDUCE_SUM);
		cv::reduce(tmp1, tmp2, 0, CV_REDUCE_SUM);
		if(tmp2.at<double>(0,0) == 0.0)
			continue;



		std::cout << homography << std::endl;

		double mm[3][3];
		for(int j = 0; j < 3; j++)
		{
			for(int k = 0; k < 3; k++)
			{
				mm[j][k] = homography.at<double>(j,k);
				std::cout << mm[j][k] << std::endl;
			}
		}

		ClusterOfFeature transformedCluster;
		transformedCluster = clusters[i];
		//座標変換処理
		for(int j = 0; j < transformedCluster.metaKeypoints.size(); j++)
		{
			double x = transformedCluster.metaKeypoints[j].pt.x;
			double y = transformedCluster.metaKeypoints[j].pt.y;

			double t_x = (mm[0][0]*x + mm[0][1]*y + mm[0][2])/(mm[2][0]*x + mm[2][1] + mm[2][2]);
			double t_y = (mm[1][0]*x + mm[1][1]*y + mm[1][2])/(mm[2][0]*x + mm[2][1] + mm[2][2]);

			transformedCluster.metaKeypoints[j].pt.x = t_x;
			transformedCluster.metaKeypoints[j].pt.y = t_y;

		}
		/*
		for(int j = 0; j < transformedCluster.singleKeypoints.size(); j++)
		{
			double x = transformedCluster.singleKeypoints[j].pt.x;
			double y = transformedCluster.singleKeypoints[j].pt.y;

			double t_x = (mm[0][0]*x + mm[0][1]*y + mm[0][2])/(mm[2][0]*x + mm[2][1] + mm[2][2]);
			double t_y = (mm[1][0]*x + mm[1][1]*y + mm[1][2])/(mm[2][0]*x + mm[2][1] + mm[2][2]);

			transformedCluster.singleKeypoints[j].pt.x = t_x;
			transformedCluster.singleKeypoints[j].pt.y = t_y;

		}*/

		//
		ClusterOfFeature refinedCluster;
		//ユークリッド距離計算
		for(int j = 0; j < transformedCluster.metaKeypoints.size(); j++)
		{
			double minDist = 10;
			int minDistanceKeypointNum = -1;
			double x1 = transformedCluster.metaKeypoints[j].pt.x;
			double y1 = transformedCluster.metaKeypoints[j].pt.y;

			for(int k = 0; k < basisCluster.metaKeypoints.size(); k++)
			{
				double x2 = basisCluster.metaKeypoints[k].pt.x;
				double y2 = basisCluster.metaKeypoints[k].pt.y;

				double dist = sqrt( (x1 -x2)*(x1 -x2) + (y1 - y2)*(y1 - y2) );

				if(dist < minDist)
				{
					minDist = dist;
					minDistanceKeypointNum = k;
				}
			}

			//閾値以内で最小のユークリッド距離をもつ点を基本となるクラスタに置換
			if(minDist < 6.0)
			{
				basisCluster.rankingList[minDistanceKeypointNum] += transformedCluster.rankingList[j];

				//descriptorを平均化
				//
			}else //無かったら追加
			{/*
				basisCluster.metaKeypoints.push_back(transformedCluster.metaKeypoints[j]);
				basisCluster.metaDescriptors.push_back(transformedCluster.metaDescriptors.row(j) );
				basisCluster.rankingList.push_back(transformedCluster.rankingList[j]);
				*/
			}
		}

		cv::Mat resultImg = patterns[basisImgNum].image.clone();

		for(int i = 0; i < basisCluster.metaKeypoints.size(); i++)
		{//white
			cv::circle(resultImg, basisCluster.metaKeypoints[i].pt , 2, cv::Scalar(0,0,255),2, CV_FILLED);
		}
		
		for(int i = 0; i < transformedCluster.metaKeypoints.size(); i++)
		{//white
			cv::circle(resultImg, transformedCluster.metaKeypoints[i].pt , 1, cv::Scalar(255,0,0),2, CV_FILLED);
		}
		
		//cv::imshow("result", resultImg);
		//cv::waitKey(0);

/*		static int count = 0;
		std::stringstream ss;
		ss << count;
		std::string result = "matchingPoint";
		result +=  ss.str();
		result += ".jpg";
		cv::imwrite(result,resultImg);
		count++;
	*/	

	}

	std::vector<std::pair<int, int>> index;						//並び替え用処理変数pair(rank, 特徴量のindex)
															//clusters[num]は最も画像ランキングが高いやつのcluster
	for(int i = 0; i < basisCluster.rankingList.size(); i++)
	{
		std::pair<int, int> pair;								//pair(特徴量のrank, 特徴量のindex)
		pair.first = basisCluster.rankingList[i];
		pair.second = i;

		index.push_back(pair);
	}
	//並び替え
	std::sort(index.begin(), index.end(), std::greater<std::pair<int, int>>());

	//各clusterのmetaDescriptorsをclusterサイズ(マッチングした数)に基づいて降順に並び替え

	for(int j =0; j < 200; j++)
	{
		if(j >= (int)basisCluster.metaKeypoints.size() )
		{
			break;
		}

		int num = index[j].second;
		metaFeature.descriptors.push_back(basisCluster.metaDescriptors.row(num) );
		metaFeature.keypoints.push_back(basisCluster.metaKeypoints[num] );


	}
	
	cv::Mat image = patterns[basisImgNum].image.clone();
	for(int i = 0; i < metaFeature.keypoints.size(); i++)
	{//white
		cv::circle(image, metaFeature.keypoints[i].pt , 1, cv::Scalar(255,0,0),2, CV_FILLED);
	}
		
	cv::imshow("result", image);
	cv::waitKey(0);

}