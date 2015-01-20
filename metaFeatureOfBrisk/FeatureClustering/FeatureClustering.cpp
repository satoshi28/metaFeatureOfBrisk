#include "stdafx.h"
#include "FeatureClustering.h"

FeatureClustering::FeatureClustering(int buget)
	: m_budget(buget)
{
}

void FeatureClustering::clusterFeatures(std::vector<cv::Mat> images, Pattern& metaFeatures)
{
	ExtractFeatures extract;
	Matching matching;

	std::vector< std::vector<cv::DMatch> > clusterMatches;
	

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
	}
	*/

	//画像をランク付け
	std::vector< std::pair<int, int> > imageRankingList;
	rankImages(clusters, imageRankingList);

	//特徴点の変換
	int basisImgNum = imageRankingList[0].second;
	ClusterOfFeature basisCluster = clusters[basisImgNum];
	AllHomography = matching.getHomography();
	cv::Mat homography;
	for (int i = 0; i < clusters.size(); i++)
	{
		if (i == basisImgNum) continue;

		bool homographyFound = false;
		int num = basisImgNum;
		int n = 1;
		while (homographyFound == false && n < AllHomography[i].size() )
		{
			cv::Mat firstHomography;

			if (i == num){
				n++;
				continue;
			}
			if (i < num)
				firstHomography = AllHomography[i][num - 1];
			else if (i > num)
				firstHomography = AllHomography[i][num];

			//ゼロ行列か判定
			cv::Mat tmp1, tmp2;
			cv::reduce(firstHomography, tmp1, 1, CV_REDUCE_SUM);
			cv::reduce(tmp1, tmp2, 0, CV_REDUCE_SUM);
			if (tmp2.at<double>(0, 0) == 0.0)
			{
				num = imageRankingList[n].second;
			}
			else
			{
				homography = firstHomography;
				homographyFound = true;
			}
			n++;
		}

		if (num != basisImgNum && homographyFound == true)
		{
			cv::Mat secondHomography;
			if (num < basisImgNum)
				secondHomography = AllHomography[num][basisImgNum - 1];
			else if (num > basisImgNum)
				secondHomography = AllHomography[i][num];

			//ゼロ行列か判定
			cv::Mat tmp1, tmp2;
			cv::reduce(secondHomography, tmp1, 1, CV_REDUCE_SUM);
			cv::reduce(tmp1, tmp2, 0, CV_REDUCE_SUM);
			if (tmp2.at<double>(0, 0) == 0.0)
			{
				num = imageRankingList[n].second;
			}
			else
			{
				homography = homography * secondHomography;
			}
		}
		else if (homographyFound == false)
		{
			homography = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
		}
		//特徴点変換処理
		std::vector<cv::KeyPoint> dst_metaKeypoints;
		adjustKeypoints(homography, clusters[i].metaKeypoints, dst_metaKeypoints);

		std::vector<cv::KeyPoint> dst_singleKeypoints;
		adjustKeypoints(homography, clusters[i].singleKeypoints, dst_singleKeypoints);

		//要素の移し替え
		clusters[i].metaKeypoints.swap(dst_metaKeypoints);
		clusters[i].singleKeypoints.swap(dst_singleKeypoints);
		/*
		cv::Mat image = patterns[basisImgNum].image.clone();
		for (int j = 0; j < clusters[i].metaKeypoints.size(); j++)
		{//white
			cv::circle(image, clusters[i].metaKeypoints[j].pt, 1, cv::Scalar(255, 0, 0), 2, CV_FILLED);
		}
		cv::imshow("result", image);
		cv::waitKey(0);
		*/
	}

	//クラスタリング特徴量からメタ特徴量を作成する
	featureBudgeting(clusters, imageRankingList, metaFeatures);
	
	//showMetaFeatures(patterns, metaFeatures);
	//後処理
	patterns.clear();

}


//マッチングリストからマッチングした特徴量同士に基づいて一つの特徴量にまとめる処理
void FeatureClustering::clusterDescriptors( std::vector<std::vector<cv::DMatch>> clusterMatches, std::vector<ClusterOfFeature>& clusters)
{
	for(int i = 0; i < clusterMatches.size(); i++)
	{
		//処理用変数
		ClusterOfFeature cluster;		//マッチング結果から一つにまとめた特徴量
		std::vector<int> matchList;		//登録済みリスト
		matchList.clear();

		//マッチングした特徴量をひとつにまとめメタ特徴量として保存する処理
		for(int j = 0; j < clusterMatches[i].size(); j++)
		{
			int queryId = clusterMatches[i][j].queryIdx;					//マッチングさせた特徴量のID（クエリ番号）
			int rank = 0;					//同じ空間から来た特徴点の数

			//重複して登録しないようにqueryIdが登録済みリストに載っているか確認。ある場合は飛ばす
			for (int k = 0; k < matchList.size(); k++)
			{
				if (matchList[k] == queryId)
					break;
			}

			for(int k = 0; k < clusterMatches[i].size(); k++)	//query番号が一致する特徴量を探す
			{
				if (queryId == clusterMatches[i][k].queryIdx)
					rank += 1;									//必ずrank>1(自身を参照しているから)
			}

			matchList.push_back(queryId);						//マッチングリストにqueryの番号を保存
			
			//特徴量の行に追加
			cluster.metaDescriptors.push_back(patterns[i].descriptors.row(queryId) );
			cluster.metaKeypoints.push_back(patterns[i].keypoints.at(queryId) );
			cluster.rankingList.push_back(rank);

		}
		std::sort(matchList.begin(), matchList.end());

		//単体の特徴量をsingleDescriptorsに保存
		for(int k = 0; k < patterns[i].descriptors.rows; k++)
		{
			//マッチングリストにIDが登録されているか検索
			std::vector<int>::iterator cIter = std::find(matchList.begin(), matchList.end(), k);
			if (cIter != matchList.end())
				break;

			//マッチングリストになければ特徴量の行に追加
			cluster.singleDescriptors.push_back(patterns[i].descriptors.row(k) );
			cluster.singleKeypoints.push_back(patterns[i].keypoints.at(k) );

		}

		//シングル特徴量はランダムに並べ替え			
		//random_shuffle(cluster.singleKeypoints.begin(), cluster.singleKeypoints.end());

		//clusterの最大サイズを保存
		cluster.maxClusterSize = clusterMatches.size();

		//保存
		clusters.push_back(cluster);
	}



}

void FeatureClustering::featureBudgeting(std::vector<ClusterOfFeature> clusters, std::vector< std::pair<int, int> > imageRankingList, Pattern& metaFeature)
{
	//特徴量の次元数
	int cols = clusters[0].metaDescriptors.cols;

	//処理用変数
	std::vector<cv::Mat> rankedDescriptors;						//rankに基づいて並び替えた各clusterの特徴量を保存
	std::vector< std::vector<cv::KeyPoint>> rankedKeypoints;	//rankに基づいて並び替えた各clusterの特徴点を保存
	std::vector<int> imgNumbers;

	//-----------------step 2 --------------------------------------------//

	//下準備、各clusterのmetaDescriptorsをclusterサイズ(マッチングした数)に基づいて降順に並び替え
	for(int i = 0; i < clusters.size(); i++)
	{
		std::vector<std::pair<int, int>> index;						//メタ特徴量のランキングリスト(rank, 特徴量のindex)
		int num = imageRankingList[i].second;						//画像のランキング
																	//clusters[num]は最も画像ランキングが高いやつのcluster
		//各特徴量をマッチングした数に基づいて評価
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
	
	//メタ特徴量で予算が埋まらなかったら単体の特徴量を割り当てる
	if(isBeFilled == false)
		addSingleFeatures(clusters, imageRankingList, metaFeature);

}

void FeatureClustering::rankImages(std::vector<ClusterOfFeature> clusters, std::vector< std::pair<int, int> >& imageRankingList)
{
	imageRankingList.clear();

	//画像ランキングの作成(clustersのサイズは画像の枚数)
	for (int i = 0; i < clusters.size(); i++)
	{
		int rank = 0;												//画像のランキング
		std::pair<int, int> list;									//ランキングのリスト

		//各画像に対してメタ特徴量を構成する特徴量の数に基づいて投票
		for (int j = 0; j < clusters[i].rankingList.size(); j++)
			rank += clusters[i].rankingList[j];

		//画像のランクを保存
		list.first = rank;
		list.second = i;
		imageRankingList.push_back(list);
	}

	//画像のランキングに基づいてリストを降順に並び替え
	std::sort(imageRankingList.begin(), imageRankingList.end(), std::greater<std::pair<int, int>>());

}

bool FeatureClustering::createMetaFeature(std::vector<cv::Mat> rankedDescriptors,std::vector< std::vector<cv::KeyPoint>> rankedKeypoints,std::vector<int> imgNumbers, Pattern& metaFeature)
{
	bool isBeFilled = false;									//特徴量の割り当てが予算まで達したか
	std::vector<int> descSize;									//各descriptorsの残り特徴量数
	int descSum = 0 ;											//特徴量の数の合計
	std::pair<bool, int> paramOfKeypoint;						//メタ特徴量か単体の特徴量かを示す変数(デバック用)
	
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
			int max = total + 30;		//30点が埋まるごとに画像を変える
			for(total; total < max; total++)
			{
				if(descSize[i] < rankedDescriptors[i].rows )
				{
					if(total >= m_budget)	//予算が埋まった時の処理
					{
						isBeFilled = true;
						return isBeFilled;
						break;
					}else
					{
						//keypointのパラメータを保存
						paramOfKeypoint.first = true;
						paramOfKeypoint.second = imgNumbers[i];
						metaFeature.paramOfKeypoints.push_back(paramOfKeypoint);

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
			}

			int sum = std::accumulate(descSize.begin(),descSize.end(), 0);
			if(sum == descSum)
			{
				metaFeature.descriptors.resize(sum, 0);
				isBeFilled = false;
				return isBeFilled;
			}
		}
	}

	//メタ特徴量が予算に達しなかった場合
	return isBeFilled;
}

void FeatureClustering::addSingleFeatures(std::vector<ClusterOfFeature> clusters,std::vector<std::pair<int, int>> rankingIndex, Pattern& metaFeature)
{
	std::vector<int> descSize;									//各descriptorsの残り特徴量数
	bool isBeFilled = false;									//特徴量の割り当てが予算まで達したか
	int total = metaFeature.descriptors.rows;					//メタ特徴量数の合計
	int descSum = 0;											//特徴量の数の合計
	std::pair<bool, int> paramOfKeypoint;						//メタ特徴量か単体の特徴量かを示す変数(デバック用)

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
			int max = total + 30;									//30点が埋まるごとに画像を変える
			for(total; total < max; total++)
			{
				if(descSize[num] < clusters[num].singleDescriptors.rows )
				{
					if(total >= m_budget)							//予算が埋まった時の処理
					{
						isBeFilled = true;
						break;
					}else
					{
						//keypointのパラメータを保存
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

bool FeatureClustering::adjustKeypoints(const cv::Mat_<double>& H, std::vector<cv::KeyPoint> src_keypoints, std::vector<cv::KeyPoint>& dst_keyponts)
{
	//ゼロ行列か判定
	cv::Mat tmp1, tmp2;
	cv::reduce(H, tmp1, 1, CV_REDUCE_SUM);
	cv::reduce(tmp1, tmp2, 0, CV_REDUCE_SUM);
	if (tmp2.at<double>(0, 0) == 0.0)
	{
		std::cout << "zeros matrix" << std::endl;
		return false;
	}

	//座標変換処理
	for (int j = 0; j < src_keypoints.size(); j++)
	{
		double x = src_keypoints[j].pt.x;
		double y = src_keypoints[j].pt.y;

		cv::KeyPoint keypoint;
		keypoint.pt.x = (float)(H(0, 0) * x + H(0, 1) * y + H(0, 2)) / (H(2, 0) * x + H(2, 1) + H(2, 2));
		keypoint.pt.y = (float)(H(1, 0) * x + H(1, 1) * y + H(1, 2)) / (H(2, 0) * x + H(2, 1) + H(2, 2));

		dst_keyponts.push_back(keypoint);
	}

	return true;
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

cv::Mat FeatureClustering::getModeDescriptors(cv::Mat trainDescriptors)
{
	//入力された特徴点の特徴量が2以下の場合は最後の要素を返す
	if (trainDescriptors.rows <= 2)
		return trainDescriptors.row(trainDescriptors.rows - 1);

	std::vector < std::vector < std::vector<bool> >> bitTrainDescriptors;
	unsigned char mask[8] = { 0 };		//unsigned charをbool配列に変換するためのマスク

	//マスクの作成
	mask[0] = 1;
	for (int i = 1; i < 8; i++)
	{
		mask[i] = mask[i - 1] * 2;
	}

	//特徴量をビット配列に変換
	for (int i = 0; i < trainDescriptors.rows; i++)
	{
		cv::Mat descriptors = trainDescriptors.row(i).clone();	//一つの特徴点の特徴量
		std::vector<std::vector<bool> > bitDescriptors;		//ビット配列に変換した特徴量
		//std::cout << descriptors << std::endl;

		for (int j = 0; j < 64; j++)		//dimension
		{
			unsigned char descriptor = descriptors.at<unsigned char>(0, j);	//次元ごとの特徴量
			std::vector<bool> bitArray(8);												//特徴量のビット表現

			for (int k = 0; k < 8; k++)		//bit
			{
				//各ビットを抽出
				unsigned char bit = descriptor & mask[k];

				if (bit != 0)
					bitArray[k] = true;
				else
					bitArray[k] = false;
			}
			//ビット表現にした特徴量の格納
			bitDescriptors.push_back(bitArray);
		}
		bitTrainDescriptors.push_back(bitDescriptors);
	}

	//複数の特徴量から最頻値のビットに置き換えた特徴量へ変換する
	cv::Mat modeDescriptors = cv::Mat::zeros(cv::Size(64, 1), CV_8U);		//ビット配列に変換した特徴量
	for (int i = 0; i < 64; i++)		//dimension
	{
		std::vector<unsigned char> bitArray(8);
		for (int j = 0; j < 8; j++)
		{
			int countOfZero = 0;
			int countOfOne = 0;

			for (int k = 0; k < bitTrainDescriptors.size(); k++)		//bit
			{
				bool isOne = bitTrainDescriptors[k][i][j];
				if (isOne == true)
					countOfOne++;
				else
					countOfZero++;
			}

			//最頻値に基づいてビットを決定
			if (countOfOne < countOfZero)
				bitArray[j] = 0;
			else if (countOfOne >= countOfZero)
				bitArray[j] = 1;
		}

		//bitArrayからunsignend charに変換
		unsigned char dim = 0;
		for (int j = 0; j < bitArray.size(); j++)
		{
			if (bitArray[j] != 0)
				dim += (unsigned char)std::pow(2.0, (double)j);
		}
		//Matに格納
		modeDescriptors.at<unsigned char>(0, i) = dim;
	}
	//std::cout << modeDescriptors << std::endl;
	return modeDescriptors;
}