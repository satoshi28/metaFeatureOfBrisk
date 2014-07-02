#include "stdafx.h"
#include "Matching.h"

Matching::Matching(cv::Ptr<cv::DescriptorMatcher> matcher)
	: m_matcher(matcher)
{
}

Matching::Matching(bool flag)
{
	m_enableMultipleRatioTest = flag;
}

void Matching::getMatches(const std::vector<Pattern> patterns, std::vector< std::vector<cv::DMatch> >& clusterMatches)
{
	dataSetSize = patterns.size();
	imgNumberOfAdjstment = 0;

	//
	//すべての画像をマッチングする
	for(int i = 0; i < dataSetSize; i++)
	{
		//処理用変数宣言
		std::vector<cv::KeyPoint> queryKeypoints;
		cv::Mat queryDescriptors;						//クエリディスクリプタ
		std::vector<std::vector<cv::KeyPoint>> trainKeypoints;
		std::vector<cv::Mat> trainDescriptors;			//訓練ディスクリプタ
		std::vector<cv::DMatch> matches;				//マッチングで得られた結果


		//macherを用意
		std::vector< cv::Ptr<cv::DescriptorMatcher> > matchers( dataSetSize -1 );
		for(int k = 0; k < dataSetSize -1; k++)
		{
			matchers[k] = cv::DescriptorMatcher::create(matcherName);
		}


		//処理部
		queryDescriptors = patterns[i].descriptors;		//クエリ画像
		queryKeypoints = patterns[i].keypoints;

		for(int j = 0; j < dataSetSize; j++)
		{
			//クエリ画像以外を訓練ディスクリプタに格納
			if(i != j)
			{
				trainDescriptors.push_back( patterns[j].descriptors );
				trainKeypoints.push_back(patterns[j].keypoints);
			}
		}

		//trainデータをmatchersに追加
		train(trainDescriptors, matchers);

		// Get matches
		match( queryKeypoints, queryDescriptors,trainKeypoints,  matchers, matches);

		//結果を格納
		clusterMatches.push_back(matches);

		for(int j =0; j < dataSetSize; j++)
		{
			//クエリ画像以外を訓練ディスクリプタに格納
			if(i != j)
			{
				cv::Mat img1,img2,result;
				std::vector<cv::DMatch> match;
				img1 = patterns[i].image.clone();
				img2 = patterns[j].image.clone();

				for(int k = 0; k < matches.size();k++)
				{
					if(j == matches[k].imgIdx)
					{
						match.push_back(matches[k]);
					}
				}

				cv::drawMatches(img1,patterns[i].keypoints,img2 ,patterns[j].keypoints, match, result);

				static int count = 0;
				std::stringstream ss;
				ss << count;
				std::string name = "matching";
				name +=  ss.str();
				name += ".jpg";
				cv::imwrite(name,result);
				count++;

			}
		}

		//matchesの画像番号修正用
		imgNumberOfAdjstment++;
	}
}


void Matching::train(const std::vector<cv::Mat> trainDescriptors, std::vector<cv::Ptr<cv::DescriptorMatcher> >& matchers)
{

	std::vector<cv::Mat> descriptors(1);

	for(int i = 0; i < trainDescriptors.size(); i++)
	{
		// API of cv::DescriptorMatcher is somewhat tricky
		// First we clear old train data:
		matchers[i]->clear();

		// Then we add vector of descriptors (each descriptors matrix describe one image). 
		// This allows us to perform search across multiple images:

		descriptors[0]= trainDescriptors[i].clone();
		matchers[i]->add(descriptors);

		// After adding train data perform actual train:
		matchers[i]->train();
	}
}



void Matching::match(std::vector<cv::KeyPoint> queryKeypoints,cv::Mat queryDescriptors,
				std::vector<std::vector<cv::KeyPoint>> trainKeypoints,
				std::vector<cv::Ptr<cv::DescriptorMatcher> >& matchers, std::vector<cv::DMatch>& matches)
{

	if(m_enableMultipleRatioTest == true)
	{
		matches.clear();

		//マッチングを格納
		std::vector< std::vector<cv::DMatch>> patternMatches(dataSetSize -1);
		std::vector<cv::DMatch> tmpMatches;


		//最近傍点の探索
		for(int i = 0; i < dataSetSize -1 ; i++)
		{
			//knnマッチング
			std::vector< std::vector<cv::DMatch>>  knnMatches;

			// queryとmatcherに保存されている特徴量をknn構造体を用いて最近傍点を検索する.
			matchers[i]->knnMatch(queryDescriptors, knnMatches, 1);

			//_matchesにm_knnMathesの要素をコピー
			for(int l = 0; l < knnMatches.size(); l++)
			{
				if(knnMatches[l].empty() == false)						//マッチングが存在するか
					patternMatches[i].push_back(knnMatches[l][0]);
			}

			//初期化
			knnMatches.clear();

		}

		//クエリ特徴点の数までループ
		for (size_t j = 0; j <patternMatches[0].size(); j++)
		{
			float worstdistance = 0;		//最も悪いマッチングのユークリッド距離(0未満になることはない)
			int worstId=-1;					//最も悪いマッチングのID

			//最も悪いマッチングの探索
			for(size_t k = 0; k < dataSetSize - 1 ; k++)
			{
				if(worstdistance <= patternMatches[k][j].distance)
				{
					worstdistance =  patternMatches[k][j].distance;
					worstId = k;
				}
			}

			for(size_t i=0; i< dataSetSize - 1 ; i++)
			{

				if(worstId != i)
				{
					cv::DMatch& currentMatch   =patternMatches[i][j];
					float distanceRatio = currentMatch.distance / worstdistance;

					// Pass only matches where distance ratio between 
					// nearest matches is greater than 1.5 (distinct criteria)
					if (distanceRatio < minRatio)
					{
						//
						currentMatch.imgIdx = i;
						//当たり値を格納
					    tmpMatches.push_back(currentMatch);
					}
				}
			}
		}
		//幾何学的整合性チェックの下準備
		std::vector< std::vector<cv::DMatch>> correctMatches(dataSetSize -1);	//multipleRatioTestで通過ペアを画像ごとに保存

		for(int i = 0; i < dataSetSize -1; i++)
		{
			for(int j =0; j < tmpMatches.size(); j++)
			{
				if( i== tmpMatches[j].imgIdx)
					correctMatches[i].push_back(tmpMatches[j]);
			}
		}

		for(int i = 0; i < dataSetSize -1; i++)
		{
			//幾何学的整合性チェック
			bool passFlag = geometricConsistencyCheck(queryKeypoints, trainKeypoints[i], correctMatches[i]);
			//幾何学的整合性チェックに通過したもののみ登録する
			if(passFlag == true){
				//要素の移し替え
				for(int k = 0; k < correctMatches[i].size(); k++)
				{
					matches.push_back(correctMatches[i][k]);
				}
			}
		}


	}else
	{
		matches.clear();

		int imgNumber = 0;

		//最近傍点の探索
		for(int i = 0; i < dataSetSize -1 ; i++)
		{
			//knnマッチング
			std::vector< std::vector<cv::DMatch>>  knnMatches;

			// queryとmatcherに保存されている特徴量をknn構造体を用いて最近傍点を検索する.
			matchers[i]->knnMatch(queryDescriptors, knnMatches, 2);

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

					//距離の比が1.5以下の特徴だけ保存
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
}

bool Matching::geometricConsistencyCheck(std::vector<cv::KeyPoint> queryKeypoints, std::vector<cv::KeyPoint> trainKeypoints, std::vector<cv::DMatch>& match)
{
	if(match.size() < 8)
	{
		match.clear();
		return false;
	}
	std::vector<cv::Point2f>  queryPoints, trainPoints; 
	for(int i = 0; i < match.size(); i++)
	{
		queryPoints.push_back(queryKeypoints[match[i].queryIdx].pt);
		trainPoints.push_back(trainKeypoints[match[i].trainIdx].pt);
	}

	//幾何学的整合性チェック
	std::vector<unsigned char> inliersMask(queryPoints.size() );

	//幾何学的整合性チェックによって当たり値を抽出
	cv::findHomography( queryPoints, trainPoints, CV_FM_RANSAC, 10, inliersMask);

	std::vector<cv::DMatch> inliers;
	for(size_t i =0 ; i < inliersMask.size(); i++)
	{
		if(inliersMask[i])
			inliers.push_back(match[i]);
	}

	match.swap(inliers);
	return true;
}