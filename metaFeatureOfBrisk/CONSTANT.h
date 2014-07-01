#ifndef _CONSTANT
#define _CONSTANT

const std::string detectorName = "SURF";		//特徴点検出アルゴリズム
const std::string extractorName = "BRISK";	//特徴量抽出アルゴリズム
const std::string matcherName = "BruteForce-Hamming";		//マッチングアルゴリズム

const int budget = 200;					//メタ特徴量のサイズ
const bool enableMultipleRatioTest = false;	//複数比判定法を用いるか
const float minRatio = 0.8f;			// To avoid NaN's when best match has zero distance we will use inversed ratio. 
#endif