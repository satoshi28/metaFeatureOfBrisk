#ifndef _CONSTANT
#define _CONSTANT

const std::string detectorName = "SURF";		//特徴点検出アルゴリズム
const std::string extractorName = "BRISK";	//特徴量抽出アルゴリズム
const std::string matcherName = "BruteForce-Hamming";		//マッチングアルゴリズム

const int budget = 1200;					//メタ特徴量のサイズ
const float minRatio = 0.8f;			// To avoid NaN's when best match has zero distance we will use inversed ratio. 
#endif