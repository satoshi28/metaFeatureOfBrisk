#ifndef EXIF_GPS
#define EXIF_GPS

#include "../Pattern.h"

#using <System.Drawing.dll>
using namespace System;
using namespace System::Drawing::Imaging;

/*
* 画像のタグからGPS座標を取得するクラス
*/

class exifGps
{
public:
	exifGps();
	~exifGps();

	/*
	* @brief 入力されたパスの画像からGPS座標を取得する
	* @param[in] filename 絶対パス
	* return Gps
	* @note 取得できなかった場合は緯度・経度を-1にして返す
	*/
	void GetGps(const char* filename, Gps& gps);
private:
	/*
	* @brief PropertyItemから緯度or経度のdouble型で取得する
	* @param[in] propItemRef
	* @param[in] propItem
	* return 緯度 or 経度
	*/
	double exifGpsToDouble(PropertyItem^ propItemRef, PropertyItem^ propItem);
};

#endif