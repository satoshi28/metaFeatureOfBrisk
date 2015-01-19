#ifndef EXIF_GPS
#define EXIF_GPS

#include "../Pattern.h"

#using <System.Drawing.dll>
using namespace System;
using namespace System::Drawing::Imaging;

/*
* �摜�̃^�O����GPS���W���擾����N���X
*/

class exifGps
{
public:
	exifGps();
	~exifGps();

	/*
	* @brief ���͂��ꂽ�p�X�̉摜����GPS���W���擾����
	* @param[in] filename ��΃p�X
	* return Gps
	* @note �擾�ł��Ȃ������ꍇ�͈ܓx�E�o�x��-1�ɂ��ĕԂ�
	*/
	void GetGps(const char* filename, Gps& gps);
private:
	/*
	* @brief PropertyItem����ܓxor�o�x��double�^�Ŏ擾����
	* @param[in] propItemRef
	* @param[in] propItem
	* return �ܓx or �o�x
	*/
	double exifGpsToDouble(PropertyItem^ propItemRef, PropertyItem^ propItem);
};

#endif