#ifndef EXIF_GPS
#define EXIF_GPS

#include "../Pattern.h"

#using <System.Drawing.dll>

using namespace System;
using namespace System::Drawing::Imaging;

class exifGps
{
public:
	exifGps();
	~exifGps();
	
	void GetGps(const char* filename, Gps& gps);
private:
	double exifGpsToDouble(PropertyItem^ propItemRef, PropertyItem^ propItem);

};

#endif