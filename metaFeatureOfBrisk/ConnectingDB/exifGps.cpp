#include "stdafx.h"
#include "exifGps.h"


exifGps::exifGps()
{
}

exifGps::~exifGps()
{
}

void exifGps::GetGps(const char* _filename, Gps& gps)
{
	System::String^ filename = gcnew System::String(_filename);
	System::Drawing::Bitmap^ image = gcnew System::Drawing::Bitmap(filename);
	
	try{
		//Latitude
		System::Drawing::Imaging::PropertyItem^ propItemLatRef = image->GetPropertyItem(1);
		System::Drawing::Imaging::PropertyItem^ propItemLat = image->GetPropertyItem(2);
		gps.latitude = exifGpsToDouble(propItemLatRef, propItemLat);

		//Longitude
		System::Drawing::Imaging::PropertyItem^ propItemLongRef = image->GetPropertyItem(3);
		System::Drawing::Imaging::PropertyItem^ propItemLong = image->GetPropertyItem(4);
		gps.longitude = exifGpsToDouble(propItemLongRef, propItemLong);
	}catch(ArgumentException^)
	{
		gps.latitude = -1;
		gps.longitude = -1;
	}

}

double exifGps::exifGpsToDouble(PropertyItem^ propItemRef, PropertyItem^ propItem)
{
    double degreesNumerator = BitConverter::ToUInt32(propItem->Value, 0);
    double degreesDenominator = BitConverter::ToUInt32(propItem->Value, 4);
    double degrees = degreesNumerator / (float)degreesDenominator;

    double minutesNumerator = BitConverter::ToUInt32(propItem->Value, 8);
    double minutesDenominator = BitConverter::ToUInt32(propItem->Value, 12);
    double minutes = minutesNumerator / (float)minutesDenominator;

    double secondsNumerator = BitConverter::ToUInt32(propItem->Value, 16);
    double secondsDenominator = BitConverter::ToUInt32(propItem->Value, 20);
    double seconds = secondsNumerator / (float)secondsDenominator;

    double coorditate = degrees + (minutes / 60.0f) + (seconds / 3600.0f);

	String^ gpsRef = Text::Encoding::ASCII->GetString( propItemRef->Value );
    if (gpsRef == "S" || gpsRef == "W")	//N, S, E, or W
        coorditate = 0 - coorditate;
    return coorditate;
}