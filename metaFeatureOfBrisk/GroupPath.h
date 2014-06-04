#ifndef GROUP_PATH
#define GROUP_PATH

#include <cliext/vector>

#include <vector>
#include <cstdlib>

using namespace System;
using namespace System::IO;

class GroupPath
{
public:
	GroupPath();
	~GroupPath();

public:
	/*
	フォルダ内の全画像のパスを取得し,あらかじめ決めた命名規則に基づいて分類し、グループごとにまとめて返す
	命名規則:物体.その物体を見た視点.拡張子(apple.view001.jpg)
	画像が一枚なら1を返す
	画像が複数枚あり、グループ化できたら0を返す
	*/
	int getPath(String^ folderPath, std::vector< std::vector<std::string> >& groupedFile)
	{
		//分類後のファイル
		cliext::vector< cliext::vector<String^> > sortFile;

		/* step1: 指定フォルダ下の画像pathをfileに保存 */
		array<String^>^ file = Directory::GetFiles( folderPath );
	
		//指定フォルダに画像が1枚しかなかった場合は1を返す
		if(file->Length == 1){
			std::vector<std::string> tmpFolder;
			String^ tmp = file[0];
			
			//String^->std::string
			std::string filePath;
			MarshalString(tmp, filePath);

			tmpFolder.push_back(filePath);

			groupedFile.push_back(tmpFolder);
			
			return 1;
		}
	
		/* step2: パスから画像パスを分類する */
		/* 命名規則: ~.~~.jpg
		** ~(最初の「.」までのパス)を判別用パスと命名する
		** 判別用パスが同一なら同じ物体の画像である
		*/
	
		//分割する文字列の配列
		array<String^>^ SepString ={"."}; 
		cliext::vector<String^> tmpFile;	//パス分類処理用一時ファイル
		array<String^>^ preString;			//ひとつ前の判別用パスを保存

	
		preString = file[0]->Split(SepString,StringSplitOptions::RemoveEmptyEntries);
		for(int i = 0; i < file->Length; i++)
		{
			//現在の画像の判別用パスを取得
			array<String^>^ curString = file[i]->Split(SepString,StringSplitOptions::RemoveEmptyEntries);

			//判別用パスがひとつ前と違うなら、それでそのグループは分類完了
			if(0 != String::Compare(preString[0], curString[0] , true))
			{
				//グループごとに保存
				sortFile.push_back(tmpFile );
				//一時ファイルの初期化
				tmpFile.clear();
				//判別用パスの更新
				preString = file[i]->Split(SepString,StringSplitOptions::RemoveEmptyEntries);
			}
			//絶対パスを保存
			tmpFile.push_back(file[i]);
		}

		sortFile.push_back(tmpFile);
		//
		std::vector<std::string> tmpFolder;
		for(int i = 0; i < sortFile.size(); i++)
		{
			for(int j = 0; j < sortFile[i].size(); j++)
			{
				String^ tmp = sortFile[i].at(j);
			
				//String^->std::string
				std::string file;
				MarshalString(tmp, file);
				tmpFolder.push_back(file);
			}
			groupedFile.push_back(tmpFolder);
			tmpFolder.clear();

		}
	
		return 0;
	}

private:
	/*
	System::String^ -> std::string の変換
	*/
	void MarshalString ( String ^ s, std::string& os ) 
	{
	using namespace Runtime::InteropServices;
	const char* chars = 
	   (const char*)(Marshal::StringToHGlobalAnsi(s)).ToPointer();
	os = chars;
	Marshal::FreeHGlobal(IntPtr((void*)chars));
	}

};

GroupPath::GroupPath()
{
}

GroupPath::~GroupPath()
{
}

#endif