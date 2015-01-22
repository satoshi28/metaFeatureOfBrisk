#include "stdafx.h"
#include "ConnectingDB.h"

ConnectingDB::ConnectingDB()
{
}

ConnectingDB::~ConnectingDB()
{
}

#ifdef _OLEDB

int ConnectingDB::updateDB(std::vector<Pattern>& patterns)
{

	//conect
	System::String^ strConn = "Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\\Users\\satoshi\\Documents\\Visual Studio 2012\\study_db\\zubud_200.accdb";
	System::String^ strLocation ="SELECT tb_ロケーション情報.[ID], tb_ロケーション情報.[name_no], tb_ロケーション情報.[latitude], tb_ロケーション情報.[longitude] FROM tb_ロケーション情報";
	System::String^ strDesc ="SELECT * FROM tb_特徴量";
	System::String^ strKeypoints ="SELECT * FROM tb_特徴点";

	OleDbConnection^ conn = gcnew OleDbConnection(strConn);
	conn->Open();



	//トランザクションの開始
	OleDbTransaction^ transaction = conn->BeginTransaction(System::Data::IsolationLevel::ReadCommitted);
	try{
		//ロケーション情報の更新用
		OleDbDataAdapter^ locationAdapter = gcnew OleDbDataAdapter(strLocation, strConn);
		OleDbCommandBuilder^ builder = gcnew OleDbCommandBuilder(locationAdapter);

		//特徴量の更新用
		OleDbDataAdapter^ descAdapter = gcnew OleDbDataAdapter(strDesc, strConn);
		OleDbCommandBuilder^ descBuilder = gcnew OleDbCommandBuilder(descAdapter);

		//特徴点の更新用
		OleDbDataAdapter^ keypointsAdapter = gcnew OleDbDataAdapter(strKeypoints, strConn);
		OleDbCommandBuilder^ keypointsBuilder = gcnew OleDbCommandBuilder(keypointsAdapter);


		//tb_特徴量を更新
		updateLocationTable(locationAdapter, patterns);
		//tb_特徴量を更新
		updateDescTable(descAdapter, patterns);
		//tb_特徴点を更新
		updateKeypointTable(keypointsAdapter, patterns);

		//トランザクションをコミットします。
        transaction->Commit();

		return 0;
	}
	catch(System::Exception^ e){
		//トランザクションのロールバック
		transaction->Rollback();

		return -1;
	}
	finally
	{
		conn->Close();
	}

/*
#ifdef _DEBUG
	System::String^ quote = "";
    System::String^ separator = ",";
    System::String^ replace = "";

	System::String^ filename = "sample.csv";
	SaveToCSV(table, filename, true,separator ,quote, replace);
#endif //_DEBUG
	
*/
}

void ConnectingDB::updateLocationTable(OleDbDataAdapter^ adapter, std::vector<Pattern>& patterns)
{
	//取得用データテーブル
	System::Data::DataTable^ table = gcnew System::Data::DataTable("data");
	//更新用データテーブル
	System::Data::DataTable^ dataChanges = gcnew System::Data::DataTable("dataChanges");

	adapter->Fill(table);

	//データの追加
	System::Data::DataRow^ datarow;
	for(int i = 0; i < patterns.size() ; i++)
	{
		int ad =  patterns.size();
		//新しい行の作成
		datarow = table->NewRow();

		//新しく追加する行に対して,列名を指定してデータを追加する
		datarow["latitude"]= patterns[i].gps.latitude;
		datarow["longitude"]= patterns[i].gps.longitude;

		//行の追加
		table->Rows->Add(datarow);
	}
	dataChanges = table->GetChanges();

	//DBの更新
	//オートナンバー取得用にupdateイベントを有効化
	adapter->RowUpdated += gcnew OleDbRowUpdatedEventHandler(OnRowUpdated);
	adapter->Update(dataChanges);

	//オートナンバーをPatternsに保存
	for(int i = 0; i < patterns.size() ; i++)
	{
		int num = System::Convert::ToInt32( dataChanges->Rows[i][0]->ToString() );
		patterns[i].numberOfDB= num;
	}
}

void ConnectingDB::updateDescTable(OleDbDataAdapter^ adapter, std::vector<Pattern> patterns)
{
	System::Data::DataTable^ table = gcnew System::Data::DataTable("descriptors");
	adapter->Fill(table);



	//新しい行の型を作成
	System::Data::DataRow^ descRow;
	for(int i = 0; i < patterns.size() ; i++)
	{
		for(int j = 0; j < patterns[i].descriptors.rows ; j++)
		{
			//新しい行の作成
			descRow = table->NewRow();

			for(int k = 0; k < patterns[i].descriptors.cols ; k++)//64
			{
				descRow["ID"] = patterns[i].numberOfDB;
				descRow["featureID"] = j;
				descRow[(k+1).ToString()] = patterns[i].descriptors.at<unsigned char>(j,k);

			}
			//行の追加
			table->Rows->Add(descRow);
		}
	}
	//DB更新
	adapter->Update(table);
}

void ConnectingDB::updateKeypointTable(OleDbDataAdapter^ adapter, std::vector<Pattern> patterns)
{
	System::Data::DataTable^ table = gcnew System::Data::DataTable("keypoints");
	adapter->Fill(table);



	//新しい行の型を作成
	System::Data::DataRow^ row;
	for(int i = 0; i < patterns.size() ; i++)
	{
		for(int j = 0; j < patterns[i].keypoints.size() ; j++)
		{
			//新しい行の作成
			row = table->NewRow();

			row["ID"] = patterns[i].numberOfDB;
			row["featureID"] = j;
			row["px"] = patterns[i].keypoints[j].pt.x;
			row["py"] = patterns[i].keypoints[j].pt.y;
			//行の追加
			table->Rows->Add(row);
		}
	}
	//DB更新
	adapter->Update(table);
}

#endif

void ConnectingDB::SaveToCSV(System::Data::DataTable^ dt, System::String^ fileName, bool hasHeader, System::String^ separator, System::String^ quote, System::String^ replace)
{

  int rows = dt->Rows->Count;
  
  int cols = dt->Columns->Count;

  System::String^ text;

  //保存用のファイルを開く。上書きモードで。

  System::IO::StreamWriter^ writer = gcnew System::IO::StreamWriter(fileName, false, System::Text::Encoding::GetEncoding("shift_jis"));
  
  //カラム名を保存するか

  if (hasHeader)
  {
      //カラム名を保存する場合
      for (int i = 0; i < cols; i++)
      {
          //カラム名を取得
          if (quote != "")
          {	  
              text = dt->Columns[i]->ColumnName->Replace(quote, replace);
          }
          else
          {
              text = dt->Columns[i]->ColumnName;
          }

          if (i != cols - 1)
          {
              writer->Write(quote + text + quote + separator);
          }
          else
          {
              writer->WriteLine(quote + text + quote);
          }
      }
  }

  //データの保存処理
  for (int i = 0; i < rows; i++)
  {
      for (int j = 0; j < cols; j++)
      {
          if (quote != "")
          {
              text = dt->Rows[i][j]->ToString()->Replace(quote, replace);
          }
          else
          {
              text = dt->Rows[i][j]->ToString();
          }

          if (j != cols - 1)
          {
              writer->Write(quote + text + quote + separator);
          }
          else
          {
              writer->WriteLine(quote + text + quote);
          }
      }
  }
  //ストリームを閉じる
  writer->Close();
}



#ifdef _MYSQL
int ConnectingDB::updateDB(std::vector<Pattern>& patterns)
{

	//conect
	System::String^ strConn = "userid=root;password=root;database=objectdatabase2;Host=localhost";
	System::String^ strLocation ="SELECT * FROM tb_location";
	System::String^ strDesc ="SELECT * FROM tb_descriptors";
	System::String^ strInfo ="SELECT * FROM tb_information";

	MySqlConnection^ conn = gcnew MySqlConnection(strConn);
	conn->Open();



	//トランザクションの開始
	MySqlTransaction^ transaction = conn->BeginTransaction(System::Data::IsolationLevel::ReadCommitted);
	try{
		//ロケーション情報の更新用
		MySqlDataAdapter^ locationAdapter = gcnew MySqlDataAdapter(strLocation, strConn);
		MySqlCommandBuilder^ locationBuilder = gcnew MySqlCommandBuilder(locationAdapter);

		//特徴量の更新用
		MySqlDataAdapter^ descAdapter = gcnew MySqlDataAdapter(strDesc, strConn);
		MySqlCommandBuilder^ descBuilder = gcnew MySqlCommandBuilder(descAdapter);

		//情報の更新用
		MySqlDataAdapter^ infoAdapter = gcnew MySqlDataAdapter(strInfo, strConn);
		MySqlCommandBuilder^ infoBuilder = gcnew MySqlCommandBuilder(infoAdapter);

		
		//tb_特徴量を更新
		updateLocationTable(locationAdapter, patterns);
		//tb_特徴量を更新
		updateDescTable(descAdapter, patterns);
		
		//tb_informationを更新
		updateInfoTable(infoAdapter, patterns);
		//トランザクションをコミットします。
        transaction->Commit();

		return 0;
	}
	catch(MySqlException^ ex){
		//トランザクションのロールバック
		transaction->Rollback();
		System::Console::WriteLine(ex->Message);
		return -1;
	}
	finally
	{
		conn->Close();
	}

/*
#ifdef _DEBUG
	System::String^ quote = "";
    System::String^ separator = ",";
    System::String^ replace = "";

	System::String^ filename = "sample.csv";
	SaveToCSV(table, filename, true,separator ,quote, replace);
#endif //_DEBUG
	
*/
}

void ConnectingDB::updateLocationTable(MySqlDataAdapter^ adapter, std::vector<Pattern>& patterns)
{
	//取得用データテーブル
	System::Data::DataTable^ table = gcnew System::Data::DataTable("data");
	//更新用データテーブル
	System::Data::DataTable^ dataChanges = gcnew System::Data::DataTable("dataChanges");

	adapter->Fill(table);

	//データの追加
	System::Data::DataRow^ datarow;
	for(int i = 0; i < patterns.size() ; i++)
	{
		int ad =  patterns.size();
		//新しい行の作成
		datarow = table->NewRow();
	
		//新しく追加する行に対して,列名を指定してデータを追加する
		datarow["latitude"]= patterns[i].gps.latitude;
		datarow["longitude"]= patterns[i].gps.longitude;

		//行の追加
		table->Rows->Add(datarow);
	}
	dataChanges = table->GetChanges();

	//DBの更新
	//オートナンバー取得用にupdateイベントを有効化
	adapter->RowUpdated += gcnew MySqlRowUpdatedEventHandler(OnRowUpdated);
	adapter->Update(dataChanges);

	//オートナンバーをPatternsに保存
	for(int i = 0; i < patterns.size() ; i++)
	{
		int num = System::Convert::ToInt32( dataChanges->Rows[i][0]->ToString() );
		patterns[i].numberOfDB= num;
	}
}

void ConnectingDB::updateDescTable(MySqlDataAdapter^ adapter, std::vector<Pattern> patterns)
{
	System::Data::DataTable^ table = gcnew System::Data::DataTable("descriptors");
	adapter->Fill(table);



	//新しい行の型を作成
	System::Data::DataRow^ descRow;
	for(int i = 0; i < patterns.size() ; i++)
	{
		for(int j = 0; j < patterns[i].descriptors.rows ; j++)
		{
			//新しい行の作成
			descRow = table->NewRow();

			for(int k = 0; k < patterns[i].descriptors.cols ; k++)//64
			{
				descRow["ID"] = patterns[i].numberOfDB;
				descRow["featureID"] = j;
				descRow[(k+1).ToString()] = patterns[i].descriptors.at<unsigned char>(j,k);

			}
			//行の追加
			table->Rows->Add(descRow);
		}
	}
	//DB更新
	adapter->Update(table);
}

void ConnectingDB::updateInfoTable(MySqlDataAdapter^ adapter, std::vector<Pattern>& patterns)
{
	System::Data::DataTable^ table = gcnew System::Data::DataTable("info");
	adapter->Fill(table);

	//新しい行の型を作成
	System::Data::DataRow^ row;
	for(int i = 0; i < patterns.size() ; i++)
	{
		//新しい行の作成
		row = table->NewRow();
		/*
		unsigned char*  data = patterns[i].image.data;
		int len = strlen((char*)data);
		
		array<unsigned char>^ byteArray = gcnew array<unsigned char>(len);
		// convert native pointer to System::IntPtr with C-Style cast
		System::Runtime::InteropServices::Marshal::Copy((System::IntPtr)data,byteArray, 0, len);
	
		row["ID"] = i+1;
		row["image"] =  byteArray; 
		*/
		//int number = patterns[i].numberOfDB;
		int number = i+1;
		
		std::stringstream ss;
		ss <<number;
		std::string result = "C:\\Apache2.2\\htdocs\\images\\";
		std::string name =  ss.str();
		name += ".jpg";

		result += name;

		cv::imwrite( result, patterns[i].image);

		std::string path = "http://localhost/images/" + name;
		System::String^ str = gcnew System::String(path.c_str());
		row["ID"] = number;
		row["imageID"] = str;

		//行の追加
		table->Rows->Add(row);
		
	}
	//DB更新
	adapter->Update(table);
}
#endif