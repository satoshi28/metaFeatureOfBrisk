#include "stdafx.h"
#include "ConnectingDB.h"


ConnectingDB::ConnectingDB()
{
}

ConnectingDB::~ConnectingDB()
{
}


int ConnectingDB::updateDB(std::vector<Pattern>& patterns)
{

	//conect
	System::String^ strConn = "userid=root;password=root;database=objectdatabase;Host=localhost";
	System::String^ strLocation ="SELECT * FROM tb_location";
	System::String^ strDesc ="SELECT * FROM tb_descriptors";
	System::String^ strKeypoints ="SELECT * FROM tb_keypoints";
	System::String^ strInfo ="SELECT * FROM tb_information";

	MySqlConnection^ conn = gcnew MySqlConnection(strConn);
	conn->Open();



	//�g�����U�N�V�����̊J�n
	MySqlTransaction^ transaction = conn->BeginTransaction(System::Data::IsolationLevel::ReadCommitted);
	try{
		//���P�[�V�������̍X�V�p
		MySqlDataAdapter^ locationAdapter = gcnew MySqlDataAdapter(strLocation, strConn);
		MySqlCommandBuilder^ locationBuilder = gcnew MySqlCommandBuilder(locationAdapter);

		//�����ʂ̍X�V�p
		MySqlDataAdapter^ descAdapter = gcnew MySqlDataAdapter(strDesc, strConn);
		MySqlCommandBuilder^ descBuilder = gcnew MySqlCommandBuilder(descAdapter);

		//�����_�̍X�V�p
		MySqlDataAdapter^ keypointsAdapter = gcnew MySqlDataAdapter(strKeypoints, strConn);
		MySqlCommandBuilder^ keypointsBuilder = gcnew MySqlCommandBuilder(keypointsAdapter);

		//���̍X�V�p
		MySqlDataAdapter^ infoAdapter = gcnew MySqlDataAdapter(strInfo, strConn);
		MySqlCommandBuilder^ infoBuilder = gcnew MySqlCommandBuilder(infoAdapter);

		/*
		//tb_�����ʂ��X�V
		updateLocationTable(locationAdapter, patterns);
		//tb_�����ʂ��X�V
		updateDescTable(descAdapter, patterns);
		//tb_�����_���X�V
		updateKeypointTable(keypointsAdapter, patterns);
		*/
		//tb_information���X�V
		updateInfoTable(infoAdapter, patterns);
		//�g�����U�N�V�������R�~�b�g���܂��B
        transaction->Commit();

		return 0;
	}
	catch(MySqlException^ ex){
		//�g�����U�N�V�����̃��[���o�b�N
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
	//�擾�p�f�[�^�e�[�u��
	System::Data::DataTable^ table = gcnew System::Data::DataTable("data");
	//�X�V�p�f�[�^�e�[�u��
	System::Data::DataTable^ dataChanges = gcnew System::Data::DataTable("dataChanges");

	adapter->Fill(table);

	//�f�[�^�̒ǉ�
	System::Data::DataRow^ datarow;
	for(int i = 0; i < patterns.size() ; i++)
	{
		int ad =  patterns.size();
		//�V�����s�̍쐬
		datarow = table->NewRow();
	
		//�V�����ǉ�����s�ɑ΂���,�񖼂��w�肵�ăf�[�^��ǉ�����
		datarow["latitude"]= patterns[i].gps.latitude;
		datarow["longitude"]= patterns[i].gps.longitude;

		//�s�̒ǉ�
		table->Rows->Add(datarow);
	}
	dataChanges = table->GetChanges();

	//DB�̍X�V
	//�I�[�g�i���o�[�擾�p��update�C�x���g��L����
	adapter->RowUpdated += gcnew MySqlRowUpdatedEventHandler(OnRowUpdated);
	adapter->Update(dataChanges);

	//�I�[�g�i���o�[��Patterns�ɕۑ�
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



	//�V�����s�̌^���쐬
	System::Data::DataRow^ descRow;
	for(int i = 0; i < patterns.size() ; i++)
	{
		for(int j = 0; j < patterns[i].descriptors.rows ; j++)
		{
			//�V�����s�̍쐬
			descRow = table->NewRow();

			for(int k = 0; k < patterns[i].descriptors.cols ; k++)//64
			{
				descRow["ID"] = patterns[i].numberOfDB;
				descRow[(k+1).ToString()] = patterns[i].descriptors.at<unsigned char>(j,k);

			}
			//�s�̒ǉ�
			table->Rows->Add(descRow);
		}
	}
	//DB�X�V
	adapter->Update(table);
}

void ConnectingDB::updateKeypointTable(MySqlDataAdapter^ adapter, std::vector<Pattern> patterns)
{
	System::Data::DataTable^ table = gcnew System::Data::DataTable("keypoints");
	adapter->Fill(table);



	//�V�����s�̌^���쐬
	System::Data::DataRow^ row;
	for(int i = 0; i < patterns.size() ; i++)
	{
		for(int j = 0; j < patterns[i].keypoints.size() ; j++)
		{
			//�V�����s�̍쐬
			row = table->NewRow();

			row["ID"] = patterns[i].numberOfDB;
			row["px"] = patterns[i].keypoints[j].pt.x;
			row["py"] = patterns[i].keypoints[j].pt.y;
			//�s�̒ǉ�
			table->Rows->Add(row);
		}
	}
	//DB�X�V
	adapter->Update(table);
}


void ConnectingDB::updateInfoTable(MySqlDataAdapter^ adapter, std::vector<Pattern>& patterns)
{
	System::Data::DataTable^ table = gcnew System::Data::DataTable("info");
	adapter->Fill(table);

	//�V�����s�̌^���쐬
	System::Data::DataRow^ row;
	for(int i = 0; i < patterns.size() ; i++)
	{
		//�V�����s�̍쐬
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

		//�s�̒ǉ�
		table->Rows->Add(row);
		
	}
	//DB�X�V
	adapter->Update(table);
}


void ConnectingDB::SaveToCSV(System::Data::DataTable^ dt, System::String^ fileName, bool hasHeader, System::String^ separator, System::String^ quote, System::String^ replace)
{

  int rows = dt->Rows->Count;
  
  int cols = dt->Columns->Count;

  System::String^ text;

  //�ۑ��p�̃t�@�C�����J���B�㏑�����[�h�ŁB

  System::IO::StreamWriter^ writer = gcnew System::IO::StreamWriter(fileName, false, System::Text::Encoding::GetEncoding("shift_jis"));
  
  //�J��������ۑ����邩

  if (hasHeader)
  {
      //�J��������ۑ�����ꍇ
      for (int i = 0; i < cols; i++)
      {
          //�J���������擾
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

  //�f�[�^�̕ۑ�����
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
  //�X�g���[�������
  writer->Close();
}
