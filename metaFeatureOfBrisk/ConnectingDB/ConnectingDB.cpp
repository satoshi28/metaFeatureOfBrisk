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
	System::String^ strConn = "Provider=Microsoft.ACE.OLEDB.12.0;Data Source=C:\\Users\\satoshi\\Documents\\Visual Studio 2012\\DB\\SURF_BRISK.accdb";
	System::String^ strLocation ="SELECT tb_���P�[�V�������.[ID],tb_���P�[�V�������.[latitude], tb_���P�[�V�������.[longitude] FROM tb_���P�[�V�������";
	System::String^ strDesc ="SELECT * FROM tb_������";
	System::String^ strKeypoints ="SELECT * FROM tb_�����_";

	OleDbConnection^ conn = gcnew OleDbConnection(strConn);
	conn->Open();



	//�g�����U�N�V�����̊J�n
	OleDbTransaction^ transaction = conn->BeginTransaction(System::Data::IsolationLevel::ReadCommitted);
	try{
		//���P�[�V�������̍X�V�p
		OleDbDataAdapter^ locationAdapter = gcnew OleDbDataAdapter(strLocation, strConn);
		OleDbCommandBuilder^ builder = gcnew OleDbCommandBuilder(locationAdapter);

		//�����ʂ̍X�V�p
		OleDbDataAdapter^ descAdapter = gcnew OleDbDataAdapter(strDesc, strConn);
		OleDbCommandBuilder^ descBuilder = gcnew OleDbCommandBuilder(descAdapter);

		//�����_�̍X�V�p
		OleDbDataAdapter^ keypointsAdapter = gcnew OleDbDataAdapter(strKeypoints, strConn);
		OleDbCommandBuilder^ keypointsBuilder = gcnew OleDbCommandBuilder(keypointsAdapter);


		//tb_�����ʂ��X�V
		updateLocationTable(locationAdapter, patterns);
		//tb_�����ʂ��X�V
		updateDescTable(descAdapter, patterns);
		//tb_�����_���X�V
		updateKeypointTable(keypointsAdapter, patterns);

		//�g�����U�N�V�������R�~�b�g���܂��B
        transaction->Commit();

		return 0;
	}
	catch(System::Exception^){
		//�g�����U�N�V�����̃��[���o�b�N
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
	adapter->RowUpdated += gcnew OleDbRowUpdatedEventHandler(OnRowUpdated);
	adapter->Update(dataChanges);

	//�I�[�g�i���o�[��Patterns�ɕۑ�
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

void ConnectingDB::updateKeypointTable(OleDbDataAdapter^ adapter, std::vector<Pattern> patterns)
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
