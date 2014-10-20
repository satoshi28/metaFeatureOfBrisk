#ifndef CONNECTING_DB
#define CONNECTING_DB

////////////////////////////////////////////////////////////////////
using namespace System::Diagnostics;
//using namespace System::Data::OleDb;
//using namespace System::Data::SQLite;
using namespace MySql::Data::MySqlClient;


#include "../Pattern.h"

class ConnectingDB
{
public:
	ConnectingDB();
	~ConnectingDB();
	
	/* Pattern�����̏����f�[�^�x�[�X�ɒǉ����� */
	int updateDB(std::vector<Pattern>& patterns);

	/* DataTable��csv�t�@�C���ɕۑ����� */
	void SaveToCSV(System::Data::DataTable^ dt, System::String^ fileName, bool hasHeader, 
		System::String^ separator, System::String^ quote, System::String^ replace);

private:
	/* DB��tb_���P�[�V���������X�V */
	void updateLocationTable(MySqlDataAdapter^ adapter, std::vector<Pattern>& patterns);

	/* DB��tb_�����ʂ��X�V */
	void updateDescTable(MySqlDataAdapter^ adapter, std::vector<Pattern> patterns);

	/* DB��tb_�����_���X�V */
	void ConnectingDB::updateKeypointTable(MySqlDataAdapter^ adapter, std::vector<Pattern> patterns);

	/* DB�X�V���ɔ�������C�x���g */
	static void  OnRowUpdated(System::Object^ sender, MySqlRowUpdatedEventArgs^ e)
	{
		if (e->Status == System::Data::UpdateStatus::Continue && e->StatementType == System::Data::StatementType::Insert)
	    {
			
			MySqlCommand^ cmdNewID= gcnew MySqlCommand("SELECT LAST_INSERT_ID()", e->Command->Connection);
			System::Object^ o;
			o = cmdNewID->ExecuteScalar();
			e->Row["ID"] = System::Int32::Parse( o->ToString() );
			e->Status = System::Data::UpdateStatus::SkipCurrentRow;
	
		}
	};

private:
	


};



#endif