#ifndef CONNECTING_DB
#define CONNECTING_DB

#define _MYSQL

using namespace System::Diagnostics;
#ifdef _OLEDB
using namespace System::Data::OleDb;
#endif

#ifdef _MYSQL
using namespace MySql::Data::MySqlClient;
#endif


#include "../Pattern.h"

class ConnectingDB
{
public:
	ConnectingDB();
	~ConnectingDB();
	
	/* Pattern�����̏����f�[�^�x�[�X�ɒǉ����� */
	int updateDB(std::vector<Pattern>& patterns, std::string fileName);

	/* DataTable��csv�t�@�C���ɕۑ����� */
	void SaveToCSV(System::Data::DataTable^ dt, System::String^ fileName, bool hasHeader, 
		System::String^ separator, System::String^ quote, System::String^ replace);

private:
	#ifdef _OLEDB
	/* DB��tb_���P�[�V���������X�V */
	void updateLocationTable(OleDbDataAdapter^ adapter, std::vector<Pattern>& patterns);

	/* DB��tb_�����ʂ��X�V */
	void updateDescTable(OleDbDataAdapter^ adapter, std::vector<Pattern> patterns);

	/* DB��tb_�����_���X�V */
	void ConnectingDB::updateKeypointTable(OleDbDataAdapter^ adapter, std::vector<Pattern> patterns);

	/* DB�X�V���ɔ�������C�x���g */
	static void OnRowUpdated(System::Object^ sender, OleDbRowUpdatedEventArgs^ e)
	{
		if (e->Status == System::Data::UpdateStatus::Continue && e->StatementType == System::Data::StatementType::Insert)
	    {
			OleDbCommand^ cmdNewID = gcnew OleDbCommand("SELECT @@IDENTITY", e->Command->Connection);
			int s = (int)cmdNewID->ExecuteScalar();
			e->Row["ID"] = (int)cmdNewID->ExecuteScalar();
			e->Status = System::Data::UpdateStatus::SkipCurrentRow;
	
		}
	};
	#endif

	#ifdef _MYSQL
	/* DB��tb_���P�[�V���������X�V */
	void updateLocationTable(MySqlDataAdapter^ adapter, std::vector<Pattern>& patterns);

	/* DB��tb_�����ʂ��X�V */
	void updateDescTable(MySqlDataAdapter^ adapter, std::vector<Pattern> patterns);

	/* DB��tb_information���X�V */
	void ConnectingDB::updateInfoTable(MySqlDataAdapter^ adapter, std::vector<Pattern>& patterns);

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
	#endif
};



#endif