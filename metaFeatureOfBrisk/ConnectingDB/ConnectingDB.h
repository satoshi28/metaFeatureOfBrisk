#ifndef CONNECTING_DB
#define CONNECTING_DB

////////////////////////////////////////////////////////////////////
using namespace System::Diagnostics;
using namespace System::Data::OleDb;

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
	void updateLocationTable(OleDbDataAdapter^ adapter, std::vector<Pattern>& patterns);

	/* DB��tb_�����ʂ��X�V */
	void updateDescTable(OleDbDataAdapter^ adapter, std::vector<Pattern> patterns);

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
};



#endif