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
	
	/* Pattern内部の情報をデータベースに追加する */
	int updateDB(std::vector<Pattern>& patterns);

	/* DataTableをcsvファイルに保存する */
	void SaveToCSV(System::Data::DataTable^ dt, System::String^ fileName, bool hasHeader, 
		System::String^ separator, System::String^ quote, System::String^ replace);

private:
	/* DBのtb_ロケーション情報を更新 */
	void updateLocationTable(OleDbDataAdapter^ adapter, std::vector<Pattern>& patterns);

	/* DBのtb_特徴量を更新 */
	void updateDescTable(OleDbDataAdapter^ adapter, std::vector<Pattern> patterns);

	/* DB更新時に発生するイベント */
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