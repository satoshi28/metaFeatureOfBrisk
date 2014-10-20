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
	
	/* Pattern内部の情報をデータベースに追加する */
	int updateDB(std::vector<Pattern>& patterns);

	/* DataTableをcsvファイルに保存する */
	void SaveToCSV(System::Data::DataTable^ dt, System::String^ fileName, bool hasHeader, 
		System::String^ separator, System::String^ quote, System::String^ replace);

private:
	/* DBのtb_ロケーション情報を更新 */
	void updateLocationTable(MySqlDataAdapter^ adapter, std::vector<Pattern>& patterns);

	/* DBのtb_特徴量を更新 */
	void updateDescTable(MySqlDataAdapter^ adapter, std::vector<Pattern> patterns);

	/* DBのtb_特徴点を更新 */
	void ConnectingDB::updateKeypointTable(MySqlDataAdapter^ adapter, std::vector<Pattern> patterns);

	/* DB更新時に発生するイベント */
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