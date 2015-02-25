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
	
	/* Pattern内部の情報をデータベースに追加する */
	int updateDB(std::vector<Pattern>& patterns, std::string fileName);

	/* DataTableをcsvファイルに保存する */
	void SaveToCSV(System::Data::DataTable^ dt, System::String^ fileName, bool hasHeader, 
		System::String^ separator, System::String^ quote, System::String^ replace);

private:
	#ifdef _OLEDB
	/* DBのtb_ロケーション情報を更新 */
	void updateLocationTable(OleDbDataAdapter^ adapter, std::vector<Pattern>& patterns);

	/* DBのtb_特徴量を更新 */
	void updateDescTable(OleDbDataAdapter^ adapter, std::vector<Pattern> patterns);

	/* DBのtb_特徴点を更新 */
	void ConnectingDB::updateKeypointTable(OleDbDataAdapter^ adapter, std::vector<Pattern> patterns);

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
	#endif

	#ifdef _MYSQL
	/* DBのtb_ロケーション情報を更新 */
	void updateLocationTable(MySqlDataAdapter^ adapter, std::vector<Pattern>& patterns);

	/* DBのtb_特徴量を更新 */
	void updateDescTable(MySqlDataAdapter^ adapter, std::vector<Pattern> patterns);

	/* DBのtb_informationを更新 */
	void ConnectingDB::updateInfoTable(MySqlDataAdapter^ adapter, std::vector<Pattern>& patterns);

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
	#endif
};



#endif