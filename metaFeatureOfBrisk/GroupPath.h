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
	�t�H���_���̑S�摜�̃p�X���擾��,���炩���ߌ��߂������K���Ɋ�Â��ĕ��ނ��A�O���[�v���Ƃɂ܂Ƃ߂ĕԂ�
	�����K��:����.���̕��̂��������_.�g���q(apple.view001.jpg)
	�摜���ꖇ�Ȃ�1��Ԃ�
	�摜������������A�O���[�v���ł�����0��Ԃ�
	*/
	int getPath(String^ folderPath, std::vector< std::vector<std::string> >& groupedFile)
	{
		//���ތ�̃t�@�C��
		cliext::vector< cliext::vector<String^> > sortFile;

		/* step1: �w��t�H���_���̉摜path��file�ɕۑ� */
		array<String^>^ file = Directory::GetFiles( folderPath );
	
		//�w��t�H���_�ɉ摜��1�������Ȃ������ꍇ��1��Ԃ�
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
	
		/* step2: �p�X����摜�p�X�𕪗ނ��� */
		/* �����K��: ~.~~.jpg
		** ~(�ŏ��́u.�v�܂ł̃p�X)�𔻕ʗp�p�X�Ɩ�������
		** ���ʗp�p�X������Ȃ瓯�����̂̉摜�ł���
		*/
	
		//�������镶����̔z��
		array<String^>^ SepString ={"."}; 
		cliext::vector<String^> tmpFile;	//�p�X���ޏ����p�ꎞ�t�@�C��
		array<String^>^ preString;			//�ЂƂO�̔��ʗp�p�X��ۑ�

	
		preString = file[0]->Split(SepString,StringSplitOptions::RemoveEmptyEntries);
		for(int i = 0; i < file->Length; i++)
		{
			//���݂̉摜�̔��ʗp�p�X���擾
			array<String^>^ curString = file[i]->Split(SepString,StringSplitOptions::RemoveEmptyEntries);

			//���ʗp�p�X���ЂƂO�ƈႤ�Ȃ�A����ł��̃O���[�v�͕��ފ���
			if(0 != String::Compare(preString[0], curString[0] , true))
			{
				//�O���[�v���Ƃɕۑ�
				sortFile.push_back(tmpFile );
				//�ꎞ�t�@�C���̏�����
				tmpFile.clear();
				//���ʗp�p�X�̍X�V
				preString = file[i]->Split(SepString,StringSplitOptions::RemoveEmptyEntries);
			}
			//��΃p�X��ۑ�
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
	System::String^ -> std::string �̕ϊ�
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