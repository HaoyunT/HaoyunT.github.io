import pandas as pd

class Preprocessor:
    '''
    1. 特征增强（数据清洗）
        1. 缺失值填充
        1. 冗余删除
        1. 错误修正
        1. 归一化
    '''
    def __init__(self, 
                 file_name, 
                 file_dir="../public/赛题/2025_MCM-ICM_Problems/2025_Problem_C_Data/", 
                 encoding="Windows-1252", 
                 test_mode=True):
        self.dir = file_dir + file_name
        self.encoding = encoding
        self.test_mode = test_mode
        self.file_name = file_name
        self.csv_file = self.__read__(self.encoding)

        self.__preprocess_info__()
        
    def __read__(self, encoding="Windows-1252"):
        return pd.read_csv(self.dir, encoding=encoding)
    
    def __preprocess_info__(self):
        na_num = self.csv_file.isnull().sum().sum()
        # print("%s null value number: %d" % (self.file_name, na_num))
        
    def save_csv(self):
        self.csv_file.to_csv(self.file_name + "_preprocessed.csv", index=False, encoding="utf-8")

class Athlete(Preprocessor):
    def __init__(self, test_mode=True):
        super().__init__(file_name="summerOly_athletes.csv", test_mode=test_mode)
        self.__preprocess__()

    def __preprocess__(self):
        self.csv_file = self.csv_file[ self.csv_file['Year'] != 1906 ]
        self.csv_file['Sport'] = self.csv_file['Sport'].replace('3x3 Basketball, Basketball','3x3 Basketball')
        self.csv_file['Sport'] = self.csv_file['Sport'].replace('Marathon Swimming, Swimming','Marathon Swimming')

class Host(Preprocessor):
    def __init__(self, test_mode=True):
        # 调用父类的构造函数，传入文件名和编码方式
        super().__init__(file_name="summerOly_hosts.csv", 
                         encoding="utf-8", # Fuck host
                         test_mode=test_mode)
        # 调用预处理函数
        self.__preprocess__()
    
    def __preprocess__(self):
        self.csv_file.drop(index=[5, 11, 12], inplace=True)
        self.csv_file['Host'] = self.csv_file['Host'].apply(lambda x: x.split(',')[1][1:])
        self.csv_file.iloc[28, 1] = 'Japan'
    
class Medal(Preprocessor):
    def __init__(self, test_mode=True):
        super().__init__(file_name="summerOly_medal_counts.csv", test_mode=test_mode)
        
class Program(Preprocessor):
    def __init__(self, test_mode=True):
        super().__init__(file_name="summerOly_programs.csv", test_mode=test_mode)
        self.__preprocess__()

    def __preprocess__(self):
        self.csv_file.loc[69, '1924'] = 0
        self.csv_file.loc[70, '1924'] = 0
        self.csv_file = self.csv_file.iloc[:-3, :]
        
        self.csv_file.loc[39, 'Code'] = "JDP"
        self.csv_file.loc[47, 'Code'] = "ROQ"

        self.csv_file.loc[37, 'Code'] = "IndoorHBL"
        self.csv_file.loc[38, 'Code'] = "FieldHBL"

        self.csv_file.loc[42, 'Code'] = "SixesLAX"
        self.csv_file.loc[43, 'Code'] = "FieldLAX"
        
        self.csv_file.loc[49, '1896'] = 0
        self.csv_file.loc[52, '1896'] = 0
       
        self.csv_file.fillna(0, inplace=True)
        self.csv_file.replace('•', 0, inplace=True)

        self.csv_file.drop(columns=['Discipline', 'Sports Governing Body'], inplace=True)

class RawDataset(Preprocessor):
    def __init__(self, test_mode=True):
        super().__init__(file_name="full_features.csv", 
                         file_dir="./mid_data/",
                         encoding="utf-8", 
                         test_mode=test_mode)
        
        self.__preprocess__()

    def __preprocess__(self):
        pass

class InputFuture(Preprocessor):
    def __init__(self, test_mode=True):
        super().__init__(file_name="input_future.csv", 
                         file_dir="./mid_data/",
                         encoding="utf-8", 
                         test_mode=test_mode)
        
        self.__preprocess__()

    def __preprocess__(self):
        pass