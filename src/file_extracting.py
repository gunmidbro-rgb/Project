import zipfile
import tarfile
import pandas as pd
import os

class DataExtract:
    def __init__(self,base_folder: str = 'data/raw'):
        self.base_folder = base_folder
        self.zip_path = os.path.join(base_folder,'amazon_review_polarity_csv.tgz.zip')
        self.tgz_path = os.path.join(base_folder,'amazon_review_polarity_csv.tgz')
        self.extracted_folder = os.path.join(base_folder,'amazon_review_polarity_csv')
        self.source_csv = os.path.join(self.extracted_folder, 'test.csv')
        self.output_sample_csv = os.path.join(self.extracted_folder, 'sample.csv')

    def extract_zip(self):
        print('Extracting zip')
        with zipfile.ZipFile(self.zip_path, 'r') as zip:
            inner_file_name = 'amazon_review_polarity_csv.tgz'
            zip.extract(inner_file_name, path=self.base_folder)
        print('Successful')

    def extract_tgz(self):
        print('Extracting tgz')
        with tarfile.open(self.tgz_path, 'r:gz') as tgz:
            tgz.extractall(path=self.base_folder)
        print('Successful')

    def get_sample(self,nums_row:int):
        print(f'Get sample {nums_row} rows')
        sample = pd.read_csv(self.source_csv,nrows=nums_row,header=None)
        sample.to_csv(self.output_sample_csv,index=False,header=None)

    def runpipeline(self):
        print('Start run pipeline')
        self.extract_zip()
        self.extract_tgz()
        self.get_sample(nums_row=50000)
        print('Successful')


if __name__ == '__main__':
    extractor = DataExtract()
    extractor.runpipeline()