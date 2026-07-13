import zipfile
import tarfile
import pandas as pd

def extract_zip(zip_path,output):
    with zipfile.ZipFile(zip_path, 'r') as zip:
        inner_file_name = 'amazon_review_polarity_csv.tgz'
        zip.extract(inner_file_name, path=output)
    
def extract_tgz(tgz_path,output):
    with tarfile.open(tgz_path, 'r:gz') as tgz:
        tgz.extractall(path=output)

def get_sample(file_path,output):
    sample = pd.read_csv(file_path,nrows=50000,header=None)
    sample.to_csv(output,index=False,header=None)

def main():
    zip_path = 'data/raw/amazon_review_polarity_csv.tgz.zip'
    extract_to_path = 'data/raw/'
    tgz_path = 'data/raw/amazon_review_polarity_csv.tgz'
    extract_zip(zip_path,extract_to_path)
    extract_tgz(tgz_path,extract_to_path)
    test = 'data/raw/amazon_review_polarity_csv/test.csv'
    get_sample(test,'data/raw/amazon_review_polarity_csv/sample.csv')

if __name__ == '__main__':
    main()