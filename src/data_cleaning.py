import pandas as pd

class TextCleaner:
    
    def rename_and_map(self, df:pd.DataFrame ) -> pd.DataFrame:
        data = df.copy()
        rename_df = data.rename(columns = {0:'sentiment',1:'title',2:'text'})
        rename_df.replace({'sentiment':{1:'negative',2:'positive'}},inplace=True)
        return rename_df

    def convert_dtypes(self,df:pd.DataFrame):
        df['sentiment'] = df['sentiment'].astype('category')
        return df
    
    def processing(self,df):
        data = self.rename_and_map(df)
        data = self.convert_dtypes(data)
        return data