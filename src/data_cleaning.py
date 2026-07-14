import pandas as pd
import contractions

class TextCleaner:

    def rename_and_map(self, df:pd.DataFrame ) -> pd.DataFrame:
        data = df.copy()
        rename_df = data.rename(columns = {0:'sentiment',1:'title',2:'text'})
        rename_df.replace({'sentiment':{1:'negative',2:'positive'}},inplace=True)
        return rename_df

    def convert_dtypes(self,df:pd.DataFrame):
        df['sentiment'] = df['sentiment'].astype('category')
        return df

    def clean_text(self,text:str):
        if type(text) is not str:
            return ''
        cleaned_text = contractions.fix(text)
        cleaned_text = cleaned_text.lower().strip()
        return cleaned_text
    
    def processing(self,df):
        data = self.rename_and_map(df)
        data = self.convert_dtypes(data)
        data['title'] = data['title'].apply(self.clean_text)
        data['text'] = data['text'].apply(self.clean_text)
        return data