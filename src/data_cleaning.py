import pandas as pd

class TextCleaner:
    
    def processing(self, df:pd.DataFrame ) -> pd.DataFrame:
        data = df.copy()
        rename_df = data.rename(columns = {0:'reviews',1:'title',2:'text'})
        rename_df.replace({'reviews':{1:'negative',2:'positive'}},inplace=True)
        return rename_df
    