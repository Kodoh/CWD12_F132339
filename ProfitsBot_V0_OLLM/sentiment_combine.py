# Script to combine reddit data + news data

import pandas as pd

csv_news = 'news_data.csv'
csv_combined_data = '/ds_builder/combined_data.csv'
newcsv_data = '/ds_builder/combined_data.csv'

df_combined_data = pd.read_csv(csv_combined_data)
df_combined_data = df_combined_data.drop(columns=['id'])
df_news = pd.read_csv(csv_news)
df_news = df_news.rename(columns={'webTitle': 'title'})
df_combined_data = pd.concat([df_combined_data, df_news], ignore_index=True)

df_combined_data.to_csv(newcsv_data, index=False)

print(f"\nRows from {csv_news} added to {csv_combined_data} successfully.")
