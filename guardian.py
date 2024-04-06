# Collect guardian news API data
import requests
import csv

# Kept key out as public repo - security reasons
KEY = ''
url = (f'https://content.guardianapis.com/search?from-date=2015-01-01&to-date=2016-12-12&page-size=200&q=bitcoin&api-key={KEY}')

response = requests.get(url)

if response.status_code == 200:
    json_data = response.json()
    articles = json_data.get('response', {}).get('results', [])
    csv_file_path = 'news_data.csv'

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ['webPublicationDate', 'webTitle']
        csv_writer.writerow(header)

        for article in articles:
            row = [article.get('webPublicationDate', ''), article.get('webTitle', '')]
            csv_writer.writerow(row)

    print(f"CSV file '{csv_file_path}' created successfully.")
else:
    print(f"Error: {response.status_code}")
