# Script to get Cryptomanic news

import requests
import csv

def saveCSV(url,coin):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        
        if "results" in data:
            articles = data["results"]
            
            csv_filename = f"/Users/jakeanderson/Documents/Uni/23COC257/news_data.csv"
            
            with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                writer.writerow(["Title", "Votes", "Published Date"])
                
                for article in articles:
                    title = article.get("title", "N/A")
                    votes = article.get("votes", "N/A")
                    published_date = article.get("published_at", "N/A")
                    
                    writer.writerow([title, votes, published_date])
            
            print(f"Saved data to {csv_filename}")
        else:
            print(f"No articles available for {coin} at the moment.")
    else:
        print(f"Failed to fetch data from the API. Status Code: {response.status_code}")

# Key not saved for security reasons

KEY = ''

saveCSV(f"https://cryptopanic.com/api/v1/posts/?auth_token={KEY}&currencies=BTC","BTC")
saveCSV(f"https://cryptopanic.com/api/v1/posts/?auth_token={KEY}&currencies=ETH","ETH")