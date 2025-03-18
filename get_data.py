import requests
from bs4 import BeautifulSoup
import time

year = 2011

def process(year):
    time.sleep(1.0)
    print(f"processing {year}")

    base_url = f"https://www.timeanddate.com/eclipse/{year}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url=base_url, headers=headers)
    content = response.text

    soup = BeautifulSoup(content, "html.parser")
    links = soup.find_all("a")

    for link in reversed(links):
        lik = link.get("href")
        if lik.startswith(f"/eclipse/solar/{year}"):
            print(lik)
            new_url = f"https://www.timeanddate.com/{lik}"
            new_response = requests.get(url=new_url, headers=headers)
            new_soup = BeautifulSoup(new_response.text, "html.parser")
            tables = new_soup.find_all("table")
            extracted_tables = []

            for table in tables:
                table_headers = []
                header_row = table.find("tr")
                if header_row:
                    table_headers = [th.text.strip() for th in header_row.find_all("th")]
                if not "Eclipse Stages Worldwide" in table_headers: break

                rows = []
                for row in table.find_all("tr")[1:]: 
                    cells = [td.text.strip() for td in row.find_all("td")]
                    if cells:  
                        rows.append(cells)

                with open('./traing_data/start_time.txt', 'a') as f:
                    f.write(str(year) + ' ' + rows[0][1] + '\n')

                with open('./traing_data/end_time.txt', 'a') as f:
                    f.write(str(year) + ' ' + rows[-1][1] + '\n')

                print(rows[0][1])
                print(rows[-1][1])

            for table in extracted_tables:
                print(table)

for i in range(2011, 1975, -1):
    process(i)

