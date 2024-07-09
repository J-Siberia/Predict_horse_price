import csv
import os
import requests
import pandas as pd
from urllib.parse import urljoin
import re
from bs4 import NavigableString
from lxml import html
from lxml import etree
from selenium import webdriver
from selenium.webdriver.chrome import service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 検索結果のURLのベース (ページ番号が変わる部分を含む)
base_url = 'https://carrotclub.net/horse/bosyuba-list.asp'

# 画像を保存するフォルダ
image_folder = 'carrot'
os.makedirs(image_folder, exist_ok=True)

# CSVファイルの準備
csv_file = 'data_carrot.csv'
csv_data = []

# ページにアクセス
response = requests.get(base_url)
response.raise_for_status()

# BeautifulSoupでパース
soup = BeautifulSoup(response.content, 'html.parser')

# class="tblType1"の<table>要素を取得
table = soup.find('table', class_='tblType1')

# 取得したURLを格納するリスト
urls = []

# テーブル内のすべての<tr>要素を取得
rows = table.find_all('tr')
# 各<tr>要素の5番目の<td>からURLを取得
for row in rows:
    cells = row.find_all('td')
    if len(cells) >= 5:
        # 5番目の<td>要素を取得
        fifth_cell = cells[4]
        # <a>タグを探してhref属性を取得
        a_tag = fifth_cell.find('a')
        if a_tag and 'href' in a_tag.attrs:
            urls.append('https://carrotclub.net/horse/' + a_tag['href'])

# Selenium WebDriverのインスタンスを作成（Chromeを使用する場合）
CHROMEDRIVER = "www"
chrome_service = service.Service(executable_path=CHROMEDRIVER)
driver = webdriver.Chrome(service=chrome_service)

try:
    for url in urls:
        try:
            print("Accessing " + url + "...")
            # 対象のURLを開く
            driver.get(url)

            # 画像のXpathを指定
            xpath = '/html/body/main/article/section/section[2]/ul/li[2]/a/img'

            # ページが完全に読み込まれるのを待つ
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )

            # 画像を取得
            img = driver.find_element(By.XPATH, xpath)
            src = img.get_attribute('src')

            # 画像の名前をURLから取得
            img_name = os.path.basename(urlparse(src).path)      
            # 画像をダウンロード
            img_data = requests.get(src).content

            # 画像の保存
            with open(os.path.join(image_folder, img_name), 'wb') as handler:
                handler.write(img_data)

            # 馬名を取得
            name = None
            elements = driver.find_element(By.XPATH, "/html/body/main/article/section/div/div[2]/h1")
            if elements:
                name = (re.search(r'[^\x00-\x7F]+.*', elements.text)).group(0).strip()
                elements = None

            # 父を取得
            sire = None
            elements = driver.find_element(By.XPATH, "/html/body/main/article/section/div/div[3]/a[1]")
            if elements:
                sire = (elements.text).replace('*', '')
                elements = None

            # 母の父を取得
            bms = None
            elements = driver.find_element(By.XPATH, "/html/body/main/article/section/div/div[3]/a[2]")
            if elements:    
                bms = (elements.text).replace('*', '')
                elements = None

            # 募集総額を取得
            price = None
            elements = driver.find_element(By.XPATH, "/html/body/main/article/section/section[2]/ul/li[1]/p[1]")
            if elements:
                investment_match = re.search(r'募集総額(\d{1,3}(,\d{3})*)万', elements.text)
                if investment_match:
                    investment_amount = investment_match.group(1).strip()
                    price = int(re.sub(r'\D', '', investment_amount)) * 10000 
                    elements = None

            # 性別と毛色を取得
            sex = None
            elements = driver.find_element(By.XPATH, "/html/body/main/article/section/div/div[2]/p")
            if elements:
                sex = (elements.text).split()[0]
                if(sex=="牡"):
                    sex="オス"
                elif(sex=="牝"):
                    sex="メス"
            hair_color = None
            if elements:
                hair_color = (elements.text).split()[1]

            # CSV用データに追加
            csv_data.append({'name': name, 'image': '/silk/'+img_name, 'sire': sire, 'bms': bms, 'price': price, 'sex': sex, 'hair color': hair_color})

        except Exception as e:
                print(f"Error processing {url}: {e}")
                continue  # 次のURLに移る

finally:
    # ブラウザを閉じる
    driver.quit()

# DataFrameを作成してCSVに保存
df = pd.DataFrame(csv_data)
df.to_csv(csv_file, index=False)

print(f"データが {csv_file} に保存されました。")
