import os
import time
import pandas as pd
from urllib.parse import urljoin
import requests
import re
from selenium import webdriver
from selenium.webdriver.chrome import service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Selenium WebDriverのインスタンスを作成（Chromeを使用する場合）
CHROMEDRIVER = "www"
chrome_service = service.Service(executable_path=CHROMEDRIVER)
driver = webdriver.Chrome(service=chrome_service)

# 検索結果のURLのベース (ページ番号が変わる部分を含む)
base_url = 'https://www.normandyoc.com/collect/list.aspx'

# 画像を保存するフォルダ
image_folder = 'normandy'
os.makedirs(image_folder, exist_ok=True)

# CSVファイルの準備
csv_file = 'data_normandy.csv'
csv_data = []

# 取得したURLを格納するリスト
urls = []

#対象のURLを開く
driver.get(base_url)
time.sleep(6) 
# テーブルの行数を特定するために行要素を取得
rows = driver.find_elements(By.XPATH, '/html/body/div[2]/div[3]/div[2]/div[5]/table/tbody/tr')

# 各行からリンクを取得
for i in range(2, len(rows) + 1):
    xpath = f'/html/body/div[2]/div[3]/div[2]/div[5]/table/tbody/tr[{i}]/td[3]/a[1]'
    link_element = driver.find_element(By.XPATH, xpath)
    link = link_element.get_attribute('href')
    urls.append(link)

# ブラウザを閉じる
# driver.quit()
#print(urls)

try:
    for url in urls:
        try:
            print("Accessing " + url + "...")
            # 対象のURLを開く
            driver.get(url)

            # 画像のXpathを指定
            xpath = '/html/body/div[2]/div[3]/div[2]/div/div[4]/a/img'
            
            # ページが完全に読み込まれるのを待つ
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            
            # 画像を取得
            img = driver.find_element(By.XPATH, xpath)
            src = img.get_attribute('src')
            
            # 画像の名前をURLから取得
            img_name = os.path.basename(urlparse(src).path)     
            img_name = img_name
            # 画像をダウンロード
            img_data = requests.get(src).content
            
            # 画像の保存
            with open(os.path.join(image_folder, img_name), 'wb') as handler:
                handler.write(img_data)

            # 馬名を取得
            name = None
            elements = driver.find_element(By.XPATH, "/html/body/div[2]/div[3]/div[2]/div/div[2]/table/tbody/tr[1]/td/b")
            if elements:
                #name = (re.search(r'[^\x00-\x7F]+.*', elements.text)).group(0).strip()
                name = elements.text
                print(name)
                elements = None
            else:
                continue

            # 父と母の父を取得
            sire = None
            bms = None
            elements = driver.find_element(By.XPATH, "/html/body/div[2]/div[3]/div[2]/div/div[1]/p[2]")
            first_pattern = r'父：(.*?) ×'
            second_pattern = r'母の父：(.*?)）'
            first_match = re.search(first_pattern, elements.text)
            second_match = re.search(second_pattern, elements.text)
            if first_match:
                first_result = first_match.group(1)
                sire = first_result
                print(sire)
            else:
                continue
            if second_match:
                second_result = second_match.group(1)
                bms = second_result
                print(bms)
            else:
                continue

            # 募集総額を取得
            price = None
            elements = driver.find_element(By.XPATH, "/html/body/div[2]/div[3]/div[2]/div/div[1]/table/tbody/tr[1]/td[2]")
            if elements:
                # 「,」を削除
                text_without_comma = elements.text.replace(",", "")
                # 「万円」を削除
                text_cleaned = text_without_comma.replace("万円", "")
                if text_cleaned:
                    investment_amount = text_cleaned
                    price = int(re.sub(r'\D', '', investment_amount)) * 10000 
                    print(price)
                    elements = None

            # 性別と毛色を取得
            sex = None
            elements = driver.find_element(By.XPATH, "/html/body/div[2]/div[3]/div[2]/div/div[1]/p[1]/span[1]")
            # 2行目を抽出
            lines = elements.text.split('\n')
            second_line = lines[1]
            parts = second_line.split()
            horse_type = parts[0]
            color = parts[1]
            if horse_type:
                sex = horse_type
                if(sex=="牡馬"):
                    sex="オス"
                elif(sex=="牝馬"):
                    sex="メス"
                print(sex)
            if color:
                hair_color = color+"毛"
                print(hair_color)
            # CSV用データに追加
            csv_data.append({'name': name, 'image': '/normandy/'+img_name, 'sire': sire, 'bms': bms, 'price': price, 'sex': sex, 'hair color': hair_color})

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
