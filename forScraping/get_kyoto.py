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
base_url = 'https://kyoto-tc.jp/ch'

# 画像を保存するフォルダ
image_folder = 'kyoto'
os.makedirs(image_folder, exist_ok=True)

# CSVファイルの準備
csv_file = 'data_kyoto.csv'
csv_data = []

# 取得したURLを格納するリスト
urls = []

#対象のURLを開く
driver.get(base_url)
time.sleep(6) 
# テーブルの行数を特定するために行要素を取得
rows = driver.find_elements(By.XPATH, '/html/body/section[1]/div[1]/div[2]/ul/li')

# 各行からリンクを取得
for i in range(1, len(rows) + 1):
    xpath = f'/html/body/section[1]/div[1]/div[2]/ul/li[{i}]/div[1]/div[1]/a'
    link_element = driver.find_element(By.XPATH, xpath)
    link = link_element.get_attribute('href')
    urls.append(link)

# ブラウザを閉じる
# driver.quit()
#print(urls)

no_img = 0

try:
    for url in urls:
        try:
            print("Accessing " + url + "...")
            # 対象のURLを開く
            driver.get(url)

            # 画像のXpathを指定
            xpath = '/html/body/section[1]/div/div[1]/img'
            
            # ページが完全に読み込まれるのを待つ
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            
            # 画像を取得
            img = driver.find_element(By.XPATH, xpath)
            src = img.get_attribute('src')
            
            # 画像の名前をURLから取得
            img_name = os.path.basename(urlparse(src).path)     
            img_name = str(no_img) + img_name
            no_img += 1 
            # 画像をダウンロード
            img_data = requests.get(src).content
            
            # 画像の保存
            with open(os.path.join(image_folder, img_name), 'wb') as handler:
                handler.write(img_data)

            # 馬名を取得
            name = None
            elements = driver.find_element(By.XPATH, "/html/body/section[1]/div/h3")
            if elements:
                name = elements.text
                print(name)
                elements = None
            else:
                continue

            # 父と母の父を取得
            sire = None
            bms = None
            elements = driver.find_element(By.XPATH, "/html/body/section[1]/div/div[3]/div/table/tbody/tr[2]/td[1]")
            if elements:
                sire = elements.text
                elements = None
                print(sire)
            else:
                continue
            elements = driver.find_element(By.XPATH, "/html/body/section[1]/div/div[3]/div/table/tbody/tr[3]/td[1]")
            if elements:
                bms = elements.text
                elements = None
                print(bms)
            else:
                continue

            # 募集総額を取得
            price = None
            elements = driver.find_element(By.XPATH, "/html/body/section[1]/div/div[2]/div/div[2]/ul/li[2]/span")
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
            elements = driver.find_element(By.XPATH, "/html/body/section[1]/div/div[3]/div/table/tbody/tr[1]/td[1]")
            if elements:
                sex = elements.text
                if(sex=="牡馬"):
                    sex="オス"
                elif(sex=="牝馬"):
                    sex="メス"
                elements = None
                print(sex)
            hair_color = None
            elements = driver.find_element(By.XPATH, "/html/body/section[1]/div/div[3]/div/table/tbody/tr[1]/td[2]")
            if elements:
                hair_color = elements.text
                print(hair_color)
            # CSV用データに追加
            csv_data.append({'name': name, 'image': '/kyoto/'+img_name, 'sire': sire, 'bms': bms, 'price': price, 'sex': sex, 'hair color': hair_color})

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
