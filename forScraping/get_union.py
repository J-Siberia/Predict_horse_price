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
base_url = 'https://www.union-oc.co.jp/active/list.jsp#open_LEFT_25'

# 画像を保存するフォルダ
image_folder = 'union'
os.makedirs(image_folder, exist_ok=True)

# CSVファイルの準備
csv_file = 'data_union.csv'
csv_data = []

# 取得したURLを格納するリスト
urls = []

#対象のURLを開く
driver.get(base_url)
time.sleep(6) 
# テーブルの行数を特定するために行要素を取得
rows = driver.find_elements(By.XPATH, '/html/body/div[2]/div[3]/div[5]/table/tbody/tr')

# 各行からリンクを取得
for i in range(1, len(rows) + 1):
    xpath = f'/html/body/div[2]/div[3]/div[5]/table/tbody/tr[{i}]/td[1]/p[1]/a'
    link_element = driver.find_element(By.XPATH, xpath)
    link = link_element.get_attribute('href')
    urls.append(link+'#open_PROFILE')

# ブラウザを閉じる
#driver.quit()

time.sleep(3) 

base_url = 'https://www.union-oc.co.jp/active/list.jsp#open_LEFT_26'
#対象のURLを開く
driver.get(base_url)
time.sleep(4) 
# テーブルの行数を特定するために行要素を取得
rows = driver.find_elements(By.XPATH, '/html/body/div[2]/div[3]/div[6]/table/tbody/tr')

# 各行からリンクを取得
for i in range(1, len(rows) + 1):
    xpath = f'/html/body/div[2]/div[3]/div[6]/table/tbody/tr[{i}]/td[1]/p[1]/a'
    link_element = driver.find_element(By.XPATH, xpath)
    link = link_element.get_attribute('href')
    urls.append(link+'#open_PROFILE')

# ブラウザを閉じる
#driver.quit()

no_img = 0

try:
    for url in urls:
        try:
            print("Accessing " + url + "...")
            # 対象のURLを開く
            driver.get(url)

            # 画像のXpathを指定
            xpath = '/html/body/div[2]/div[3]/div[1]/div[3]/img'

            # ページが完全に読み込まれるのを待つ
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )

            # 画像を取得
            img = driver.find_element(By.XPATH, xpath)
            src = img.get_attribute('src')

            # 画像の名前をURLから取得
            img_name = os.path.basename(urlparse(src).path)     
            img_name = str(no_img)+img_name
            # 画像をダウンロード
            img_data = requests.get(src).content

            # 画像の保存
            with open(os.path.join(image_folder, img_name), 'wb') as handler:
                handler.write(img_data)

            no_img=no_img+1

            # 馬名を取得
            name = None
            elements = driver.find_element(By.XPATH, "/html/body/div[2]/div[3]/div[1]/div[1]")
            if elements:
                #name = (re.search(r'[^\x00-\x7F]+.*', elements.text)).group(0).strip()
                name = elements.text
                elements = None

            # 父を取得
            sire = None
            elements = driver.find_element(By.XPATH, "/html/body/div[2]/div[3]/div[1]/div[4]/table/tbody/tr[1]/td")
            if elements:
                sire = (elements.text).replace('*', '')
                elements = None

            # 母の父を取得
            bms = None
            elements = driver.find_element(By.XPATH, "/html/body/div[2]/div[3]/div[1]/div[4]/table/tbody/tr[3]/td")
            if elements:    
                bms = (elements.text).replace('*', '')
                elements = None

            # 募集総額を取得
            price = None
            elements = driver.find_element(By.XPATH, "/html/body/div[2]/div[3]/div[2]/div[4]")
            if elements:
                investment_match = re.search(r'募集価格 (\d{1,3}(,\d{3})*)万', elements.text)
                if investment_match:
                    investment_amount = investment_match.group(1).strip()
                    price = int(re.sub(r'\D', '', investment_amount)) * 10000 
                    elements = None

            # 性別と毛色を取得
            sex = None
            elements = driver.find_element(By.XPATH, "/html/body/div[2]/div[3]/div[1]/div[4]/table/tbody/tr[7]/td[1]")
            if elements:
                sex = elements.text
                if(sex=="牡"):
                    sex="オス"
                elif(sex=="牝"):
                    sex="メス"
            hair_color = None
            elements = driver.find_element(By.XPATH, "/html/body/div[2]/div[3]/div[1]/div[4]/table/tbody/tr[7]/td[2]") 
            if elements:
                hair_color = elements.text
            # CSV用データに追加
            csv_data.append({'name': name, 'image': '/union/'+img_name, 'sire': sire, 'bms': bms, 'price': price, 'sex': sex, 'hair color': hair_color})

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
