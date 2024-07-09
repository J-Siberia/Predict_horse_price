import csv
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import re

# 検索結果のURLのベース (ページ番号が変わる部分を含む)
base_url = 'https://www.silkhorseclub.jp/horse_info/boshu/list'

# すべてのリンクを格納するリスト
all_links = []

# 画像を保存するフォルダ
image_folder = 'silk'
os.makedirs(image_folder, exist_ok=True)

# CSVファイルの準備
csv_file = 'data_silk.csv'
csv_data = []

for page_num in range(1, 2):
    # ページのURLを生成
    #url = base_url.format(page_num)
    url = base_url
    print(f'Fetching page {page_num}: {url}')
    
    # ページを取得
    response = requests.get(url)
    if response.status_code != 200:
        print(f'Failed to fetch page {page_num}')
        continue
    
    # BeautifulSoupで解析
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 検索結果リンクを抽出 (適切なタグとクラスを指定)
    search_results = soup.find_all('a', class_='c-btn--link--blue--horseListDetail') # クラス名は適宜変更
    
    for result in search_results:
        link = result.get('href')
        link = "https://www.silkhorseclub.jp" + link
        all_links.append(link)

for url in all_links:
    try:
        # ページにアクセス
        response = requests.get(url)
        response.raise_for_status()
        
        # BeautifulSoupでパース
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # ページ内容をテキストとして取得
        page_content = soup.get_text(separator='\n', strip=True)
        
        # 特定のdivクラスの下にある画像を探す
        img_name = None
        target_div = soup.find('div', class_='data1')
        if target_div:
            img_tag = target_div.find('img')
            if img_tag:
                img_url = urljoin(url, img_tag.get('src'))
                img_name = os.path.basename(img_url)
                img_path = os.path.join(image_folder, img_name)
                
                # 画像をダウンロードして保存
                img_response = requests.get(img_url)
                img_response.raise_for_status()
                with open(img_path, 'wb') as img_file:
                    img_file.write(img_response.content)
        
        # 特定のh1タグのクラス"labelHeading"を持つ要素を取得
        label_heading = soup.find('h1', class_='labelHeading')
        print("Getting "+label_heading.get_text(strip=True)+"'s Information")
        if label_heading:
            name = label_heading.get_text(strip=True)
        else:
            name = None

        # <div class="moreBox spMore">の下にあるdl要素を取得
        more_box = soup.find('div', class_='moreBox spMore')
        sire_dt = "父"
        bms_dt = "母の父"
        price_dt = "募集総額 / 一口出資額"
        sex_hair_dt = "性別 / 毛色"
        sire = None
        bms = None
        price = None
        sex = None
        hair_color = None
        
        if more_box:
            dls = more_box.find_all('dl')
            for dl in dls:
                dt = dl.find('dt').get_text(strip=True)
                if dt == sire_dt:
                    sire = dl.find('dd').get_text(strip=True)
                elif dt == bms_dt:
                    bms = dl.find('dd').get_text(strip=True)
                elif dt == price_dt:
                    before_slash = dl.find('dd').get_text(strip=True).split(' / ')[0]
                    price = int(re.sub(r'\D', '', before_slash)) * 10000
                elif dt == sex_hair_dt:
                    sex_hair = dl.find('dd').get_text(strip=True).split(' / ')
                    if len(sex_hair) == 2:
                        sex = sex_hair[0]
                        if sex == "牡":
                            sex = "オス"
                        hair_color = sex_hair[1]
            
        # CSV用データに追加
        csv_data.append({'name': name, 'image': '/silk/'+img_name, 'sire': sire, 'bms': bms, 'price': price, 'sex': sex, 'hair color': hair_color})
    
    except Exception as e:
        print(f"Error processing {url}: {e}")

# DataFrameを作成してCSVに保存
df = pd.DataFrame(csv_data)
df.to_csv(csv_file, index=False)

print(f"データが {csv_file} に保存されました。")
