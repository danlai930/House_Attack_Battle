#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Dan<dan@gis.tw>' # 2023/10/23


import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import geopandas as gpd
import twd97
import os
import re
from tqdm import tqdm, trange
import datetime


external_data_path = 'House_Attack_Battle/external_data/'
distanceRange = [100, 200, 500, 1000, 2000, 5000]

myConcernedExtData = {
    '金融機構基本資料.csv':'金融機構基本資料',
    '火車站點資料.csv':'火車站點資料',
    '公車站點資料.csv':'公車站點資料',
    '腳踏車站點資料.csv':'腳踏車站點資料',
    '郵局據點資料.csv':'郵局據點資料',
    '捷運站點資料.csv':'捷運站點資料', # no 縣市 column
    '國中基本資料.csv':'國中基本資料',
    '醫療機構基本資料.csv':'型態別',
    '國小基本資料.csv':'國小基本資料',
    '高中基本資料.csv':'',
    'ATM資料.csv':'',
    '大學基本資料.csv':'',
    '便利商店.csv':'公司名稱'
}

醫療機構基本資料 = [
    ['中醫診所','中醫一般診所','中醫專科診所',],
    ['西醫診所','西醫專科診所','西醫醫務室',],
    ['牙醫診所','牙醫一般診所','牙醫專科診所','牙醫醫院',],
    ['醫院','中醫醫院','專科醫院','慢性醫院','綜合醫院',],
    ['其他醫療機構','捐血中心','捐血站','病理中心','精神科教學醫院','精神科醫院','衛生所',]
]

cities = [
    ['台北市'], ['新北市'], ['桃園市'], ['基隆市'], ['新竹市'],  ['新竹縣'], 
    ['苗栗縣'], ['台中市'], 
    ['彰化縣', '雲林縣'],
    ['嘉義市', '嘉義縣'], 
    ['台南市'], ['高雄市'], ['屏東縣'],
    ['宜蘭縣'], ['花蓮縣'], 
    ['金門縣']
]


def get_distanceRange(distance):
    interval = -1
    for dis in distanceRange:
        if distance <= dis:
            interval = dis
            break
    return interval
    

def get_external_data(file_name='便利商店.csv'):

    df_ext = pd.read_csv(external_data_path + file_name)
    gdf = gpd.GeoDataFrame(
        df_ext, geometry=gpd.points_from_xy(df_ext.lng, df_ext.lat), crs={'init':'epsg:4326'}
    )
    gdf.to_crs(epsg=3310, inplace=True)
    return gdf


def print_info(dataset, col_idx):
    print(dataset[dataset.columns[:col_idx]].info())

    obj = (dataset.dtypes == 'object')
    object_cols = list(obj[obj].index)
    print("Categorical variables:",len(object_cols))
     
    int_ = (dataset.dtypes == 'int')
    num_cols = list(int_[int_].index)
    print("Integer variables:",len(num_cols))
     
    fl = (dataset.dtypes == 'float')
    fl_cols = list(fl[fl].index)
    print("Float variables:",len(fl_cols))
    

def get_zipcode(x, df_zipcode):
    x['路名'] = x['路名'].replace('東州', '東洲')
    zipcode = df_zipcode[(df_zipcode['縣市']==x['縣市']) & (df_zipcode['路名']==x['路名'])]
    if len(zipcode) >= 1:
        return zipcode['郵遞區號'].values[0]
    else:
        return x['郵遞區號']
        
        
def merge_external_data(dataset_df):
    '''
    金融機構基本資料.csv: 11751/11751 [09:10<00:00, 21.35it/s] <br>

    火車站點資料.csv ['火車站點資料_100', '火車站點資料_200', '火車站點資料_500', '火車站點資料_1000', '火車站點資料_2000', '火車站點資料_5000']
    火車站點資料.csv: 11751/11751 [00:33<00:00, 355.60it/s] 

    公車站點資料.csv ['公車站點資料_100', '公車站點資料_200', '公車站點資料_500', '公車站點資料_1000', '公車站點資料_2000', '公車站點資料_5000']
    公車站點資料.csv: 11751/11751 [3:34:57<00:00,  1.10s/it]

    腳踏車站點資料.csv ['腳踏車站點資料_100', '腳踏車站點資料_200', '腳踏車站點資料_500', '腳踏車站點資料_1000', '腳踏車站點資料_2000', '腳踏車站點資料_5000']
    腳踏車站點資料.csv: 11751/11751 [14:48<00:00, 13.22it/s]

    郵局據點資料.csv ['郵局據點資料_100', '郵局據點資料_200', '郵局據點資料_500', '郵局據點資料_1000', '郵局據點資料_2000', '郵局據點資料_5000']
    郵局據點資料.csv: 11751/11751 [01:58<00:00, 98.99it/s]

    捷運站點資料.csv ['捷運站點資料_100', '捷運站點資料_200', '捷運站點資料_500', '捷運站點資料_1000', '捷運站點資料_2000', '捷運站點資料_5000']
    捷運站點資料.csv: 11751/11751 [01:52<00:00, 104.69it/s]

    國中基本資料.csv ['國中基本資料_100', '國中基本資料_200', '國中基本資料_500', '國中基本資料_1000', '國中基本資料_2000', '國中基本資料_5000']
    國中基本資料.csv: 11751/11751 [01:21<00:00, 143.50it/s]

    醫療機構基本資料.csv ['中醫一般診所_100', ...
    醫療機構基本資料.csv: 11751/11751 [1:30:43<00:00,  2.16it/s]

    國小基本資料.csv ['國小基本資料_100', '國小基本資料_200', '國小基本資料_500', '國小基本資料_1000', '國小基本資料_2000', '國小基本資料_5000']
    國小基本資料.csv: 11751/11751 [02:15<00:00, 86.80it/s]

    便利商店.csv ['全聯實業股份有限公司_100',...
    便利商店.csv: 11751/11751 [4:19:03<00:00,  1.32s/it]
    '''
    #
    dataset_df['lat'] = dataset_df.apply(lambda x: twd97.towgs84(x['橫坐標'], x['縱坐標'])[0], axis=1)
    dataset_df['lng'] = dataset_df.apply(lambda x: twd97.towgs84(x['橫坐標'], x['縱坐標'])[1], axis=1)
    dataset_df = gpd.GeoDataFrame(
        dataset_df, geometry=gpd.points_from_xy(dataset_df.lng, dataset_df.lat), crs={'init':'epsg:4326'}
    )
    dataset_df.to_crs(epsg=3310, inplace=True)
    #
    external_data = list(myConcernedExtData.keys())

    for idx_extData in range(len(external_data)):
        #
        file_name = external_data[idx_extData]
        df_extData = get_external_data(file_name)
        #
        if file_name == myConcernedExtData[file_name]+'.csv' :
            newColumns = [myConcernedExtData[file_name]+'_'+str(distance) for distance in distanceRange]
        elif myConcernedExtData[file_name] != '':
            newColumns = []
            for col in df_extData[myConcernedExtData[file_name]].unique():
                newColumns += [col+'_'+str(distance) for distance in distanceRange]
        else:
            continue
        print(file_name, newColumns)
        # 給初始值0
        dataset_df[newColumns] = 0
        # 產生縣市欄位
        if file_name=='金融機構基本資料.csv':
            df_extData['縣市'] = df_extData['地址'].apply(lambda x: x[:3])
        
        elif file_name == '火車站點資料.csv':
            df_extData['縣市'] = df_extData['站點地址'].apply(lambda x: re.sub(r'[0-9]', '', x)[:3].replace('臺','台'))    
          
        elif file_name == '郵局據點資料.csv':
            df_extData['縣市'] = df_extData['局址'].apply(lambda x: x[:3])
          
        elif file_name == '國中基本資料.csv':
            df_extData['縣市'] = df_extData['縣市名稱'].apply(lambda x: x.replace('臺','台'))
    
        elif file_name == '國小基本資料.csv':
            df_extData['縣市'] = df_extData['縣市名稱'].apply(lambda x: x.replace('臺','台'))
          
        elif file_name == '醫療機構基本資料.csv':
            df_extData['縣市'] = df_extData['地址'].apply(lambda x: x[:3])
        #
        for i, row in tqdm(dataset_df.iterrows(), desc=file_name, total=dataset_df.shape[0]):
    
            # 只算同縣市資料
            gdf = df_extData[df_extData['縣市']==row['縣市']].copy().reset_index(drop=True) if '縣市' in df_extData.columns else df_extData.copy()
            # print(row['lat'],',',row['lng'])
            # 計算distanceRange中的數量
            for idx_gdf in range(len(gdf)):
                distance = row.geometry.distance(gdf.geometry.iloc[idx_gdf])
                interval = get_distanceRange(distance)
                column = myConcernedExtData[file_name]+'_'+str(interval) if len(newColumns)==len(distanceRange) else gdf.iloc[idx_gdf][myConcernedExtData[file_name]]+'_'+str(interval)
                # print(gdf.iloc[idx_gdf]['縣市'], gdf.iloc[idx_gdf]['金融機構名稱'], gdf.iloc[idx_gdf]['lat'], ',', gdf.iloc[idx_gdf]['lng'], distance, interval)
                if interval != -1:
                    dataset_df.loc[i, column] += 1
                    
    return dataset_df


def merge_and_drop_columns(dataset_df):
    for cols in 醫療機構基本資料:
        for distance in distanceRange:
            cols_to_merge = [col+'_'+str(distance) for col in cols]
            dataset_df[cols_to_merge[0]] = dataset_df[cols_to_merge].sum(axis=1)
            dataset_df = dataset_df.drop(columns=cols_to_merge[1:])
    # dataset_df = dataset_df[[col for col in dataset_df.columns if '_5000' not in col]]
    # 將經緯度轉成bins
    featrues = {
        'longtitude': dataset_df['lng'].to_list(),
        'latitude': dataset_df['lat'].to_list()
    }
    
    lng = tf.feature_column.numeric_column('longtitude')
    lat = tf.feature_column.numeric_column('latitude')
    
    lng_b_c = tf.feature_column.bucketized_column(lng, [119.18, 124.34])
    lat_b_c = tf.feature_column.bucketized_column(lat, [21.45,  25.56])
    
    crossed_column = tf.feature_column.crossed_column([lng_b_c, lat_b_c], 100)
    indicator_column = tf.feature_column.indicator_column(crossed_column)
    feature_layer = tf.keras.layers.DenseFeatures(indicator_column)
    loc_featrues = feature_layer(featrues).numpy()
        
    loc_columns = ['loc_bins_'+str(i) for i in range(len(loc_featrues[0]))]
    # dataset_df = pd.concat([dataset_df, pd.DataFrame(columns=loc_columns, data=loc_featrues)], axis=1)
    
    # 把路名轉成郵遞區號
    df_zipcode = pd.read_csv('3+3郵遞區號簿_2303A_2303B.csv')
    df_zipcode.columns = ['縣市', '鄉鎮市區', '郵遞區號', '路名']
    df_zipcode = df_zipcode.drop_duplicates(subset=['縣市', '鄉鎮市區', '路名'], keep="first")
    df_zipcode['縣市'] = df_zipcode['縣市'].apply(lambda x: x.replace('臺', '台'))
    df_zipcode['路名'] = df_zipcode['路名'].apply(lambda x: x.replace('臺', '台'))
    dataset_df = pd.merge(dataset_df, df_zipcode, how="left", on=['縣市', '鄉鎮市區', '路名'])
    dataset_df['郵遞區號'] = dataset_df.apply(lambda x: x['郵遞區號'] if x['郵遞區號']==x['郵遞區號'] else get_zipcode(x, df_zipcode), axis=1)
    df_zipcode['縣市鄉鎮市區'] = df_zipcode['縣市'] + df_zipcode['鄉鎮市區']
    df_zipcode = df_zipcode.drop_duplicates(subset=['郵遞區號'], keep="first")
    dataset_df = pd.merge(dataset_df, df_zipcode[['郵遞區號','縣市鄉鎮市區']], how="left", on=['郵遞區號'])
    
    if len(dataset_df[dataset_df['郵遞區號'].isna()]) > 0:
        print( dataset_df[dataset_df['郵遞區號'].isna()][['縣市', '鄉鎮市區', '路名', '郵遞區號']] )
        raise Exception('存在無法對應郵遞區號的地址。')
    
    dataset_df = dataset_df.drop(columns=['使用分區', '備註', '路名', '鄉鎮市區'])
    dataset_df = dataset_df.drop(columns=['lat','lng'])
    dataset_df = dataset_df.drop(columns=['縣市鄉鎮市區'])
    # dataset_df = dataset_df.drop(columns=['縱坐標','橫坐標'])
    
    dataset_df['縣市'] = dataset_df['縣市'].apply(lambda x: ','.join([c for c in cities if x in c][0]))
    # dataset_df = dataset_df.drop(columns=['縣市'])
    
    
    return dataset_df
