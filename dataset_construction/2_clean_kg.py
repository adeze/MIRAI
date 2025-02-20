import os
from tqdm import tqdm
from collections import defaultdict
import pyarrow.csv as pacsv
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import numpy as np
import pandas as pd  # Import pandas
import pyarrow.parquet as pq

START_DATE = 202300 # dataset start date in format of yyyymm
END_DATE = 202311 # dataset end date in format of yyyymm

DATA_DIR = '../data/kg_raw'

COL_NAMES = [ # total 61 columns
    'GlobalEventID', 'Day', 'MonthYear', 'Year', 'FractionDate', # 5 event date attributes
    'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode', 'Actor1EthnicCode', 'Actor1Religion1Code', 'Actor1Religion2Code', 'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code', # 10 actor1 attributes
    'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode', 'Actor2EthnicCode', 'Actor2Religion1Code', 'Actor2Religion2Code', 'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code', # 10 actor2 attributes
    'IsRootEvent', 'EventCode', 'EventBaseCode', 'EventRootCode', 'QuadClass', 'GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles', 'AvgTone', # 10 event action attributes
    'Actor1Geo_Type', 'Actor1Geo_Fullname', 'Actor1Geo_CountryCode', 'Actor1Geo_ADM1Code', 'Actor1Geo_ADM2Code', 'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor1Geo_FeatureID', # 8 actor1 geography
    'Actor2Geo_Type', 'Actor2Geo_Fullname', 'Actor2Geo_CountryCode', 'Actor2Geo_ADM1Code', 'Actor2Geo_ADM2Code', 'Actor2Geo_Lat', 'Actor2Geo_Long', 'Actor2Geo_FeatureID', # 8 actor2 geography
    'EventGeo_Type', 'EventGeo_Fullname', 'EventGeo_CountryCode', 'EventGeo_ADM1Code', 'EventGeo_ADM2Code', 'EventGeo_Lat', 'EventGeo_Long', 'EventGeo_FeatureID', # 8 event geography
    'DATEADDED', 'SOURCEURL' # 2 other event information
]


def merge_csv_files(csv_files):
    try:
        dataset = ds.dataset(
            DATA_DIR,
            format="csv",
            partitioning=None,
            parse_options=pacsv.ParseOptions(delimiter='\t', quote_char=False),
            read_options=pacsv.ReadOptions(column_names=COL_NAMES, autogenerate_column_names=False)
        )
        table = dataset.to_table()
        event_ids = table.column('GlobalEventID').to_numpy()
        unique_event_ids, indices = np.unique(event_ids, return_index=True)
        table_unique = table.take(indices)
        return table_unique
    except Exception as e:
        print(f"Error reading data: {e}")
        return pa.table([])

def load_txt_dict(lines):
    dict_a2b, dict_b2a = {}, {}
    for line in lines:
        line = line.strip()
        a, b = line.split('\t')
        dict_a2b[a] = b
        dict_b2a[b] = a
    return dict_a2b, dict_b2a



if __name__ == "__main__":

    output_directory = '../data/kg_tmp'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # merge all 15min files to one file and save the raw data
    csv_files = os.listdir(DATA_DIR)
    table = merge_csv_files(csv_files)
    pq.write_table(table, os.path.join(output_directory, 'kg_samedate.parquet'))
   #pacsv.write_csv(table, os.path.join(output_directory, 'kg_raw.csv'), delimiter='\t')
    print(f'kg_raw.parquet saved, length: {table.num_rows}')

    # check event date and the news date should be the same
    news_dates = pc.utf8_slice(table['DATEADDED'], 0, 8)
    table = table.append_column('NewsDate', news_dates)
    mask = pc.equal(table['Day'], table['NewsDate'])
    all_table = table.filter(mask)
    pq.write_table(table, os.path.join(output_directory, 'kg_samedate.parquet'))
    # pacsv.write_csv(all_table, os.path.join(output_directory, 'kg_samedate.csv'), delimiter='\t')
    print(f'kg_samedate.csv saved, length: {all_table.num_rows}')

    # keep the earliest added date for each URL
    urls = all_table['SOURCEURL'].to_numpy()
    dates = all_table['NewsDate'].to_numpy()
    dict_url2date = defaultdict(set)
    for idx, url in tqdm(enumerate(urls), total=len(urls)):
        dict_url2date[url].add(dates[idx])

    dict_url2date_unique = {url: min(dates) for url, dates in dict_url2date.items()}
    url_days = pa.array([dict_url2date_unique[url.as_py()] for url in all_table['SOURCEURL']])
    all_table = all_table.append_column('URLday', url_days)

    mask = pc.equal(all_table['NewsDate'], all_table['URLday'])
    all_table_url = all_table.filter(mask)
    pacsv.write_csv(all_table_url, os.path.join(output_directory, 'kg_urldate.csv'), delimiter='\t')
    print(f'kg_urldate.csv saved, length: {all_table_url.num_rows}')

    # standardize actor name and event type
    dict_iso2country, dict_country2iso = load_txt_dict(open('../data/info/ISO_country_GeoNames.txt', 'r').readlines())
    iso_keys = list(dict_iso2country.keys())

    # filter out actors without country code and not in iso dictionary
    mask_notnull = pc.and_(
        pc.is_valid(all_table_url['Actor1CountryCode']),
        pc.is_valid(all_table_url['Actor2CountryCode'])
    )
    mask_in_iso = pc.and_(
        pc.is_in(all_table_url['Actor1CountryCode'], value_set=iso_keys),
        pc.is_in(all_table_url['Actor2CountryCode'], value_set=iso_keys)
    )
    mask_not_self = pc.not_equal(
        all_table_url['Actor1CountryCode'],
        all_table_url['Actor2CountryCode']
    )

    all_table_info = all_table_url.filter(pc.and_(mask_notnull, pc.and_(mask_in_iso, mask_not_self)))

    # map country codes to names
    actor1_names = pa.array([dict_iso2country[code.as_py()] for code in all_table_info['Actor1CountryCode']])
    actor2_names = pa.array([dict_iso2country[code.as_py()] for code in all_table_info['Actor2CountryCode']])
    all_table_info = all_table_info.append_column('Actor1CountryName', actor1_names)
    all_table_info = all_table_info.append_column('Actor2CountryName', actor2_names)

    # filter out invalid event types and standardize CAMEO codes
    mask_valid_event = pc.not_equal(all_table_info['EventRootCode'], '--')
    all_table_info = all_table_info.filter(mask_valid_event)

    dict_cameo2name, dict_name2cameo = load_txt_dict(open('../data/info/CAMEO_relation.txt', 'r').readlines())
    rel_names = pa.array([dict_cameo2name[code.as_py()] for code in all_table_info['EventBaseCode']])
    all_table_info = all_table_info.append_column('RelName', rel_names)

    # standardize date format
    dates = all_table_info['URLday']
    datestrs = pa.array([
        f"{date.as_py()[:4]}-{date.as_py()[4:6]}-{date.as_py()[6:]}"
        for date in dates
    ])
    all_table_info = all_table_info.append_column('DateStr', datestrs)

    # generate event strings
    eventcodes = pa.array([
        f"{date.as_py()}, {a1.as_py()}, {rel.as_py()}, {a2.as_py()}"
        for date, a1, rel, a2 in zip(
            datestrs,
            all_table_info['Actor1CountryCode'],
            all_table_info['EventBaseCode'],
            all_table_info['Actor2CountryCode']
        )
    ])

    eventnames = pa.array([
        f"{date.as_py()}, {a1.as_py()}, {rel.as_py()}, {a2.as_py()}"
        for date, a1, rel, a2 in zip(
            datestrs,
            actor1_names,
            rel_names,
            actor2_names
        )
    ])

    eventfullstrs = pa.array([
        f"{date.as_py()}, {a1c.as_py()} - {a1n.as_py()}, {relc.as_py()}-{reln.as_py()}, {a2c.as_py()}-{a2n.as_py()}"
        for date, a1c, a1n, relc, reln, a2c, a2n in zip(
            datestrs,
            all_table_info['Actor1CountryCode'],
            actor1_names,
            all_table_info['EventBaseCode'],
            rel_names,
            all_table_info['Actor2CountryCode'],
            actor2_names
        )
    ])

    all_table_info = all_table_info.append_column('QuadEventCode', eventcodes)
    all_table_info = all_table_info.append_column('QuadEventName', eventnames)
    all_table_info = all_table_info.append_column('QuadEventFullStr', eventfullstrs)

    pacsv.write_csv(all_table_info, os.path.join(output_directory, 'kg_info.csv'), delimiter='\t')
    print(f'kg_info.csv saved, length: {all_table_info.num_rows}')



