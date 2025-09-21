import time
import pandas as pd
from statics.Statics import Statics

start_time = time.time()

self = Statics()
year = 2028
countries = self.get_all_countries()
arr_dataset = []

for country in countries:
    arr = []
    arr.append(year)
    arr.append(country)
    arr.append(self.get_strong_point_num(year=year-4, country=country))
    arr.append(self.get_hhi_index_by_country(year=year, country=country))
    arr.append(self.get_award_rate(year=year, country=country))
    arr.append(self.get_host(year))        
    arr.append(len(set(self.get_participates(year=year, country=country)['Name'])))
    arr.append(self.get_history_performance(year=year, country=country))
    # 强制赋值为 2024 年
    arr.append(self.get_total_medal(year=2024, country=country).shape[0])
    arr.append(self.get_gold_medal(year=2024, country=country).shape[0])
    arr_dataset.append(arr)

res = pd.DataFrame(arr_dataset, 
             columns=['Year','NOC', 'strong_point', 'hhi', 'award_rate', 'host', 'participate_num', 'history_performance', 'TotalMedal', 'GoldMedal'])

res.to_csv('./input_future.csv', index=False)

print(f"Time taken: {time.time() - start_time} seconds")