from statics.Statics import Statics
import pandas as pd

self = Statics()

# print(self.get_strong_point_num(year=2024, country='China'))
# print(self.get_hhi_index(2024))
# print(self.get_award_rate(2024))
# print(self.get_host(2024))
# print(self.get_participate_num(2024))
# print(self.history_performance(2024, country='China'))

countries = self.get_all_countries()
valid_years = self.get_valid_years(end_year=2024)

arr_dataset = []

cnt = 0

for country in countries:
    dataset = pd.DataFrame(data=arr_dataset, columns=['Year','NOC', 'strong_point', 'hhi', 'award_rate', 'host', 'participate_num', 'history_performance'])
    dataset.to_csv('./statics-gen.csv', index=False)     
    for year in valid_years:
        cnt += 1
        # if cnt >= 100:
        #     dataset = pd.DataFrame(data=arr_dataset, columns=['Year','NOC', 'strong_point', 'hhi', 'award_rate', 'host', 'participate_num', 'history_performance'])
        #     dataset.to_csv('./statics.csv', index=False) 
        #     exit()
        print("Processing: %d year, %s country" % (year, country))
        arr = []
        arr.append(year)
        arr.append(country)
        arr.append(self.get_strong_point_num(year=year, country=country))
        arr.append(self.get_hhi_index_by_country(year=year, country=country))
        arr.append(self.get_award_rate(year=year, country=country))
        arr.append(self.get_host(year))        
        arr.append(len(set(self.get_participates(year=year, country=country)['Name'])))
        arr.append(self.get_history_performance(year=year, country=country))
        arr_dataset.append(arr)

dataset = pd.DataFrame(data=arr_dataset, columns=['Year','NOC', 'strong_point', 'hhi', 'award_rate', 'host', 'participate_num', 'history_performance'])
dataset.to_csv('./statics.csv', index=False)        