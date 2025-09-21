from statics.Statics import Statics
import pandas as pd
import time

start_time = time.time()

self = Statics()

countries = self.get_all_countries()
valid_years = self.get_valid_years(end_year=2024)

total_medal = []
gold_medal = []
debug = 0
cnt = 0

def save(total_medal, gold_medal):
    data = self.raw_dataset.csv_file
    total_medal = pd.Series(total_medal, dtype='int64')
    gold_medal = pd.Series(gold_medal, dtype='int64')
    data['TotalMedal'] = total_medal
    data['GoldMedal'] = gold_medal
    data.to_csv('./full_features.csv', index=False)
    print("--- %s seconds ---" % (time.time() - start_time))

for country in countries: 
    for year in valid_years:
        if debug and cnt > 50:
            save(total_medal, gold_medal)
            exit()
        cnt += 1
        print("Processing: %d year, %s country" % (year, country))
        total_medal.append(self.get_total_medal(year=year, country=country).shape[0])
        gold_medal.append(self.get_gold_medal(year=year, country=country).shape[0])

save(total_medal, gold_medal)