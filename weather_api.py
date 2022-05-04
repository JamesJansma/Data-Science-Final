############################################################
#  Programmer: James Jasma                                 #
 # Class: CPSC 222, Spring 2022                            #
#  Data Assignment #4                                      #
 # 3/14/2022                                               #
#                                                          #
 #                                                         #
#  Description: I this program it asks a reader for a large
# city and then with that city finds the longitude
# and latitude of it using the mapquest geocoding api
# and once that is aqcuired then uses the metostat
# rapid api to find a weather station nearby the 
# large city and once that is found the same api
# is used to get the daily weather data from the past year.
# Once that data is read then it is cleaned by
# filling in missing values using interpolate and if
# there are more than 50% missing values in a column
# the column is deleted. once the data is cleaned it 
# is written to a csv file        
#############################################################

import json
import requests
import pandas as pd
import numpy as np
import utils
import importlib
importlib.reload(utils)

url = "http://open.mapquestapi.com/geocoding/v1/address?key=lUajORcSmVUPPfsznOFRXohmkYHzGTBS&location="

location = input("Enter the name of a large city: ")
url += location.replace(" ","+")

response = requests.get(url)
json_str = response.text
json_obj = json.loads(json_str)
results_list = json_obj["results"]

for results_obj in results_list:
    location_list = results_obj["locations"]
    first_location = location_list[0]
    lat_and_long = first_location["latLng"]
    lat = str(lat_and_long["lat"])
    long = str(lat_and_long["lng"])
#print(lat)
#print(long)


station_url = "https://meteostat.p.rapidapi.com/stations/nearby?"
station_url += "lat=" + lat
station_url += "&lon=" + long
headers = {"x-rapidapi-key": "{Your API key here}"}

response_station = requests.get(url=station_url, headers=headers)
station_json_str = response_station.text
station_json_obj = json.loads(station_json_str)
station_list = station_json_obj["data"]
station_data = station_list[0]
station_id = station_data["id"]
#print(station_id)


weather_url = "https://meteostat.p.rapidapi.com/stations/daily?"
weather_url += "station=" + station_id
weather_url += "&start=2021-10-03" 
weather_url += "&end=2022-04-14&units=imperial" 

response_weather = requests.get(url=weather_url,headers=headers)
weather_str = response_weather.text
weather_obj = json.loads(weather_str)
results_arr = weather_obj["data"]
df = pd.DataFrame(results_arr)
df.set_index("date")
df.to_csv(location+"_daily_weather.csv",index=False)


df.replace("", np.NaN, inplace=True)

for column in df:
    if df[column].isnull().sum() > 182:
        df.drop(column, axis=1, inplace=True)
    else:
        df.interpolate(method ='linear', limit_direction ='forward')
        df[column].fillna( method ='ffill', inplace = True)
df.drop(["tmax","tmin"],axis=1,inplace=True)



df.to_csv(location+"_daily_weather_cleaned.csv",index=False)