import pandas as pd
import csv
import time
import os
from MovieManager import MovieManager

BATCH_SIZE = 40
RATE_LIMITER = 11 #seconds

def write_to_csv(fileName, headers, rows):
    with open(fileName, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        writer.writeheader()
        [writer.writerow(row) for row in rows]

def initialize_csv(fileName, headers):
    with open(fileName, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

def append_to_csv(fileName, headers, rows):
    with open(fileName, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = headers)
        [writer.writerow(row) for row in rows]
    
def prepare_metadata(movDet, destination):
    movDet['dest'] = destination
    return movDet


movMan = MovieManager('./config.ini')
metadataFile = 'movieMeta_all.csv'
location = '/scratch/team4/posters_all/'
links = pd.read_csv('./links.csv')
movList = []
tmdbID = links['tmdbId']

headers = ['title','img','rating','genre','desc','trailer','id','dest']

initialize_csv(metadataFile, headers)

batch_count = 0
start_time = time.time()

for id in tmdbID:
    batch_count += 1
    try:
        movDet, destination = movMan.download_poster(id, location)
        movList.append(prepare_metadata(movDet, destination))
    except:
        print("Movie", id, "not found")
    if batch_count % BATCH_SIZE == 0:
        # get code execution time for rate limiting
        end_time = time.time()
        running_time = end_time-start_time
        buffer = max(0, RATE_LIMITER - running_time)
        print(running_time, buffer)
        # append new movies to csv and reset list
        append_to_csv(metadataFile, headers, movList)
        movList = []
        # wait for buffer time and continue
        time.sleep(buffer)
        start_time = time.time()
        
append_to_csv(metadataFile, headers, movList)
    
#write_to_csv(metadataFile, movList[0].keys(), movList)
print('hello')

