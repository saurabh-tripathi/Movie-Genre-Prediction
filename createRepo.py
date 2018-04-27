import pandas as pd
import csv
from MovieManager import MovieManager


def write_to_csv(fileName, headers, rows):
    with open(fileName, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        writer.writeheader()
        [writer.writerow(row) for row in rows]


def prepare_metadata(movDet, destination):
    movDet['dest'] = destination
    return movDet


movMan = MovieManager('./config.ini')
metadataFile = 'movieMeta.csv'
location = './posters/'
links = pd.read_csv('.\ml-20m\ml-20m\links.csv')
movList = []
tmdbID = links['tmdbId'].head(5)
for id in tmdbID:
    movDet, destination = movMan.download_poster(id, location)
    movList.append(prepare_metadata(movDet, destination))
print('hello')

