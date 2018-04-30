# main_with_ini.py
import configparser
import requests
import json
import urllib
import urllib.request


class MovieManager:

    def __init__(self, configFile):
        config = configparser.ConfigParser()
        config.read(configFile)
        self.configs = config['MOVIEMANAGER']

    def get_movie_details(self, tmdbId):
        api_key = self.get_config('API_KEY')
        headers = {'Accept': 'application/json'}
        payload = {'api_key': api_key}
        tmdb_url = self.get_config('TMDB_URL')
        url = tmdb_url.format(tmdbId, api_key)
        response = requests.get(url, params=payload, headers=headers)
        print(tmdbId, response.status_code)
        if response.status_code != 200:
            return {
                'title': "",
                'img': "",
                'rating': "",
                'genre': "",
                'desc': "",
                'trailer': "",
                'id' : tmdbId
                }
        json_data = json.loads(response.text)
        #get genres
        genres = '. '.join([item['name'] for item in json_data['genres']])
        #get title
        title = json_data['original_title']
        #get ratings
        ratings = str(json_data['vote_average']) + '/10'
        #get description
        desc = json_data['overview']
        #get trailer
        trailer_url = self.get_config('TRAILER_URL')
        trailer = trailer_url + json_data['imdb_id']
        #get poster url
        poster_url = self.get_config('POSTER_URL')
        poster = poster_url+ json_data['poster_path']
        return {
            'title': title,
            'img': poster,
            'rating': ratings,
            'genre': genres,
            'desc':desc,
            'trailer': trailer,
            'id' : tmdbId
        }

    def get_config(self, key):
        return self.configs[key]

    def download_poster(self, tmdbId, location):
        movDet = self.get_movie_details(tmdbId)
        source = movDet['img']
        destination = location + str(int(movDet['id'])) +'.jpg'
        urllib.request.urlretrieve(source, destination)
        return [movDet, destination]

#movMan = MovieManager()
#movDet = movMan.get_movie_details(269149)
#print('hello')