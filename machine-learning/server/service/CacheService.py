import os
from server.exceptions.PathNotFoundException import PathNotFoundException
import json
class CacheService:

    def cache_data(self, path, data, file_name):
        if not os.path.exists(path):
            raise PathNotFoundException("path not found")
        with open(f"{path}/{file_name}.json", "w") as f:
            json.dump(data, f, indent=4)

    def get_data(self, path):
        if not os.path.exists(path):
            raise PathNotFoundException("path not found")
        with open(path, 'r') as f:
            return json.load(f)
