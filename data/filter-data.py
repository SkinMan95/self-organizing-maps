from sys import stdin
import csv
from unidecode import unidecode
import logging

def filter_data(d, keys):
    places = set()
    l = 0
    for row in d:
        for key in keys:
            # logging.debug(row[key])
            places.add(unidecode(row[key]))
            if l != len(places):
                l = len(places)
                logging.debug("%d: %s", l, unidecode(row[key]))

    return places

def get_data(f):
    data = csv.DictReader(f, delimiter='\t')
    return data

def main():
    logging.basicConfig(level=logging.DEBUG)
    f = stdin
    data = get_data(f)

    keys = ["municipio"]

    places = filter_data(data, keys)
    
    for place in places:
        print(place.lower())
    

if __name__ == "__main__":
    main()
