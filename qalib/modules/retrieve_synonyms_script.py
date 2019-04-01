import urllib.request
from urllib.parse import quote
from bs4 import BeautifulSoup

def retrieve_syn_list(word, language):

    url = "http://synonyms.woxikon.com/" + quote(language) + "/" + quote(word)

    syn_vect = []

    data = urllib.request.urlopen(url).read()

    soup = BeautifulSoup(data, "html5lib")

    soup_text = soup.text

    if 'Your search did not match any results' not in soup_text:

        for div in soup.findAll('div', attrs={'class': 'synonyms-list-content'}):

            for span in div.find_all('span', attrs={'class': 'text-black'}):
                if not span.text.strip() in syn_vect:
                    syn_vect.append(span.text.strip())

    return syn_vect
