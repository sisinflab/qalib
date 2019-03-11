import urllib.request
from urllib.parse import quote
from bs4 import BeautifulSoup


def retrieve_type(word, html_page, first_syn):
    localSoup = BeautifulSoup(html_page, "html5lib")

    if localSoup.find('span', attrs={'class': 'word-type'}):
        type = localSoup.find('span', attrs={'class': 'word-type'}).contents[0]

    else:
        url = "http://synonyms.woxikon.com/it/" + first_syn
        data = urllib.request.urlopen(url).read()
        type = retrieve_type(word, data, first_syn)

    return type


def retrieve_syn_list(word, language):

    url = "http://synonyms.woxikon.com/" + quote(language) + "/" + quote(word)

    syn_vect = []

    data = urllib.request.urlopen(url).read()

    soup = BeautifulSoup(data, "html5lib")

    soup_text = soup.text

    if 'Your search did not match any results' not in soup_text:

        for div in soup.findAll('span', attrs={'class': 'synonyms-list-first-content'}):

            for a in div.find_all('a'):
                if not a.contents[0] in syn_vect:
                    syn_vect.append(a.contents[0])

        div = soup.find('ol', attrs={'class': 'synonyms-list'})
        if div:
            for li in div.find_all('li', attrs={'class': 'synonyms-list-item clear-content'}):

                div_list = li.find('div', attrs={'class': 'synonyms-list-content'})

                for a in div_list.find_all('a'):
                    if not a.contents[0] in syn_vect:
                        syn_vect.append(a.contents[0])

    return syn_vect