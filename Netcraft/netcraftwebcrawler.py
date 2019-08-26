# Implement a very simple web crawler. It should accept a single starting URL, such as
# http://www.bbc.co.uk/news, as its input. It should download the web page available at the
# input URL and extract the URLs of other pages linked to from the HTML source code. Although
# there are several types of link in HTML, just looking at the href attribute of <a> tags will
# be sufficient for this task. It should then attempt to download each of those URLs in turn to
# find even more URLs, and then download those, and so on. The program should stop after it has
# discovered 100 unique URLs and print them (one URL per line) as its output.

from html.parser import HTMLParser
from urllib.request import urlopen
from urllib import parse


class LinkParser(HTMLParser):

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for (key, value) in attrs:
                if key == 'href':
                    newUrl = parse.urljoin(self.baseUrl, value)
                    self.links = self.links + [newUrl]


    def getLinks(self, url):
        self.links = []
        self.baseUrl = url
        response = urlopen(url)
        if 'text/html' in response.getheader('Content-Type'):
            htmlBytes = response.read()
            htmlString = htmlBytes.decode("utf-8")
            self.feed(htmlString)
            return htmlString, self.links
        else:
            return "",[]


def spider(url, maxPages):
    pagesToVisit = [url]
    numberVisited = 0

    while numberVisited < maxPages and pagesToVisit != []:
        numberVisited = numberVisited +1
        url = pagesToVisit[0]
        pagesToVisit = pagesToVisit[1:]
        print(numberVisited, "Visiting:", url)
        parser = LinkParser()
        data, links = parser.getLinks(url)
        pagesToVisit = pagesToVisit + links


if __name__ == '__main__':
    spider("https://edition.cnn.com/", 100)
