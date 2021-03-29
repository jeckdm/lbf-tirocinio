import scrapy

class BotwSpider(scrapy.Spider):
    name = 'botw'
    allowed_domains = ["botw.org"]
    start_urls = ["https://botw.org"]

    # Funzione di parse per pagina principale
    def parse(self, response):
        # Prendo tutti i link alle principali sottocategorie
        links = response.xpath('//a[@class="category-title fontsize15"]/@href') # len = 21

        for link in links: 
            cur_link = link.extract()

            # Per ogni sottocategoria principale navigo ricorsivamente fino ad arrivare alla pagina degli URL
            req = scrapy.Request("https://botw.org" + cur_link, callback = self.parse_subcategory, cb_kwargs = dict(main_sub = cur_link))
            yield req

    # Funzione di parse per pagina di ogni sottocategoria
    def parse_subcategory(self, response, main_sub):
        self.logger.info("Visitato %s", response.url)
        self.logger.info("Main subcategory", main_sub)

        # Prendo tutti i link di ogni sottocategoria
        sub_link = response.xpath('//ul[@class="categories-empty"]//a/@href')
        
        # Se sub_link esistono visito ricorsivamente, altrimenti salvo URL
        if(len(sub_link) > 0):
            for link in sub_link:
                link = link.extract()
                if(main_sub in link): # Vado solo a visitare in modo ricorsivo le pagine appartenenti alla sottocategoria principale che sto visitando
                    req = scrapy.Request("https://botw.org" + link, callback = self.parse_subcategory, cb_kwargs = dict(main_sub = main_sub))
                    yield req
                else:
                    self.logger.info(f"Ignoro {link}: visito solamente link della stessa sottocategoria")
        else:
            self.logger.info("Raggiunta foglia, salvo URLs")
            for url in response.xpath('//ul[@class="listings"]/li[@class="listing"]/a/@href'):
                yield{
                    'url': url.extract()
                }
        