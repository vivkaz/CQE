from unit_disambiguator import unit_disambiguator


dis=unit_disambiguator()
print(dis.disambiguate("I ate so much and I have gained 10 pounds.","pound"))
print(dis.disambiguate("he spend around 20 pounds.","pound"))
print(dis.disambiguate("john is going to buy a new play station for around 500 pounds","pound"))
print(dis.disambiguate("this couch is heavy, maybe 20 pounds","pound"))
print(dis.disambiguate("the temperature is 2c","c"))
print(dis.disambiguate("nasdaq rose by 2p","p"))
print(dis.disambiguate("each pot of land is land measuring 0.5 a","a"))