

Hacky web app that reports Brawlstars club stats

This works by continuously fetching battle logs from the Brawlstars API, accumulating the information, and plotting some visualizations. Note that because the BS API only gives the 25 most recent battles, continuous querying and persistent storage are required.

## setup
1. Set up the webserver
    - I followed this tutorial (https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_Tutorials.WebServerDB.CreateWebServer.html).
    - It sets up an AWS EC2 t2.micro instance (free) running an Apache webserver.
    - I have a server running and a static website but nothing more yet.

2. Register with the brawl stars API
    - go here https://developer.brawlstars.com/#/getting-started, register and make an API key for the IP address of the webserver

3. Set up virtual environment on webserver
    - miniconda
    - conda env


```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```