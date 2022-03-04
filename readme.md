

Hacky web app that reports Brawlstars club stats

This works by continuously fetching battle logs from the Brawlstars API (using [brawlstats](https://github.com/SharpBit/brawlstats)), accumulating the information, and plotting some visualization. Because the BS API only gives the 25 most recent battles, continuous querying and persistent storage are required.

## example output
The visualization makes an image that, for my club, looks like this. It tracks recent activity of members (top panel) and club league tropies (bottom panel). 
<p align="center">
  <img src="http://ec2-3-142-238-38.us-east-2.compute.amazonaws.com/plot-7day-battles.png" width="500" title="hover text">
</p>


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
