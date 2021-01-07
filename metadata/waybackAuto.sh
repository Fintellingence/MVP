string="$"
for url in $(cat finvizURLs.txt); do
  wayback-machine-scraper -a "\'$url\$\'" -f 20100101 -t 20200101 $url -v &> output.txt
done
