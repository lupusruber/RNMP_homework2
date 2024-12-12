#!/bin/sh

cd data
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip && rm -f ml-100k.zip

mv  ml-100k/* .
rmdir ml-100k            

rm *.base *.test *.pl mku.sh
rm -f u.info
rm README
iconv -f ISO-8859-1 -t UTF-8 u.item -o u.item.new && mv u.item.new u.item

sed -i '1s/^/user_id\titem_id\trating\ttimestamp\n/' u.data
sed -i '1s/^/genre|genre_id\n/' u.genre
sed -i '1s/^/user_id|age|gender|occupation|zip_code\n/' u.user
sed -i '1s/^/occupation\n/' u.occupation
sed -i '1s#^#movie_id|movie_title|release_date|video_release_date|imdb_url|unknown|action|adventure|animation|childrens|comedy|crime|documentary|drama|fantasy|film_noir|horror|musical|mystery|romance|sci_fi|thriller|war|western\n#' u.item

