ls $1 | sort -n | sed "s/^/$1\//" | xargs cat | avconv -f image2pipe -i - $2
