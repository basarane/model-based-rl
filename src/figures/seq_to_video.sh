rm $2
ls $1 | sort -n | sed "s/^/$1\//" | xargs cat | avconv -f image2pipe -i - -pix_fmt yuv420p $2
#-vcodec libx264 -preset medium -tune film  -profile:v baseline -level 30 -force_key_frames 0,1,2,3 