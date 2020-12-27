import os,glob
index = 1
for song_path in glob.glob(r"./data/peace/*.mp3"):
    song_label = os.path.basename(song_path)
    song_name = song_label.split("-")[-1]
    os.rename(r'./data/peace/'+song_label,r'./data/peace/'+str(index)+'-6-'+song_name)
    index += 1



