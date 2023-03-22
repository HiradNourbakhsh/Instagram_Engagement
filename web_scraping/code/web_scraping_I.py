#!/usr/bin/python3.7
import instaloader
import pandas as pd

L = instaloader.Instaloader(download_videos=False)
L.login(user=" ",passwd=" ")
df = pd.DataFrame()
posts = instaloader.Profile.from_username(L.context,'natgeo').get_posts()

i=0
for post in posts:
    df = df.append({'Caption': post.caption, 'Comments': post.comments, 'URL': post.url}, ignore_index=True)
    df.to_excel("Insta_download.xlsx",index=False)
    i = i+1
    if i>500:
        break

print("Written to Insta_download.xlsx file")