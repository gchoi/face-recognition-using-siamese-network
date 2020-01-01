#%% Import libraries
from google_images_download import google_images_download

#%% Main
response = google_images_download.googleimagesdownload()

# Define any person(celb) you want :)
persons = ["차범근",
           "IU",
           "김혜수",
           "김연아",
           "이영표",
           "Michelle Chen",
           "박찬호",
           "박지성",
           "박태환",
           "손흥민",
           "송강호",
           "Takeuchi Yuko",
           "Ueno Juri"]

person_keywords = ""
for person in persons:
    person_keywords += person + ","

# creating list of arguments
arguments = {"keywords":person_keywords,
             "limit":10,
             "print_urls":True}

# passing the arguments to the function
paths = response.download(arguments)