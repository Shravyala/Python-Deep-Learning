# find the word counting a file for each line and then print the output.
# Finally store the output back to the file

text=open("word.txt", "r")
d=dict()
for line in text:
    line = line.strip()
    line = line.lower()
    words = line.split(" ")
    for word in words:
        if word in d:
            d[word] = d[word] + 1
        else:
            d[word] = 1
for key in list(d.keys()):
    store = key + ":" + str(d[key])
    print(store)
output_file=open("wordcount.txt", "w")
for key,val in d.items():
    output_file.write("{} {}\n".format(key,val))
output_file.close()


