import os
import re


ann_text = ""

for subdir in ["train", "dev", "test"]:
    for file in os.listdir(os.path.join("../../data", subdir)):
        if "ann" in file:
            ann_text += open(os.path.join("../../data", subdir, file)).read()


target_is_general = len(re.findall("Target_is_general", ann_text))
source_is_author = len(re.findall("Source_is_author", ann_text))
not_on_topic = len(re.findall("NOT", ann_text))
not_first_person = len(re.findall("NFP", ann_text))

print("Target_is_general: {0}".format(target_is_general))
print("Source_is_author: {0}".format(source_is_author))
print("Not_on_topic: {0}".format(not_on_topic))
print("Not_first_person: {0}".format(not_first_person))

#with open("../../allanns.tsv", "w") as out:
#    out.write(ann_text)
