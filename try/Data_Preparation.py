import cv2
import os

#------------------------PREPARATION OF DATA-----------------------------#
base_dir = 'H:\Adience'
files_list = ['fold_0_data.txt', 'fold_1_data.txt', 'fold_2_data.txt', 'fold_3_data.txt', 'fold_4_data.txt']

# get all data from txt files i.e. (user_id,face_id.imageName.jpg,gender)
all_data = []
for txt in files_list:
    with open(os.path.join(base_dir, txt), 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            data = line.strip().split('\t')
            all_data.append([data[0], data[2]+'.'+data[1], data[4]])
            
#all_data = all_data[:150]
#Converting the strings to int
gender_class = {'m': 0, 'f': 1}

#Creating the gender data which contains the path of the image and its gender in integer form
#Path of image:base_dir\\faces\\user_id\\coarse_tilt_aligned_face.face_id.imageName.jpg
gender_data = []

#Indicates undefined gender
count = 0

#Iterating over the entire data
for data in all_data:
    try:
        #If it is not undefined then put in the gender_data the image
        if data[2] != 'u':
            gender_data.append((os.path.join(base_dir,'faces\\' + data[0] + '\\coarse_tilt_aligned_face.' + data[1]),
                            gender_class[data[2]]))
    except:
        #Increase the count of undefined gender of the images
        count = count + 1

#print(len(gender_data))
        
fd = 'C:/Users/Apoorva/Documents/Downloads/try_2/try/final_data.csv'

#looping through all the images and extracting features.
with open(fd,'a') as csvfile:
    for i in range(len(gender_data)):
        img = cv2.imread(gender_data[i][0],0)
        #image resizing
        img = cv2.resize(img,(40,40))
        #conversion of image matrix to aimage array.
        img = img.flatten().reshape(-1,1).transpose()
        #writing each array to a csv file.
        for i in img[0]:
            csvfile.write(str(i)+str(","))
            
        csvfile.write(str(gender_data[i][1]))
        #csvfile.write(str(','))
        csvfile.write("\n")
    
print("Finished")