# Final Project

Benjamin Chang, bmc011@ucsd.edu


## Abstract Proposal

The goal for this project is to generate visual images that can have particular types of desired art styles or patterns. This can be useful for creative designers who wish to see what their artwork would look like in a particular style or with specific types of features or patterns. It can be used for the inspiration of new art styles as well. The method for this project is neural networks which is arguably the most popular machine-learning method to date. The three networks used in this project are DeepDream, StyleTransfer, and CycleGAN. This project is based off of the third generative visual project. The use of CycleGAN is the addition to the final project. This work will be presented regarding the creative goals, the techniques used, and the results.

Project Folder: https://datahub.ucsd.edu/user/bmc011/tree/ml-art-final-benchang

## Project Report

- Final_Project_Report.pdf

## Model/Data

Briefly describe the files that are included with your repository:
Datasets/Models are too large for Github and can be found in the Google Drive linked below:
https://drive.google.com/open?id=1vXCQPwxvfSjs6QyWm41LUl-AL25gCQSL
- trained models
    - g_model_AtoB_######.h5
    - g_model_BtoA_######.h5
- training data
    - human2animal.zip
- testing data
    - human2animal.zip
    - custom_data

## Code

Your code for generating your project:

- Jupyter notebooks 
    - DeepDream & StyleTransfer (from Project 3): https://datahub.ucsd.edu/user/bmc011/notebooks/generative-visual-benjamin-chang-1/Project_3_Generative_Visual.ipynb
    - CycleGAN: https://datahub.ucsd.edu/user/bmc011/notebooks/ml-art-final-benchang/Final%20Project.ipynb#

## Results

Video Results
- Aquaman Clip
https://www.youtube.com/watch?v=jwP2OJn-2Cc

- Processed Aquman Clip
https://www.youtube.com/watch?v=Mt5g_ZIgcpI

Image Results (Less Interesting)
- Result_1.png
- Result_2.png
- panther1_final2_final.png
- panther1_final_final.png
- panther5_final_final.png
- panther6_final_final.png

## Technical Notes

All of the code for this project runs in JupyterNotebook. I modified the code multiple times while testing to produce different types of outputs so running the code may not yield the same result as the one mentioned here. Also, sometimes the code bugs out for some reason when at other times the same identical code has no issues running. 

There were quite a few things I needed to add to CycleGAN to get it to work for videos. Particularly, there were a lot of issues with making sure the frames were in the right order. Comments on my code for running the CycleGAN model on video is below.


I wrote the below code to convert the mp4 video file into jpg image frames. The code is to be run in a folder where the video frame files are stored.

        CODE:
        
        #Use code below in folder where you want video frame files stored
        #Downloads frames as jpeg files
        import cv2
        vidcap = cv2.VideoCapture('Aquaman_clip.mp4')
        success,image = vidcap.read()
        count = 0
        while success:
          cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
          success,image = vidcap.read()
          print('Read a new frame: ', success)
          count += 1
        print(count) # 480 frames


    I modified the original load_images function and created a new loder function named 'load_images_vid'. This function loads the jpg images from the code folder into a list with variable name 'data_list'. When the images were saved into the folder, they are not automatically ordered according to the numerical index in their names, but are instead ordered somewhat randomly. This loader function grabs the numberical index in the name of each image frame file and places the image into its a spot in the data_list array matching its index. This allows all the files to be loaded into memory in correct order and makes it possible to generate a video output with its frames in the correct order.

        CODE:
        
        # load all images in a directory into memory
        def load_images_vid(path, size=(256,256)):
            #data_list = list()
            data_list = [None] * 480
            # enumerate filenames in directory, assume all are images
            for filename in listdir(path):
                print(filename)
                frameNumber = int(filename[5:-4])
                print(frameNumber)

                # load and resize the image
                pixels = load_img(path + filename, target_size=size)
                # convert to numpy array
                pixels = img_to_array(pixels)
                # store
                data_list[frameNumber] = pixels

            return asarray(data_list)
            
I did not write the code below, but will explain it.

The below code is then used to compress the image frames into a compressed numpy format.

        CODE:
        
        # dataset path
        path = 'Aquaman_clip_frames/'
        # load dataset A
        dataA1 = load_images_vid(path)
        #print(size(dataA1))

        dataA = vstack((dataA1, dataAB))
        print('Loaded dataA: ', dataA.shape)

        # save as compressed numpy array
        filename = 'Aquaman_clip_frames.npz'
        savez_compressed(filename, dataA, dataB)
        print('Saved dataset: ', filename)

The compressed frames can then be loaded into arrays as follows.

        CODE:
        
        # load dataset
        A_data, B_data = load_real_samples('Aquaman_clip_frames.npz')
        print('Loaded', A_data.shape, B_data.shape)

And a model selected to pass the frames through.

        CODE:
        
        # load the models
        cust = {'InstanceNormalization': InstanceNormalization}
        model_AtoB = load_model('g_model_AtoB_010280.h5', cust)
        model_BtoA = load_model('g_model_BtoA_010280.h5', cust)

Below, I heavily modified the image prediction code to store the original/modified files into lists named 'A_real_all', 'B_generated_all', and 'A_reconstructed_all'. This ensured the output frames were ordered correctly when the model is run on the video frame images. While doing this, I came across an issue where the predict function was requesting a 4 dimension input. I found this to be pretty strange. To solve this, I dove into the select_sample function to find that it required me to select an input image using an array [i] rather than i. That is, I called the image data at location 'i' using A_data[[i]] instead of A_data[i] which I originally used.

        CODE:

        # plot A->B->A
        import imageio

        A_real_all = [None] * 480
        B_generated_all = [None] * 480
        A_reconstructed_all = [None] * 480

        for i in range(480):
            #A_real = select_sample(A_data, 1)
            A_real = A_data[[i]]
            A_real_all[i] = A_real

            #B_generated = model_AtoB.predict(A_real)
            B_generated = model_AtoB.predict(A_real)
            B_generated_all[i] = B_generated

            #A_reconstructed = model_BtoA.predict(B_generated)
            A_reconstructed = model_BtoA.predict(B_generated)
            A_reconstructed_all[i] = A_reconstructed

            #Show Plot
            show_plot(A_real, B_generated, A_reconstructed)
            
I then added code to go convert the lists to arrays for processing purposes later.

        CODE:
        
        from numpy import array
        A_real_all = array(A_reconstructed_all)
        B_generated_all = array(B_generated_all)
        A_reconstructed_all = array(A_reconstructed_all)

I wrote the code below to then normalize the images frames from [-1,1] to [0,1] and display them. 

        CODE:
        
        # plot images row by row
        import time
        import numpy as np

        # scale from [-1,1] to [0,1]
        #IMPORTANT:
        #Uncomment line below once everytime you run to map data between [0,1]
        #Can change to A_data, A_real_all, B_generated_all, or A_reconstructed_All for different styles
        #A_data_out = (A_reconstructed_all + 1) / 2.0

        for i in range(480):
            # reshape each frame before plotting
            #np.reshape(A_data_out[i], (256,256,3))
            # define subplot
            #pyplot.subplot(1, len(A_data), 1 + i)
            # turn off axis
            #pyplot.axis('off')
            # plot raw pixel data
            #pyplot.imshow(A_data_out[i])
            show_plot(A_data_out[i],A_data_out[i],A_data_out[i])
            pyplot.show()
            time.sleep(0.1)
            
I wrote used the code below to finally conver the array containing the frames into an mp4 video file using the code below.

        CODE:

        import imageio
        A_data_out *= 255 # or any coefficient
        print(A_data_out)
        A_data_out = A_data_out.astype(np.uint8)
        imageio.mimwrite('A_reconstructed_Aquaman_clip.mp4', A_data_out , fps = 24)
        
Lastly, I displayed the video in JupyterNotebook using the below code.

        CODE:
        
        from IPython.display import Video

        Video("A_real_Aquaman_clip.mp4")

## Reference

- Brownlee, Jason. How to Develop a CycleGAN for Image-to-Image Translation with Keras,
Machine Learning Mastery, 8 Aug. 2019, https://machinelearningmastery.com/cyclegan-tutorial-with-keras/.

