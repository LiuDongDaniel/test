import os
import cv2
import numpy as np

"""
this function can produce the motion histroy image(MHI) form the continuous image

the parameter description:

image_path - the input of this function should be a file which include the continuous CT slices, 
              and the name of the image should be the continuous number like '001.bmp', '002.bmp', '003.bmp'....

save_path  - the save path of the MHI

threshold  - the value for measuring the difference between two continuous image

duration   - give this value for the recently changed picture

gradient   - if the pixel don't change in the next slice, the value of pixel in the MHI will decrease teh gradient value

direction  - describ the direction about the MHI process 
             up means the process from 1 slice to n slice
             dowm means the process from n slice to 1 slice
             center_to_up means the process from the center of the slice to the first slice
             center_to_down means the process from the center of the clice to the last slice  
"""



def motion_history_image(image_path, save_path, threshold, duration, gradient,direction):

    # obtain image and order them
    image_list = os.listdir(image_path)
    image_order = []
    for i in range(len(image_list)):
        num = image_list[i][0:3]
        image_order.append(num)

    if direction == 'up':
        image_order.sort()
    elif direction == 'down':
        image_order.sort(reverse=True)
    elif direction == 'center_to_up':
        duration = int(duration / 2)
        image_order.sort(reverse=True)
        image_order = image_order[int(len(image_order) / 2):]
    elif direction == 'center_to_down':
        duration = int(duration / 2)
        image_order.sort()
        image_order = image_order[int(len(image_order) / 2):]

    img = image_path + image_order[0] + '.bmp'
    frame = cv2.imread(img)[:, :, 1]
    h, w = frame.shape[:2]
    motion_history = np.zeros((h, w), np.float32)

    # calculate the mhi for each image
    for i in range(len(image_order)):
        img = image_path + image_order[i] + '.bmp'
        frame = cv2.imread(img)[:, :, 1]
        if i == 0:
            prev_frame = frame.copy()

        # calculate the contrast intensity
        frame_diff = cv2.absdiff(frame, prev_frame)
        fgmask = cv2.threshold(frame_diff, threshold, 1, cv2.THRESH_BINARY)

        # assign value for the changing part
        intensity = fgmask[1] * duration

        # inverse mask of fgmask
        ones_mask = np.ones((h, w), np.float32)
        inver_mask = ones_mask -fgmask[1]

        # assign value for attenuation part
        attenuation = inver_mask * motion_history - gradient
        attenuation[attenuation < 0] = 0

        motion_history = intensity + attenuation
        prev_frame = frame.copy()
    # normalized the value
    max_intensity = motion_history.max()
    min_intensity = motion_history.min()
    diff = max_intensity - min_intensity
    if diff <= 0:
        diff = 1
    mh = np.uint8(np.clip((motion_history - min_intensity) / diff, 0, 1) * 255)
    cv2.imwrite(save_path, mh)


image_path = '/home/huiying/luna/11.20/dataset/luna/0/'
save_path = '/home/huiying/luna/11.20/dataset/mhi/0/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

file_list = os.listdir(image_path)
for i in range(len(file_list)):
    file = file_list[i]
    image_sub_path = image_path + file + '/'
    save_sub_path = save_path + file + '.jpg'
    image_list = os.listdir(image_sub_path)
    duration = int(len(image_list))

    motion_history_image(image_sub_path, save_sub_path, 30, duration, 1, 'up')




