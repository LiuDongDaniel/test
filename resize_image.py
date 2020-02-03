
import cv2
import os


def resize_image(img_path, save_path, x_resize, y_resize):
    img = cv2.imread(img_path)
    img = img[:, :, 1]
    res = cv2.resize(img, (x_resize, y_resize), None)
    cv2.imwrite(save_path, res)



img_path = '/home/huiying/luna/11.11/mhi/z/up/1/'
save_path = '/home/huiying/luna/11.11/mhi_resize_to_48_48/z/up/1/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

img_list = os.listdir(img_path)
for i in range(len(img_list)):

    img = img_list[i]
    image_sub_path = img_path + img
    save_sub_path = save_path + img[:len(img)-4] + '_48_48' + '.jpg'

    res = resize_image(image_sub_path, save_sub_path, 48, 48)

# # for test
# img_path = '/home/huiying/luna/11.11/mhi/z/up/0/1.2.156.14702.1.1005.128.1.2018051111015805911856682_361_208_432.jpg'
# save_path = '/home/huiying/luna/11.11/test.jpg'
# resize_image(img_path, save_path, 48, 48)