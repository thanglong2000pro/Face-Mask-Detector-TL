import os
import cv2  #
import base64  # sử dụng để mã hóa ảnh dưới dạng string
import imutils  # thư viện xử lí ảnh dễ hơn
import numpy as np  #
from matplotlib import pyplot as plt  #


def draw_rectangle(image, face):
    (start_x, start_y, end_x, end_y) = face["rect"]
    # xác định màu sẽ được vẽ
    detection_rect_color_rgb = (0, 255, 255)
    # vẽ ô chữ nhật bao quanh
    cv2.rectangle(img=image,
                  pt1=(start_x, start_y),
                  pt2=(end_x, end_y),
                  color=detection_rect_color_rgb,
                  thickness=2)

    # hiển thị xác suất nhận diện
    if (face["recognition_prob"] != []):
        #
        text = "{}: {:.2f}%".format(face["name"], face["recognition_prob"])
        # vị trí vẽ
        y = start_y - 10 if start_y - 10 > 10 else start_y + 10
        # màu vẽ
        probability_color_rgb = (0, 255, 255)
        # vẽ
        cv2.putText(img=image,
                    text=text,
                    org=(start_x, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.45,
                    color=probability_color_rgb,
                    thickness=1)


def draw_rectangles(image, faces):
    # vẽ ô chữ nhật trên mỗi face được detect
    if len(faces) == 0:
        num_faces = 0
    else:
        num_faces = len(faces)
        # vẽ ô chữ nhật
        for face in faces:
            draw_rectangle(image, face)
    return num_faces, image


def read_image(file):
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    image = imutils.resize(image, width=600)
    return image


def prepare_image(image):
    image_content = cv2.imencode('.jpg', image)[1].tostring()
    encoded_image = base64.encodestring(image_content)
    to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    return to_send


def plot_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def get_folder_dir(folder_name):
    cur_dir = os.getcwd()
    folder_dir = cur_dir + os.sep + folder_name + os.sep
    return folder_dir