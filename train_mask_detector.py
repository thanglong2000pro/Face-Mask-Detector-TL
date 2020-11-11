# CÁCH DÙNG LỆNH
# python train_mask_detector.py --dataset dataset

# import các thư viện cần thiết
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Các tham số đầu vào
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# Đặt init learning rate, số epochs, batch_size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# lấy list images từ thư mục dataset, sau đó khởi tạo
# list data(images,...) và class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# lặp qua các đường dẫn
for imagePath in imagePaths:
	# lấy ra class label từ filename
	label = imagePath.split(os.path.sep)[-2]

	# load ảnh đầu vào(224x224) và xử lí
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image) #scale về [-1,1]

	# thêm vào data và labels list tương ứng
	data.append(image)
	labels.append(label)

# chuyển data và labels list sang Numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# chuyển label sang dạng one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# chia tập data thành 80% tập train và 20% tập test
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# sinh thêm dữ liệu bằng data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load MobileNetV2 network cho fine-tuning (bỏ đi head FC layer)
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# xây phần head của model (fine-tuning)
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# đặt phần head vừa xây vào đầu của model load
model = Model(inputs=baseModel.input, outputs=headModel)

# lặp qua các layer cơ sở của model MobileNetV2 và đóng băng để ko cập nhật hệ số những layer này
for layer in baseModel.layers:
	layer.trainable = False

# compile model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# thử predict trên tập test
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# với mỗi ảnh predict đưa ra label có xác suất lớn nhất được dự đoán tương ứng
predIdxs = np.argmax(predIdxs, axis=1)

# show classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# save model lại
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# plot trainning loss và accuracy, lưu lại plot này
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
