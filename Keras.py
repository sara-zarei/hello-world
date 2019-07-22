import keras as ks
import matplotlib.pyplot as plt

(train_img,train_lbl),(test_img,test_lbl)= ks.datasets.mnist.load_data()
print(test_img.shape)

plt.imshow(test_img[100])
plt.show()
