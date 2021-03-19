import streamlit as st
from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import cv2

labels = ['Apple Braeburn','Apple Crimson Snow','Apple Golden ','Apple Golden ','Apple Golden ','Apple Granny Smith','Apple Pink Lady','Apple Red ','Apple Red ','Apple Red ','Apple Red Delicious','Apple Red Yellow ','Apple Red Yellow ','Apricot','Avocado','Avocado ripe','Banana','Banana Lady Finger','Banana Red','Beetroot','Blueberry','Cactus fruit','Cantaloupe ','Cantaloupe ', 'Carambula', 'Cauliflower','Cherry ', 'Cherry ', 'Cherry Rainier','Cherry Wax Black','Cherry Wax Red','Cherry Wax Yellow', 'Chestnut','Clementine', 'Cocos','Corn','Corn Husk','Cucumber Ripe','Cucumber Ripe ','Dates','Eggplant','Fig','Ginger Root','Granadilla','Grape Blue','Grape Pink','Grape White','Grape White ','Grape White ','Grape White ', 'Grapefruit Pink','Grapefruit White','Guava','Hazelnut','Huckleberry', 'Kaki','Kiwi','Kohlrabi','Kumquats','Lemon','Lemon Meyer','Limes','Lychee','Mandarine','Mango','Mango Red','Mangostan','Maracuja','Melon Piel de Sapo','Mulberry','Nectarine','Nectarine Flat','Nut Forest','Nut Pecan','Onion Red','Onion Red Peeled','Onion White','Orange','Papaya','Passion Fruit','Peach','Peach ','Peach Flat','Pear','Pear ','Pear Abate','Pear Forelle','Pear Kaiser','Pear Monster','Pear Red','Pear Stone','Pear Williams','Pepino','Pepper Green','Pepper Orange','Pepper Red','Pepper Yellow','Physalis','Physalis with Husk','Pineapple','Pineapple Mini','Pitahaya Red','Plum','Plum ','Plum ','Pomegranate','Pomelo Sweetie','Potato Red','Potato Red','Potato Sweet','Potato White','Quince','Rambutan','Raspberry','Redcurrant','Salak','Strawberry','Strawberry Wedge','Tamarillo','Tangelo','Tomato ','Tomato ','Tomato ','Tomato ','Tomato Cherry Red','Tomato Heart','Tomato Maroon','Tomato Yellow','Tomato not Ripened','Walnut','Watermelon']

converter = tf.lite.TFLiteConverter.from_keras_model('fruits_98.h5')
model = converter.convert()

realtime_update = st.sidebar.checkbox("Update in realtime", True)

st.title("Fruits Image Classifier")

uploaded_file = st.file_uploader("Choose a file")

col1,col2 = st.beta_columns(2)
if st.button('PREDICT'):
  if uploaded_file is None:
    st.title("Upload Image")
  else:
    img = Image.open(uploaded_file)
    img = image.img_to_array(img)
    img = cv2.resize(img,(224,224),interpolation=cv2.INTER_NEAREST)
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)
    pred = model.predict(img)[0]*100
    list = sorted(zip(pred, labels), reverse=True)[:3]
    with col1:
      st.subheader("Model Input:")
      st.image(img,clamp=True)
    with col2:
      st.subheader("Predictions:")
      for i,j in list:
        a = str(i)+' %\t'+' : '+j
        st.subheader(a)
    
