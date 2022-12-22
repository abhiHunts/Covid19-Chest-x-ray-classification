import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
st.set_option('deprecation.showfileUploaderEncoding', False)
from PIL import Image
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model/vgg16_best.h5')
    return model
model = load_model()

st.write('''
        # Covid19 xray classification
''')
    
file = st.file_uploader('Please Upload X-ray Image', type=['jpg','png','jpeg'])

def load_and_predict(img, model):   
    size=(224,224)
    
    
    img = tf.image.decode_jpeg(img, channels=3)
    
    img = tf.image.resize(img, list(size))
    img = tf.expand_dims(img, axis=0)
    img = img/255.
    return model.predict(img)
col1, col2  = st.columns(2)

with col1:
    if file is None:
        st.text('Please Upload and Image file')
    else:        
        img = Image.open(file)
        st.image(img, use_column_width=True)
        print(type(file.getvalue()))
        predictions = load_and_predict(file.getvalue(), model)
        class_names = ['Normal', 'Viral Pneumonia', 'Covid']
        print(np.sort(predictions)[::-1])
        pred_class = class_names[predictions.argmax()]

        with col2:
            if pred_class == 'Normal':
                st.success('This x-ray is of type : '+pred_class)
            else:
                st.error('This x-ray is of type : '+pred_class)

    
