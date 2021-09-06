import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import numpy as np
import base64

st.title(' ')
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
       data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
        <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
        ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('image.png')


def classifier(img, file):
    np.set_printoptions(suppress=True)
    model = keras.models.load_model(file)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction


st.title("Useful Garbage")
uploaded_file = st.file_uploader(" ", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded IMage.', use_column_width=True)
    st.write("Classifying Image")
    label = classifier(image, 'model.h5')
    cardboard= (label[0][0])
    glass= (label[0][1])
    metal= (label[0][2])
    paper= (label[0][3])
    plastic= (label[0][4])
    trash= (label[0][5])
    if cardboard >= 0.6:
        st.title("It's cardboard and paper.Here's some information about it and how to recycle it")
        st.write("""
850 million tons and more of cardboard are thrown away annually, roughly 1.4 million trees. and as we know trees make habitats for species, along with performing photosynthesis which helps to suck carbon dioxide out of the atmosphere. Trees are integral to a healthy forest ecosystem, deforestation is detrimental, so recycling cardboard is of the utmost importance. 
And here's some to recycle and reuse these cardboards:

-Make it into a planter by lining it with a plastic bag with some holes poked in the bottom. For added decoration, you can paint the cardboard to make it more fun!

-Paint cardboard boxes or lines with fabric to make nice storage containers 

-Cut out small circles and paint them to make coasters

-Cut into small pieces and use as furniture sliders to protect your floors

-Place over weeds in your backyard to kill weeds naturally, without any chemicals. Simply lay the broken down box flat on the weeds and wet with a hose, then cover with dirt.

-Save to send packages to other people.
        """)
    elif glass >= 0.6:
        st.title("It's glass.Here's some information about it and how to recycle it")
        st.write("""
Glass is a fully recyclable material. We can recycle glass over and over endlessly without loss in quality or purity. And recycling glass has enormous environmental benefits including reducing carbon emissions, raw material consumption, energy use, and waste. 
So how we can recycle and reuse it: Once the glass is collected and taken to be reprocessed, it is:
-Crushed and contaminants removed (mechanised colour sorting is usually undertaken at this stage if required).
-Mixed with the raw materials to colour and/or enhance properties as necessary.
-Melted in a furnace.
-Moulded or blown into new bottles or jars, not only these, we can use it to produce: Glass containers, Fiberglass, Recycled glass countertops, Foam aggregate, and Ground cover.
        """)
    elif metal >= 0.6:
        st.title(" It's metal.Here's some information about it and how to recycle it")
        st.write("""
Metal is excavated from the earth through a process called mining. Over time, the areas excavated tend to be depleted, and the miners move to other areas looking for metals to mine. This means that in case this trend does not stop or is not controlled, most areas will have huge excavation holes.
So by providing an alternative to virgin ore, recycling metal helps to reduce the devastating effects of mining.
How to recycle and reuse it?
-Once the metals have been collected, the next step of course is to sort the metals.
-Sorting includes having items broken down into their individual components, 
-Then separated for processing. Metals for recycling are then cleaned to remove non-metallic materials like paper, paint, etc.
-The next step is to compact or squeeze the metal. All the recycled materials are squeezed and squashed.
-Each metal is taken to a furnace that is specifically designed to melt that particular metal based on its specific properties, this step is calling melting. 
-And finally, after the purification process, the molten metal is then carried by the conveyor belt to a cooling chamber where it is cooled and solidified. It is at this stage that the scrap metal is made into a solid metal that can be used again.
        """)
    elif paper >= 0.6:
        st.title("It's cardboard and paper.Here's some information about it and how to recycle it")
        st.write("""
850 million tons and more of cardboard are thrown away annually, roughly 1.4 million trees. and as we know trees make habitats for species, along with performing photosynthesis which helps to suck carbon dioxide out of the atmosphere. Trees are integral to a healthy forest ecosystem, deforestation is detrimental, so recycling cardboard is of the utmost importance. 
And here's some to recycle and reuse these cardboards:

-Make it into a planter by lining it with a plastic bag with some holes poked in the bottom. For added decoration, you can paint the cardboard to make it more fun!

-Paint cardboard boxes or lines with fabric to make nice storage containers 

-Cut out small circles and paint them to make coasters

-Cut into small pieces and use as furniture sliders to protect your floors

-Place over weeds in your backyard to kill weeds naturally, without any chemicals. Simply lay the broken down box flat on the weeds and wet with a hose, then cover with dirt.

-Save to send packages to other people.
        """)
    elif plastic >= 0.6:
        st.title("It's plastic.Here's some information about it and how to recycle it")
        st.write("""
        Plastic: 
Plastic recycling is very important and must be taken seriously. Plastics make up a huge amount of solid waste and take centuries to break down in landfills or the ocean. Therefore, all recyclable plastics should be recycled to reduce landfill, conserve energy and conserve the environment. 
So here are some ways and tips to recycle and reuse it : 
-Create Recycled Plastic Bottle Supply Cups.
-Reuse Coffee Creamer Containers for Snack Storage.
-Make a DIY Plastic Bottle Planter.
-Upcycle Laundry Detergent Bottles Into a Watering Can.
-Turn a Milk Carton Into a Garden Scooper.
-Start a Herb Garden With Empty 2-Liter Bottles.
-Upcycle a Lotion Bottle Into a Charging Dock.
-Turn Plastic Bottle Trash Into a Trash Can.
        """)
    elif trash >= 0.6:
        st.title("This type of garbage is trash")
        st.write("""
        As this waste can’t be recycled or isn’t biodegradable so it can be used as fuel in EfW (energy from waste). These facilities burn waste, the process produces steam which is used to make electricity by powering a steam turbine. That turbine in turn generates heat for local businesses and homes. Whilst burning, these non-recyclable substances produce non-environmentally friendly carbon dioxide emissions. The air is cleaned and purified before it is released into the atmosphere, ensuring the process is still much safer and cost-effective than using a landfill. Creating energy, not waste, also reduces the amount of waste that is sent to landfill.
        """)
