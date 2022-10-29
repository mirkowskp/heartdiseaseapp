# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

hdisease_d = {0:0,1:1}
fastingds_d = {0:0,1:1}
slslope_d = {0:"up", 1:"float"}
sex_d = {0: "Kobieta",1:"Mężczyzna"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

	st.set_page_config(page_title="HeartDisease app")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://i0.wp.com/images-prod.healthline.com/hlcmsresource/images/topic_centers/2019-4/Best_Apps_Heart_Disease_PulsePoint_Respond_400x400.png?w=315&h=840")

	with overview:
		st.title("HeartDisease app")

	with left:
		sex_radio = st.radio( "Płeć", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
		fasting_radio = st.radio( "FastingBS", list(fastingds_d.keys()), index=1, format_func= lambda x: fastingds_d[x] )
		heart_radio = st.radio( "HeartDisease", list(hdisease_d.keys()), format_func=lambda x : hdisease_d[x] )


	with right:
		age_slider = st.slider("Age", value=1, min_value=0, max_value=80)
		bp_slider = st.slider("RestingBP", min_value=0, max_value=200)
		chr_slider = st.slider("Cholesterol", min_value=0, max_value=100, step=1)
		maxhr_slider = st.slider("MaxHR", min_value=0, max_value=100, step=1)

	data = [[age_slider, sex_radio, heart_radio, bp_slider, chr_slider, maxhr_slider, fasting_radio]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba przeżyłaby zawał?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
