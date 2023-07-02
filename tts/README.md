cd tts
use only python 3.7
python -m venv venv
source venv/bin/activate ( linux )
.\venv\Scripts\activate ( window )
pip install -r requirements.txt
librosa

---

conda create -n tf python=3.7

---

Marytts docker image

docker run -it -p 59125:59125 synesthesiam/marytts:5.2

<emotionml version="1.0" xmlns="http://www.w3.org/2009/10/emotionml" 
category-set="http://www.w3.org/TR/emotion-voc/xml#everyday-categories">

<emotion><category name="angry"/>
The quick brown fox jumps over the lazy dog.
</emotion>
