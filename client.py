import requests, PIL.Image as Image,io,base64
 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
 
response = requests.post("http://127.0.0.1:5000/txt2img",json={"prompt":"Copenhagen as a solarpunk city"})
img_str = response.json()["image"]
print(img_str)
image=Image.open(io.BytesIO(base64.b64decode(img_str)))
image.save("response.png")
