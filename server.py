
                                                                                 
from flask import Flask, request, jsonify
import torch
import base64
from io import BytesIO
from diffusers import StableDiffusionXLPipeline
from PIL import Image
device = 'cuda:0' 
app = Flask(__name__)

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id).to(device)
 
@app.route('/txt2img', methods=['POST'])
def txt2img():
    data = request.get_json()
    prompt = data['prompt']
 
    image = pipe(prompt,num_inference_steps=25).images[0]

    buffered = BytesIO()
    image.save(buffered,format="PNG")
    base64_string = base64.b64encode(buffered.getvalue()).decode('UTF-8')
    response = {
        'image': base64_string
    }
    return jsonify(response)
 
if __name__ == '__main__':
   app.run()
 
