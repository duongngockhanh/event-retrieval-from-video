<h1 align="center"> Event Retrieval from Video - HCMAIC 2023 </h1>

1. Our team name: **Science AIO**
2. My roles: **Leader** - Researcher - Developer



## System pipeline

<img src="./images/system_pipeline.png" alt="pipeline image" style="zoom:70%;" />



## Usage

### Setup 
```
conda create -n py38 python==3.8
conda activate py38
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```

### Run 
```
python app.py
```

Run this URL in your browser: http://0.0.0.0:5001/home?index=0

<img src="./images/demo.jpg" alt="demo image" style="zoom:70%;" />

_Note_: We use 2 versions of CLIP to increase the diversity of displayed results.



## Documents
### Faiss
1. Faiss: Facebook AI Research Search Similarity ([Docs](https://faiss.ai/index.html))

### CLIP and Prompt Engineering for CLIP
1. Learning Transferable Visual Models From Natural Language Supervision - 2021 ([Paper](https://arxiv.org/pdf/2103.00020.pdf) - [GitHub](https://github.com/openai/CLIP) - [Blog](https://openai.com/research/clip))
2. How to Try CLIP: OpenAI's Zero-Shot Image Classifier ([Blog](https://blog.roboflow.com/how-to-use-openai-clip))
3. Learning to Prompt for Vision-Language Models - CoOp - 2022 ([Paper](https://arxiv.org/pdf/2109.01134.pdf))
4. Towards Robust Prompts on Vision-Language Models - 2023 ([Paper](https://arxiv.org/pdf/2304.08479.pdf))
5. Prompt Engineering: The Magic Words to using OpenAI's CLIP - 2021 ([Blog](https://blog.roboflow.com/openai-clip-prompt-engineering))
   
### TransNet
1. TransNet: A Deep Network for Fast Detection of Common Shot Transitions ([GitHub](https://github.com/soCzech/TransNet))
2. TransNet V2: Shot Boundary Detection Neural Network ([GitHub](https://github.com/soCzech/TransNetV2/tree/master))