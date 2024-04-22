
<h1 align="center">
  <br>
  <a href=""><img src="https://t4.ftcdn.net/jpg/06/82/81/33/360_F_682813378_Mb1bNS72oaJvoKwWpGcSvrqeK4HBkekf.jpg" alt="Wall-JuniorVB🤖" width="300"></a>
  <br>
  Junior-VB🤖
  <br>
</h1>

<h4 align="center">A model and web app service in <a href="https://monkeylearn.com/blog/what-is-aspect-based-sentiment-analysis/" target="_blank">ABSA</a> about smartphone review.</h4>

<p align="center">
  <a href="https://jvb-corp.com/vi"><img src="https://img.shields.io/badge/jvb--corp-vi-black?color=red"></a>
  <a href="https://img.shields.io/badge/release-1.0.0-black?color=%23c6e2ff">
      <img src="https://img.shields.io/badge/release-1.0.0-black?color=%23c6e2ff">
  </a>
  <a href="https://www.facebook.com/lnht1808.secsip">
    <img src="https://img.shields.io/badge/Facebook-green?style=flat&logo=facebook&color=24A3BB">
  </a>
  <a href="https://mail.google.com/mail/u/3/#inbox">
    <img src="https://img.shields.io/badge/contact-gmail-black?logo=gmail&color=%23fe935e">
  </a>
  <a href="https://img.shields.io/badge/license-MIT-green?style=flat&labelColor=gray">
    <img src="https://img.shields.io/badge/license-MIT-green?style=flat&labelColor=gray">
  </a>
</p>

<p align="center">
  <a href="#data">Data</a> •
  <a href="#model">Model</a> •  
  <a href="#web">Web</a> •
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a> •
  <a href="#license">License</a>
</p>

<p align="center">
  <img src="lottiefiles/example.gif" alt="Alt Text">
<p>

## 📁Data
* We using the data collected by [Luong Luc Phan et al., 2021](https://github.com/LuongPhan/UIT-ViSFD), you can read more about the data when read their paper on this [repo](https://github.com/LuongPhan/UIT-ViSFD).
* Or visti the paper published on [SpringerLink](https://link.springer.com/chapter/10.1007/978-3-030-82147-0_53?fbclid=IwAR00G3h4feqS5m_hu8lMbwLw22bXqOjBLrpBzs25eszMN9d7UPjjaCTEcpw).

## 📈Model
### 🤝Pre-trained Model And Fine-tuning
 - We using [PhoBERT-base](https://huggingface.co/vinai/phobert-base) model, which was created by [VinAI](https://github.com/VinAIResearch) for our problem and were inspired of fine-tuning by [Quan Dang](https://github.com/ds4v/absa-vlsp-2018).
 - You can read more details in this [paper](https://github.com/ds4v/absa-vlsp-2018/blob/main/Paper.pdf).
 - With our model, we fine tune by use slow learning rate (1e-4) and small weight decay (5e-3) of AdamW because of more aspect and data. In the 3 last pooling layers, it performance look quite better than anything.

### 🌊Results
 - You can visit our [report](https://www.canva.com/design/DAFolBANnZY/PqK7A3Vz9oXna2irnKNVdg/edit?utm_content=DAFolBANnZY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) for more information about the data, preprocessing and experiment setup.
 - Best result:

Tasks | Precision | Recall  | F1-Score  
---|---|---|---
Aspect Detection | 90.21 | 87.56 | 88.56 |
Polarity Detection | 80.59 | 77.86 | 78.66 |
Aspect + Polarity | 78.39 | 68.00 | 68.41 |

## 🤗Web

* 📝Predict random text from keyboard
  - You can try our service by enter some review about phone before uploading file.
  - By this, you can understand clearly our result.
* 🗂️Upload file
  - The limit space of streamlit is 200MB. We sorry about that.
  - From now, our model accepts 3 type of file: csv, json and excel.
  - The processing for file input may take time depending on size of file.

## 🧑‍💻How To Use

To clone and run this application, you'll need [Git](https://git-scm.com) and [Pip](https://pip.pypa.io/en/stable/installation/) installed on your computer. From your command line:

### For training models

```bash
# Clone this repository
$ git clone https://github.com/leanhtu-AI/Sentiment-Analysis.git

# Go into the repository
$ cd Sentiment-Analysis

# Install dependencies
$ pip install requirements.txt

$ python train.py

```

> [!TIP]
> I have setup for training models with gpu already, if you don't have gpu and still want training or fine tuning by yourself, you should go to this [notebook](https://colab.research.google.com/drive/1qPOqBUZ4igO62dJbiBLqs9vITMtl2AKX?usp=sharing)

### For running web (local)
```bash
$ streamlit run app.py
```
> [!TIP]
> When you running the first time, it will download the model weights(.h5) to your local filesystem. For faster running next time, you should create a folder named 'model' and move the h5 file to that. Then go to predict.py comment the downloaded model weights on cloud and uncomment the load weights local.


> [!NOTE]  
> We use the Windows command, if you are running on MacOS or Linux, you might to modify because it can be different a bit.

## 👉Credits
This repo uses the following open source packages:
- [Tensorflow](https://www.tensorflow.org/?hl=vi)
- [Streamlit](https://streamlit.io/)
- [Transformers](https://github.com/huggingface/transformers)
- [Underthesea](https://github.com/undertheseanlp/underthesea)
- Others: [nltk](https://github.com/nltk/nltk), [Regex](https://github.com/ziishaned/learn-regex), [Numpy](https://numpy.org/),[Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/stable/), [Seaborn](https://seaborn.pydata.org/index.html),etc.


## 📢Related
[Junior-VB-web]() - Web version link of our model.

> [!IMPORTANT]
> We encourage you to run this by streamlit on local becaue the web is not always available on streamlit cloud.

## 📝License

MIT

