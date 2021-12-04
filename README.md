# Neural Machine Translator App
[![Website cv.lbesson.qc.to](https://img.shields.io/website-up-down-green-red/http/cv.lbesson.qc.to.svg)](http://cv.lbesson.qc.to/) &nbsp; [![Azure](https://badgen.net/badge/icon/azure?icon=azure&label)](https://azure.microsoft.com) &nbsp; ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white&labelColor=545454)

This is a Translator app which can translate text from english to italian using NLP. More details about the NLP model architecture can be found here - https://github.com/balamurugan1603/English-to-Italian-Neural-Machine-Translator.
<br>The app was tested locally and deployed in Azure.

# Tech Stack
Tensorflow<br>
Flask<br>
HTML, CSS<br>
Azure Web apps

# Installation
1. Install python 3.9 in your local machine.
2. Install GitLFS.
3. Clone the repository using ```git lfs clone <url>```
4. Ignore ```.github``` directory. It is for deployment purpose. Deleting it locally won't cause any harm.
5. Create virtual environment and install packages using ```pip install -r requirements.txt```

# Running the app
1. In the terminal go to the local git repository where you have cloned this repo.
2. Run ```python main.py``` and navigate to ```http://localhost:5000``` in your browser.
