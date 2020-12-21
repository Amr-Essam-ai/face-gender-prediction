# CV Gender Detection

# Description
- Gender detection (from scratch) using deep learning with keras 
- API using falcon 
- this project support docker built 

# Data
- 23000 [utk](https://susanqq.github.io/UTKFace/) dataset 
- 7260 sample (our data)
- 230000 [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install -r requirements.txt

```
or 
using docker file
```bash

app="gender_detection.api"
docker build -t ${app} .
docker run -d -p 5000:5000 --name=${app} ${app}

```

## Usage
### input
```
open 127.0.0.1:5000/gender
Note : "Using POST request" 
using form data multi crop face images each image have name 
like: 
"image0" for image  one  
"image1" for image two and so on ...

```
### output
```

"image0" for image  one  

{
  "result": {
    "gender": {
      "label": [
        "male"
      ],
      "confidence": [
        99
      ]
    }
  }
}

```
