# Super Resolution \ Image restoration web-service

|Developer|Contacts|
|---------|--------|
|[Chellick](https://github.com/chellick)|<diggerpotato173@gmail.com>|

------------------------------

## Info

This web service is designed to improve image quality using modern image processing techniques and artificial intelligence. The service allows users to upload low-resolution images and receive images with improved quality, increased clarity and detail at the output.

## Setup

TODO: docker migration to fully-independent web-service

### Requirements

* python 3.8+
* virtual env

### Dependencies

```sh
pip install -r requirements.txt
```

### Activation and usage

1. Activate virtual env

```sh
source venv/bin/activate  # for Unix
venv\Scripts\activate  # for Windows
```

2. Run server in ```super-resolution``` dir

```sh
py manage.py runserver
```

3. Visit _<http://127.0.0.1:8000/index>_


________________________


Huge thanks to [SWIN2SR](https://github.com/mv-lab/swin2sr) developers. Used [pretrained model](https://github.com/mv-lab/swin2sr/releases)
