# JlyphV2_Backend
This is Backend for JlyphV2
## Project Init
- Python 3.10.11
- pip 23.2.1 (python 3.10)

Build the environment for this project
```shell
cd JlyphV2_Backend # enter project folder
python3.10 -m pip install virtualenv # install virtual environment
virtualenv env # create virtual environment
source env/bin/activate # enter virtual environment
deactivate # exit virtual environment
```

Install packages in the environment, and all package files are stored under env
```shell
source env/bin/activate
python3.10 -m pip install -U -r requirements.txt
python3.10 -m pip install accelerate # faster and less memory-intense model loading
```

update the requirements before exit
```shell
python3.10 -m pip check # check for package conflicts
python3.10 -m pip freeze > requirements.txt
```

Start the Flask app
```shell
# select free GPU device
CUDA_VISIBLE_DEVICES=1,2 python3.10 -m flask --app app run --host=0.0.0.0 --port=9005
```