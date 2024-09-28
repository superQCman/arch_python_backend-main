FROM python:3.12.3-bullseye

COPY ./ /app/
WORKDIR /app

RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

CMD ["python", "server.py"]