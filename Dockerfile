FROM pytorch/pytorch
LABEL authors="Adam Englander"

EXPOSE 5000
WORKDIR app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . ./

ENTRYPOINT ["python", "app.py"]