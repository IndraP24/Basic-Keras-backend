FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

RUN mkdir /backend

COPY requirements.txt /backend

WORKDIR /backend

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /backend

EXPOSE 8000

CMD uvicorn api:app --host=0.0.0.0 --port=${PORT:-5000}
