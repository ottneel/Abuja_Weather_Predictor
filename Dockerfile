FROM Python 3.9

#set working directory
WORKDIR /app

#cooy requirements file into the container
COPY requirements.txt /app

#install the python dependencies from the requirements file

RUN pip install -r requirements.txt