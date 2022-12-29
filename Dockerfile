# Use the aws python3.9 lambda image 
FROM public.ecr.aws/lambda/python:3.9


# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY requirements.txt  .
COPY .env .
COPY stopwords.pkl .

#COPY ./tokenizers /root/nltk_data/tokenizers

#COPY ./tokenizers ${LAMBDA_TASK_ROOT}


COPY nltk_data ${LAMBDA_TASK_ROOT}
COPY nltk_data ./nltk_data


# Install the function's dependencies using file requirements.txt
# from your project folder.

RUN yum install gcc -y
RUN pip --version
RUN pip install --upgrade pip

#RUN pip3 install nltk
#RUN [ "python", "-c", "import nltk; nltk.download('punkt')" ]



RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

#RUN yum update -y
#RUN yum groupinstall 'Development Tools' -y

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]