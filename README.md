### A Data Analysis Twitter Bot

![Python](https://img.shields.io/badge/Python-v3.10.8-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue) ![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat) [![Twitter](https://img.shields.io/twitter/url/https/twitter.com/ihenrywu.svg?style=social&label=Follow%20%40ihenrywu)](https://twitter.com/ihenrywu)


This is a template you can use to build a simple twitter bot using Python on your local computer.

## Pre-requisites

 1. Register for a [twitter developer account](https://developer.twitter.com/en)  
 2. Create a [twitter app](https://developer.twitter.com/en/portal/projects-and-apps). Make sure to give it **Read and Write** permissions. Get the APP keys.



## How to use

1. Clone this repository on your local machine
2. Create a virtual environment in your project's root directory: `python3 -m venv venv && source venv/bin/activate`
3. Install the required libraries using pip: `pip install -r requirements.txt`
4. Create a file called `.env` in the root directory of your project. Put your twitter App keys there:
```
BEARER_TOKEN=<YOUR_BEARER_TOKEN>
ACCESS_TOKEN=<YOUR_ACCESS_TOKEN_HERE>
ACCESS_TOKEN_SECRET=<YOUR_ACCESS_TOKEN_SECRET_HERE>
CONSUMER_KEY=<YOUR_CONSUMER_KEY_HERE>
CONSUMER_SECRET=<YOUR_CONSUMER_SECRET_HERE>
```
5. Test the program locally by running `python run.py` from the root directory of your project



## Limitations

I will keep updating the data analysis functions of this Twitter bot.



## Attributions

Thanks a lot to Dylan Castillo. It is impossible for me to code this program and write this article without learned from his code and article: 
https://dylancastillo.co/how-to-make-a-twitter-bot-for-free/
https://github.com/dylanjcastillo/twitter-bot-python-aws-lambda



###
BY Henry
Uploaded on 2022-12-28