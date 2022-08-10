# Loan_credit_modelisation

This repo contains what's needed to download and agregate the files, and create a modelisation and an explanation for the model, as used in the API and the app in adjacent repos.

## Organisation

The files are to be used in this order :
Kernel, then model, then modelisation_single

Requirements.txt contains the necessary libraries.

## Usage

Open a python console and run :
'''bash
pip install -r requirements.txt
'''
This will add the libraries to your environment.

Then, open a system terminal in the folder containing the files, and run in this order :
'''bash
python3 Kernel.py
'''
'''bash
python3 model.py
'''
'''bash
python3 modelisation_single.py
'''
You can delete the Inputs folder that appeared after running Kernel.py

This will generate everything you need to run the code held in the following repos:

https://github.com/Antonin-Lemaire/loan_credit_dashboard

https://github.com/Antonin-Lemaire/loan_credit_api
