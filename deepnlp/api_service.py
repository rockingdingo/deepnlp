#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
API service module of deepnlp.org
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals # compatible with python3 unicode coding

import requests
import re
import sys
if (sys.version_info>(3,0)): from urllib.parse import quote 
else : from urllib import quote

base_url = 'http://www.deepnlp.org'

def fetch_str_input(prompt):
    var = ""
    if (sys.version_info > (3,0)): #py3
        var = input(prompt)
    else :
        var = raw_input(prompt)
    return var
# username = fetch_str_input("username:")

def init():
    # register the first time
    print ("Connnecting to deepnlp.org server")
    print ("Register to get full API usage")
    
    username = fetch_str_input("username:")
    while (len(username) == 0):
        print ('username cannot be none')
        username = fetch_str_input("username:")
    
    email = fetch_str_input("email:")
    while (check_email_format(email) == False):
        print ('email format incorrect!')
        email = fetch_str_input("email:")
    
    password = fetch_str_input("password:")
    while (len(password) == 0):
        print ('password cannot be none')
        password = fetch_str_input("password:")
    
    register_url = base_url + '/account/register/'
    userInfo = {'username' : username, 
                'email' : email,
                'password' : password,
                }
    
    res = requests.post(register_url, data=userInfo)
    credentials = {}
    if (res.status_code == 200):
        print ('Registration succeeded!Return your safe login credentials.')
        credentials = {'username': username, 'password':password}
    else:
        print ('Registration failed, please check your input information')
        credentials = {}
    return credentials

def check_email_format(email):
    if re.match("^.+\\@(\\[?)[a-zA-Z0-9\\-\\.]+\\.([a-zA-Z]{2,3}|[0-9]{1,3})(\\]?)$", email) != None:
        return True
    return False

def connect(credentials):
    login = credentials
    default_login = {'username': 'pypi_default' , 'password': 'pypi_passw0rd'}
    if (len(login)==0):
        print ("Warning: credentials dictionary is empty")
        print ("Run 'login = api_service.init()' to initialize your personal account with full API access")
        print ("loading default login for pypi with limited API access")
        login = default_login

    if('username' not in login.keys()):
        print ("key 'username' missing in credentials")
        print ("loading default login with limited API access")
        login = default_login

    if('password' not in login.keys()):
        print ("key 'password' missing in credentials")
        print ("loading default login with limited API access")
        login = default_login
    
    login_url = base_url + '/account/login/'
    login_cookie = requests.post(login_url, data=login)
    secure_cookie = login_cookie.cookies # save cookie for future use
    return secure_cookie

