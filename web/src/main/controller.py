import json
import re

from flask import flash, request, render_template
from flask_mail import Message
import requests

from config import Config
from forms import ContactForm, TextUploadForm, URLUploadForm
from run import app, mail


@app.route('/')
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/search_by_text', methods=['GET', 'POST'])
def search_by_text():
    text_upload_form = TextUploadForm()
    if request.method == 'POST':
        if text_upload_form.validate():
            text = text_upload_form.text.data
            headers = {'Content-Type': 'application/json'}
            data = {'text': text}
            response = requests.get(Config.TEXT_SEARCH_URL, headers=headers, json=data)
            if response.json().get('success'):
                recommend_items = response.json().get('recommend_items')
                return render_template('result.html', recommend_items=recommend_items)
            else:
                return render_template('error.html')
        else:
            flash('小説の本文を入力して下さい。')
            return render_template('search_by_text.html', form=text_upload_form, success=False)
    else:
        return render_template('search_by_text.html', form=text_upload_form, success=False)

    
@app.route('/search_by_url', methods=['GET', 'POST'])
def search_by_url():
    url_upload_form = URLUploadForm()
    if request.method == 'POST':
        if url_upload_form.validate() and \
            re.fullmatch(r'https://ncode.syosetu.com/[nN].{6}/?', url:=url_upload_form.url.data):
            ncode = url[26:33].upper()
            headers = {'Content-Type': 'application/json'}
            data = {'ncode': ncode}
            response = requests.get(Config.NCODE_SEARCH_URL, headers=headers, json=data)
            if response.json().get('success'):
                recommend_items = response.json().get('recommend_items')
                return render_template('result.html', recommend_items=recommend_items)
            else:
                return render_template('error.html')
        else:
            flash('小説家になろうの小説URLを入力して下さい。')
            return render_template('search_by_url.html', form=url_upload_form, success=False)
    else:
        return render_template('search_by_url.html', form=url_upload_form, success=False)
            
    return render_template('search_by_url.html', form=url_upload_form, success=False)


@app.route('/about')
def about():
	return render_template('about.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
	contact_form = ContactForm()
	if request.method == 'POST':
		if contact_form.validate() == False:
			flash('全て入力して下さい。')
			return render_template('contact.html', form=contact_form, success=False)
		else:
			msg = Message(contact_form.subject.data, sender=Config.MAIL_USERNAME, recipients=[contact_form.email.data])
			msg.body = "Contact Information \n Name: {} \n Mail: {}  \n Content: {}".format(contact_form.name.data, contact_form.email.data, contact_form.message.data)
			mail.send(msg)
			return render_template('contact.html', success=True)
	elif request.method == 'GET':
		return render_template('contact.html', form=contact_form, success=False)