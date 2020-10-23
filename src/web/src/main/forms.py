from flask_wtf import FlaskForm
from wtforms import validators, SubmitField, TextField, TextAreaField


class TextUploadForm(FlaskForm):
    text = TextAreaField('小説の本文を入力して下さい。', [validators.Required('小説の本文を入力して下さい。')])
    submit = SubmitField('送信する')


class URLUploadForm(FlaskForm):
    url = TextField('小説家になろうの作品URLを入力して下さい。', [validators.Required('小説家になろうの小説URLを入力して下さい。')])
    submit = SubmitField('送信する')


class ContactForm(FlaskForm):
    name = TextField('Name', [validators.Required("名前を入力して下さい。")])
    email = TextField('Email', [validators.Required("メールアドレスを入力して下さい。"), validators.Email("メールアドレスを正しく入力して下さい。")])
    subject = TextField('Subject', [validators.Required("件名を入力して下さい。")])
    message = TextAreaField('Message', [validators.Required("本文を入力して下さい。")])
    submit = SubmitField('Send')