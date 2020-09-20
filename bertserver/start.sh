touch /var/log/uwsgi.log
tail -F /var/log/uwsgi.log > /dev/stdout &

uwsgi --ini /var/www/uwsgi.ini
nginx -g "daemon off;"
