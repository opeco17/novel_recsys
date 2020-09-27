FROM mysql:latest

COPY mysql_init_data/maindb.sql /var/lib/

COPY conf.d /etc/mysql/

COPY mysql_init /docker-entrypoint-initdb.d
