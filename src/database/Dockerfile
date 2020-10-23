FROM mysql:5.6

# COPY mysql_init_data/maindb.sql /var/lib/
COPY conf.d/mysql.cnf /etc/mysql/conf.d
COPY mysql_init/create_details.sql /docker-entrypoint-initdb.d/
