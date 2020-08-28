CREATE DATABASE maindb;

USE maindb;

CREATE TABLE details(
	title TEXT,
	ncode TEXT not null,
	userid TEXT,
	writer TEXT,
	story TEXT,
	biggenre INTEGER,
	genre INTEGER,
	keyword TEXT,
	general_firstup INTEGER,
	general_lastup INTEGER,
	novel_type INTEGER,
	end INTEGER,
	general_all_no INTEGER,
	length INTEGER,
	time INTEGER,
	isstop INTEGER,
	isr15 INTEGER,
	isbl INTEGER,
	isgl INTEGER,
	iszankoku INTEGER,
	istensei INTEGER,
	istenni INTEGER,
	pc_or_k INTEGER,
	global_point INTEGER,
	daily_point INTEGER,
	weekly_point INTEGER,
	monthly_point INTEGER,
	quarter_point INTEGER,
	yearly_point INTEGER,
	fav_novel_cnt INTEGER,
	impression_cnt INTEGER,
	review_cnt INTEGER,
	all_point INTEGER,
	all_hyoka_cnt INTEGER,
	sasie_cnt INTEGER,
	kaiwaritu INTEGER,
	novelupdated_at INTEGER,
	updated_at INTEGER,
	weekly_unique INTEGER,
	text TEXT,
	predict_point TEXT
) ENGINE=INNODB;

CREATE INDEX ncodeindex ON details(ncode(10));