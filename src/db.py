from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy import Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime
import json

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@postgres:5432/calorflow')

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TrainingRun(Base):
    __tablename__ = 'training_run'
    id = Column(Integer, primary_key=True, index=True)
    process = Column(String, index=True)
    status = Column(String, default='created')
    progress = Column(Float, default=0.0)
    start_ts = Column(DateTime, default=datetime.utcnow)
    end_ts = Column(DateTime, nullable=True)
    metrics = Column(Text, nullable=True)
    error = Column(Text, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)
