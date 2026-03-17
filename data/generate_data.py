import polars as pl
import tqdm
from faker import Faker
import random

fake = Faker()

def generate_session(is_bot=False):
    if is_bot:
        return {
            'user_id': f"bot_{random.randint(1,100)}",
            'session_id': fake.uuid4(),
            'timestamp': fake.date_time_this_month(),
            'ip_address': f"192.168.{random.randint(0,255)}.{random.randint(0,255)}",
            'pages_viewed': random.randint(500, 5000),
            'session_duration': random.randint(1, 30),
            'is_bot': True
        }
    else:
        return {
            'user_id': fake.uuid4(),
            'session_id': fake.uuid4(),
            'timestamp': fake.date_time_this_month(),
            'ip_address': fake.ipv4(),
            'pages_viewed': random.randint(1, 100),
            'session_duration': random.randint(30, 3600),
            'is_bot': False
        }

print("Генерация данных...")
sessions = [generate_session(is_bot=random.random() < 0.05) 
            for _ in tqdm.tqdm(range(1_000_000))]

df = pl.DataFrame(sessions)

df.write_parquet("user_sessions.parquet", compression="zstd")
print(f"✅ Сохранено {len(df)} сессий")