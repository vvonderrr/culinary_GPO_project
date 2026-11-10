import urllib.request
import os

# Создаем папку для картинок
os.makedirs("images", exist_ok=True)

# Пример: скачивание картинки по ссылке в папку images/
images = {
    "ananas.jpg": "https://foodcity.ru/storage/products/October2018/8No9uQ14ycYG7UluJEaM.jpg",
}

for name, url in images.items():
    try:
        urllib.request.urlretrieve(url, f"images/{name}")
        print(f"Скачано: {name}")
    except Exception as e:
        print(f"Ошибка при скачивании {name}: {e}")

print("\nГотово! Картинки в папке images/")