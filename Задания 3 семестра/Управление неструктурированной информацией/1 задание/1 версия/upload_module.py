import base64
import os
import requests

# ----------------------------------------------------
# НАСТРОЙКИ (измени только эти параметры)
# ----------------------------------------------------

# <-- ВСТАВЬ СЮДА СВОЙ ТОКЕН
GITHUB_TOKEN = "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"

REPO_OWNER = "ferroshnich"
REPO_NAME = "Learning-management-system_Miroshnichenko_Denis"

# Папка, куда загружаем файлы в репозитории GitHub
TARGET_PATH = (
    "Задания 3 семестра/Управление неструктурированной информацией/"
    "1 задание/1 версия/"
)

# Путь к локальной папке (вся папка будет выгружена)
LOCAL_DIR = r"C:\Users\Denis\Desktop\пдф"   # ← Укажи свою папку
# ----------------------------------------------------


API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/"


def upload_file(local_path, github_path):
    """Загрузить один файл на GitHub."""
    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")

    url = API_URL + github_path
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    # Проверка наличия файла
    resp = requests.get(url, headers=headers)
    sha = resp.json().get("sha") if resp.status_code == 200 else None

    data = {"message": f"Auto upload {github_path}", "content": content}
    if sha:
        data["sha"] = sha

    response = requests.put(url, json=data, headers=headers)

    if response.status_code in (200, 201):
        print(f"✓ Загружено: {github_path}")
    else:
        print(f"❌ Ошибка загрузки {github_path}: {response.text}")


def upload_folder(local_folder, target_folder):
    """Рекурсивная загрузка папки."""
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_folder)
            github_path = target_folder + rel_path.replace("\\", "/")
            upload_file(local_path, github_path)


def upload_to_github():
    """Главная функция."""
    print("\n🚀 Начинаю выгрузку всей папки на GitHub...\n")
    upload_folder(LOCAL_DIR, TARGET_PATH)
    print("\n✅ Выгрузка завершена!\n")
