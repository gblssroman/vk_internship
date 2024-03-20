# 🏢 VK Internship March '24
EDA &amp; CatBoost &amp; Optuna for giving the most accurate scores for the object success prediction task.

Repo for VK Internship Data Science task.

**Instruction:**

1. Install ```requirements.txt``` via ```pip install -r requirements.txt``` (no junk :))
1. Launch ```generate_submission.py```
2. Get final ```submission.csv``` (it has already been generated in the folder ```output``` for fast reference).

Other files description:
* ```classifier.cbm``` - Trained CatBoost model
* ```do_eda.py``` - Script used in ```generate_submission.py``` for given datasets preparation
* ```datasets``` - Folder containing datasets
* ```cols_to-drop.pkl``` - Columns to-be-dropped causing multicollinearity (dict).

---

**Инструкция:**

1. Установите ``requirements.txt`` через ``pip install -r requirements.txt``
1. Запустите ``generate_submission.py``
2. Получите окончательный ``submission.csv`` (он уже сгенерирован в папке ``output`` для быстрого референса).

Описание других файлов:
* ```classifier.cbm``` - Обученный регрессор CatBoost
* ``do_eda.py`` - скрипт, используемый в ``generate_submission.py`` для подготовки датасетов и фичей
* ``наборы данных`` - Папка, содержащая наборы данных
* ``cols_to-drop.pkl`` - Столбцы, подлежащие удалению, вызывающие мультиколлинеарность и по факту не дающие полезной информации.

