<a name="readme-top"></a>  

<div align="center">
  <p align="center">
    <h1 align="center">WAGON LABEL CLASSIFICATION (WLC)</h1>
  </p>

  <p align="center">
    <p><strong>Приложение для распознавания номеров железнодорожных вагонов.</strong></p>
    <br /><br />
  </p>
</div>

**Содержание:**
- [Проблематика](#title1)
- [Описание решения](#title2)
- [Тестирование решения](#title3)
- [Обновления](#title4)

## <h3 align="start"><a id="title1">Проблематика</a></h3> 
Необходимо создать, с применением технологий искусственного интеллекта, MVP в виде программного решения для распознавания номеров железнодорожных вагонов.

Решение может использоваться для:
* автоматизации работы систем динамического отслеживания грузов;
* оптимизации железнодорожных бизнес-процессов.

В задаче рассматривается работа с реальными фотографиями вагонов, поэтому работа решает несколько важных задач обучения моделей:
* точное определение номера вагона (или рамы) на фотографии;
* правильное считывание и распознавание номера с помощью OCR модели.

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>

## <h3 align="start"><a id="title2">Описание решения</a></h3>

**Machine Learning:**

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

 - **Использованные модели:**
    - **```Computer Vision```**:
      - ultralytics/YoloV8;
    - **```Optical Character Recognition```**:
      - microsoft/unilm/TROCR;

**Обоснование выбора моделей:**
* transformer-based модель, предобученная на большом обьеме текста - способа "из коробки" давать хороший результат;
* удобная интеграция в любой сервис благодаря библиотеке transformers;
* удобство использования благодаря большому объему туториалов;

Ссылки на репозитории моделей:
   - [YoloV8](https://github.com/ultralytics/ultralytics)
   - [TR-OCR](https://huggingface.co/docs/transformers/en/model_doc/trocr)

Ссылка для скачивания TrOCR-base:
   - [OCR-bin](https://disk.yandex.com/d/PVj--1158dcmGg)
  
<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>



## <h3 align="start"><a id="title3">Тестирование решения</a></h3> 

Данный репозиторий предполагает следующую конфигурацию тестирования решения:

  **```Gradio + ML-models;```**

  <br />

<details>
  <summary> <strong><i> Тестирование моделей с минимальным приложением на Gradio:</i></strong> </summary>
  
  - В Visual Studio Code (**Windows-PowerShell recommended**) через терминал последовательно выполнить следующие команды:

    - Клонирование репозитория:
    ```
    git clone https://github.com/AlexeyLunyakov/WLC.git
    ```
    - Создание и активация виртуального окружения:
    ```
    cd ./vkr
    python -m venv .venv
    .venv\Scripts\activate
    ```
    - Уставновка зависимостей (CUDA 12.1 required):
    ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip3 install -r requirements.txt
    ```
    - После установки зависимостей (3-5 минут) можно запустить Gradio:
    ```
    python ./gradio/ml/app.py
    ```
    или 
    ```
    cd ./gradio/ml/
    gradio app.py
    ```

</details> 

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>

## <h3 align="start"><a id="title4">Обновления</a></h3> 

***Все обновления и нововведения будут размещаться здесь!***

<p align="right">(<a href="#readme-top"><i>Вернуться наверх</i></a>)</p>


<a name="readme-top"></a>
