import gradio as gr
import webbrowser, os
import time
import plotly.express as px
from processing import *

filepath="./files/"

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="lime",
    text_size="lg",
    spacing_size="lg",
    font=[gr.themes.GoogleFont('Inter'), gr.themes.GoogleFont('Limelight'), 'system-ui', 'sans-serif'],
    # Montserrat 
).set(
    block_radius='*radius_xxl',
    button_large_radius='*radius_xl',
    button_large_text_size='*text_md',
    button_small_radius='*radius_xl',
)

def warning_file():
    gr.Warning("Выберите файл для распознавания!")

def info_fn():
    gr.Info("Для начала работы - загрузите ваш файл")

def info_req():
    startup_conf()
    gr.Info("Распознавание может занять некоторое время")
    
def info_res():
    gr.Info("Посмотреть обработанные файлы можно, нажав на кнопку ниже")

def photoProcessing(file, ):
    time.sleep(1)
    print(file)
    if file is not None:
        info_req()
        sign_detection(file, './files/detection.jpg')
        number, probability = sign_recognition('./files/detection.jpg', 'recognition.jpg')
        string = f'Номер: {number}\n\nУверенность OCR-модели: {probability:.2f}%'
        info_res()
        return './files/detection.jpg', './files/cropped_image_recognition.jpg', string
    else:
        warning_file()
        return None, None, None


def videoProcessing(file, ):
    time.sleep(1)
    if file is not None:
        info_req()
        process_video(source=file, destination='result.mp4')
        info_res()
        with open('detections.csv', mode='r') as detect_file:
            string = detect_file.readlines()
        full_text = ''
        for el in string:
            el = el.replace('\n', '')
            data = el.split(',')
            full_text += 'Начало интервала: ' + data[0] + '; Конец интервала: ' + data[1] + '; Количество встреченных лосей (автомобилей): ' + data[2] + '\n'
        return 'result.mp4', full_text
    else:
        warning_file()
        return None, None, None

def fileOpen():
    webbrowser.open(os.path.realpath(filepath))
   
output = [gr.Dataframe(row_count = (4, "dynamic"), col_count=(4, "fixed"), label="Predictions")]

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("""<a name="readme-top"></a>\
                    <p align="center" ><font size="30px"><strong style="font-family: Limelight">WAGON LABEL CLASSIFICATION</strong></font></p>
                    <p align="center"><font size="5px">Автоматизированный инструмент для детекции и распознавания номеров железнодорожных вагонов<br></font></p>
                    <p align="center"></p>""")

    with gr.Row():
        with gr.Column():
            with gr.Tab('Детектирование и распознавание по фотографии'):
                file_photo = gr.File(label="Фотография", file_types=['.png','.jpeg','.jpg'])
                with gr.Column():
                    with gr.Row():
                        # gr.Markdown() 
                        btn_photo = gr.Button(value="Начать распознавание",)
                        triggerImage = gr.Button(value="Подробнее",)
                with gr.Row():
                    with gr.Tab('Результат обработки'):
                        with gr.Row():
                            with gr.Column():
                                predictImage = gr.Image(type="pil", label="Предсказание модели")
                                cropImage = gr.Image(type="pil", label="Обрезанный фрагмент")
                            with gr.Column():
                                gr.Markdown("""<p align="start"><font size="5px">Что же происходит в данном блоке?<br></p>
                                            <br> <ul>
                                            <li>Детекция номера вагона (верхняя картинка);</li>
                                            <li>Обрезка изображения по bbox от YOLO (нижняя картинка);</li>
                                            <li>Распознание номера вагона (текстовое поле ниже);</li>
                                            </ul></font>""")
                                predictImageClass = gr.Textbox(label="Распознанный номер вагона", placeholder="Здесь будут общие данные по файлу", interactive=False, lines=7)
                                
            
            with gr.Tab('Трекинг номера по видео (дополнительно)'):
                file_video = gr.File(label="Видео", file_types=['.mp4','.mkv'])
                with gr.Column():
                    with gr.Row(): 
                        btn_video = gr.Button(value="Начать распознавание",)
                        triggerVideo = gr.Button(value="Подробнее",)
                with gr.Row():
                    with gr.Tab('Результат обработки'):
                        with gr.Row():
                            predictVideo = gr.Video(label="Обработанное видео", interactive=False)
                            predictVideoClass = gr.Textbox(label="Результат обработки", placeholder="Здесь будут общие данные по файлу", interactive=False, lines=7)

    with gr.Row(): 
        with gr.Row(): 
            btn2 = gr.Button(value="Посмотреть файлы",)
            clr_btn = gr.ClearButton([file_photo, file_video, predictImage, predictVideo, cropImage, predictImageClass,predictVideoClass, ], value="Очистить контекст",)
    
    with gr.Row():
        gr.Markdown("""<p align="center">Выполнил Луняков Алексей, студент ИКБО-04-21</p>""")

    btn_photo.click(photoProcessing, inputs=[file_photo, ], outputs=[predictImage, cropImage, predictImageClass,])
    btn_video.click(videoProcessing, inputs=[file_video, ], outputs=[predictVideo, predictVideoClass,])
    btn2.click(fileOpen)
    triggerImage.click(info_fn)
    triggerVideo.click(info_fn)

demo.launch(allowed_paths=["/assets/"])


    