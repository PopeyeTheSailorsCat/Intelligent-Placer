# Intelligent-Placer:
Требуется по фотографии нескольких предметов, расположенных на светлой горизонтальной поверхности и многоугольнику понять, какие из из предметов,
представленных на фотографии можно поместить в многоугольник, чтобы занять в нем наибольшую площадь.
Многоугольник задается фигурой, нарисованной темным маркером на белом листе бумаги, сфотографированной вместе с предметами.

Intelligent Placer будет оформлен в виде python-библиотеки intelligent_placer_lib, которая поставляется каталогом intelligent_placer_lib с файлом intelligent_placer.py,
содержащим функцию - точку входа def check_image(<path_to_png_jpg_image_on_local_computer>), которая возвращает 11-мерный вектор, со значениями 0\1\2 для 10 компонент, соответствующих предметам в датасете, и с натуральными значениями(включая 0) для 11 компоненты.


# Оценка качества работы
Оценкой качества работы Intelligent Placer будет является незанятая площадь в многоугольнике. На ее основании можно представить различные метрики, самой простой из которых является отношения незанятой площади многоугольника к общей площади(будет давать непокрытый процент).  
Более сложные метрики будут отображать количество незанятой площади и площади неиспользованных фигур.  

# Входные данные:
- На вход алгоритму подается путь к изображению в формате png/jpg.  
- На изображении обязан присутствовать многоугольник на белом листе A4. Предметы на изображении могут отсутствовать.
- Предмет должен полностью присутствовать на изображении. В случае, если присутствует только часть, его классификация не гарантирована.
- В случае, если какой то предмет из датасета присутствует несколько раз, считается что он присутствует один раз. Данное условие подразумевает, что ведется бинарная запись о наличие предмета на изображении. Он либо есть либо нет, количество не сохраняется. Если предмет классифицируется, как не принадлежащий датасету, то он записывается в счетчик. 
- Ожидаются, что все предметы повернуты всегда одной стороной к камере.  
- Предметы не должны перекрывать друг друга на фотографии. В идеале между ними должен присутствовать зазор 1-2 см.  
- Изображение должно быть сделано с точки которая  отклонена не более 30 градусов от нормали поверхности, на которой расположены предметы.  
- На изображении не должны присутствовать пересвеченные и серо-черные области. В идеале свет должен падать на поверхность перпендикулярно(Возможны отклонения соразмерные с отклонением снимка), чтобы минимизировать количество теней на изображении.  
- Толщина линии границы многоугольника не должна быть меньше 2px и превосходить 10px на изображении.  
- Минимальный размер изображения - 512 на 256.  


# Выходные данные:
11 мерный вектор, элементами которого являются классы, присвоенные 10 изначальным объектам и всему остальному. Каждому объекту соответствует компонента вектора.
"Всё остальное" включает себя любые объекты, которые не были распознаны в результате классификации - они не будут участвовать в задачи оптимизации.[Данного примера нет в примах входных данных. TODO]. Потенциально для "всё остальное" значения в векторе выходных данных будут представлять количество неопознанных предметов на фотографии.
- 0 - объект отсутствует на изображении.  
- 1 - объект присутствует, но не влезает в многоугольник при желании добиться максимальной площади покрытия.  
- 2 - объект присутствует и влезает в многоугольник в рамках достижения максимальной площади покрытия.  
В случае необходимости, вектор можно расширить для получения дополнительной информации из алгоритма. Например если мы захотим получать координаты и ориентацию объектов, которые попали в многоугольник.

# Задачи, решаемые в процессе:
Алгоритм должен последовательно решать несколько задач  
1) Задача детекции предметов - обнаружить где на изображении предметы и поместить их в bounding box  
2) Задача детекции многоугольника - обнаружить где многоугольник и поместить его в bounding box  
3) Классификация обнаруженных предметов - На этапе 1 были обнаружены все предметы. Последовательно подавать их на классификатор(например CNN)  
Получать информацию о том, какие предметы присутствуют на изображении. Компоненты отсутствующих предметов можно занулить.  
4) Задача распознавания формы многоугольника -  На этапе 2 был обнаружен многоугольник, нужно получить его геометрическую структуру.  
5)Задача "максимально упаковки прямоугольника" - Зная геометрическую структуру многоугольника и структуру предметов решить задачу геометрической  
оптимизации.

Скорее всего задача 5 будет решаться через триангуляцию многоугольника, и уже готовую триангуляцию предметов. Но этот раздел я пока не обдумывал.

Изображение white.jpg представляет собой белую плоскость, с листом А4 на ней  для масштаба.   
Остальные изображения - примеры выбранных мною объектов.   
Они все не слишком большого размера, чтобы можно было больше помещать в многоугольники.

# Предметы
## Ручка  
![Ручка](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/imgs/pen.jpg)  
## Крышка
![Крышка](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/imgs/cap.jpg)  
## Кубик
![Кубик](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/imgs/dice.jpg)  
## Медиатор
![Медиатор](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/imgs/mediator.jpg)  
## Зажигалка
![Зажигалка](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/imgs/lighter.jpg)  
## Значок
![Значок](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/imgs/badge.jpg)   
## Пульт
![Пульт](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/imgs/controller.jpg)  
## Конь
![Конь](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/imgs/horse.jpg)  
## Батарейка
![Батарейка](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/imgs/battery.jpg)  
## Ножницы
![Ножницы](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/imgs/scissors.jpg)  

# Поверхность для съемки
![Поверхность](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/imgs/white.jpg)  

# Показательные примеры входных данных
1) Многоугольник без фигур 
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/inputs_example/1.jpg)  
Проверяет случай, когда на изображении нет фигур.  
Данный случай является требованием.

2) Многоугольник с фигурой, которая не влезает
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/inputs_example/2.jpg)  
Проверяет возможность базового определения, что фигура не влезет.
Данный случай является  тривиальным вариантом работы.  

3) Многоугольник с фигурой, которая влезает
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/inputs_example/3.jpg)  
Проверяет возможность базового определения, что фигура влезет.
Данный случай является тривиальным вариантом работы.  

4) Многоугольник с двумя фигурами, не влезает обе.
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/inputs_example/4.jpg)  
Проверяет возможность нахождения двух различных фигур и определения, что они обе не влезут.

5) Многоугольник с двумя фигурами, поместится может только одна
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/inputs_example/5.jpg)  
Проверяет возможность нахождения двух различных фигур и определения, какая из них влезет.

6) Многоугольник с двумя фигурами, поместится могут обе
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/inputs_example/6.jpg)  
Проверяет возможность нахождения двух различных фигур и определения, как их обеих расположить в многоугольнике.
Данный случай делает упор на решении задачи геометрической оптимизации.  
 
7) Многоугольник с двумя фигурами, вместится может только одна, нужно выбрать с максимальной площадью
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/inputs_example/7.jpg)  
Проверяет возможность нахождения двух различных фигур и определения, какая из них покроет большую площадь.
Данный случай делает упор на решении задачи геометрической оптимизации.  

8) Многоугольник с более чем двумя фигурами, ни одна не влезает в фигуру
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/inputs_example/8.jpg)  
Проверяет нахождения произвольного количества фигур и определения, что ни одна не влезет.
Данный случай является тривиальным при произвольном количестве фигур.

9) Многоугольник с более чем двумя фигурами, влезает только одна
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/inputs_example/9.jpg)  
Проверяет нахождения произвольного количества фигур и определения какая из них может влезть.

10) Многоугольник с более чем двумя фигурами, влезают две.
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/inputs_example/10.jpg)  
Проверяет нахождения произвольного количества фигур и определения какие две из них можно расположить в многоугольнике.
Данный случай делает упор на решении задачи геометрической оптимизации.  

11) Многоугольник с более чем двумя фигурами, влезают все.
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/inputs_example/11.jpg)  
Проверяет нахождения произвольного количества фигур и определения как их все можно расположить в многоугольнике.
Данный случай делает упор на решении задачи геометрической оптимизации.  

12) Многоугольник с более чем двумя фигурами, нужно выбрать максимальных по площади.
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/main/inputs_example/12.jpg)  
Проверяет нахождения произвольного количества фигур и определения какие из них дадут максимальную площадь покрытия, если не все можно расположить в многоугольнике.
Данный случай делает упор на решении задачи геометрической оптимизации.  



# Этапы работы  
## Детекция объектов и многоугольника  
Посмотрю, как на избражении сработают различные пороги:
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/imgs/all_threshold.png)  
На этом примере видно, что минимальный порог выхватывает только объекты на изображении, продолжим работу с ним. Выделим на изображении различные компоненты связанности,
чтобы можно было избавиться от различных шумов по значению площади. Нарисуем bbox вокруг полученных площадей. 
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/imgs/bbox_example.png)  
Посмотрим как выглядит полученный результат на исходном изображении.  
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/imgs/bbox_result.png)  

Результат неплохой, однако нужно будет дописать обработку случая, когда не найдены предметы и избавиться от примера, где засвечен угол экрана.

Теперь попробуем найти на изображении многоуольник.
Воспользуемся детектором границ Canny.  
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/imgs/canny_example_edges.png)  

Добавим пост обработку - воспользуемся данными о найденных предметах, чтоб закрасить их bboxами область вокруг них. Кроме того, сделаем закрытие,
вместе с заполнением. После этого пройдемся открытием, чтобы избавиться от всего, что не представляет собой геометрическую фигуру(полосы).  
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/imgs/figure_detection_example.png)  
После чего применим к изображению поиск компонент и получим bbox объекта.  
Итого результат детекции:
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/imgs/Detection_example.png)  

Результат выглядит хорошим, хотя нужно обработать случай, когда нет объектов.


## Классификация объектов на изображении
Создадим небольшой датасет изображений предметов, где они сняты с небольшими отклонениями от нормали, под разным углом на изображении и чуть-чуть различым освещением:
Все изображения обработаем с помощью поиска и вырезания объектов на изображении.
Примеры данных для обучения:  
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/classifyer_imgs/2_cut/16_1.jpg)  
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/classifyer_imgs/5_cut/22_0.jpg)  
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/classifyer_imgs/8_cut/5_0.jpg)  
Полученный датасет подадим на вход сверточной нейросети и обучим ее.
График обучения:
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/imgs/class_train.png)  
Проверим полученный результат на примерах постановки задачи:
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/imgs/class_example_1.PNG)  
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/imgs/class_example_2.PNG)  
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/imgs/class_example_3.PNG)  

## Получение структуры фигуры
С помощью bbox фигуры получим участок, где она располагается и с помощью простейшего поиска и заполнение получим маску фигуры.


## Итог на под-этапе:
Мы можем находить и классифицировать на изображении объекты, и получать структуру фигуры.  
Пример работы:
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/imgs/class_and_figure.PNG)  
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/imgs/class_and_figure_2.PNG)  

## Получение структур объектов
Сделаем максимально близко к нормали фотоснимки объектов, на расстоянии, которое будет наиоболее ожидаемым в процессе работы. Полученный результат будем использовать после классификации, как структуру классифицированного объекта:

![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/objects_figure/2.jpg)  
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/objects_figure/5.jpg)  


## Размещение объектов внутри фигуры
Максимально переборно сделанный пункт. Бегаю маской объекта  по маске фигуры, если бинарное пересечение по количеству ячеек совпадает с площадью структуры объекта, то мы на часть изображения фигуры помещаем данный объект. И так далее. Пример работы этого в общем примере работы на данном этапе

## Пример работы на данном этапе
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/imgs/epoch_example_1.PNG)  
Нашли на изображении объекты и фигуру. Получили структуру фигуры.  
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/imgs/epoch_example_2.PNG)  
Классифицировали объекты, достали их структуру из файлов.  
![Подпись](https://github.com/PopeyeTheSailorsCat/Intelligent-Placer/raw/develop/imgs/epoch_example_3.PNG)  
Разместили их в объекте, отобразили размещение на исходном изображении.
## Что сделать к следующим версиям:
* Добавить ножницы и ручку в датасет для нейросети
* Избавить от плохих фото
* Обработать случай отсутствия предметов
* Избавиться от элементво, зависящих от расширения фотографии
* Сделать размещение объектов внутри более продвинутым
* Оформить выход в виде заявленного
* Избавиться от констант в коде <- Можно и на ревью это сделать
# В общем то можно все это сделать и на данном этапе, но времени НЕТ СОВСЕМ
