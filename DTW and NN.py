import os
import soundfile as sf
import numpy as np
import librosa.feature as ftr
import matplotlib.pyplot as plt
import scipy as sc
import librosa.core as core
import sys
import random as rnd
import librosa.sequence as seq
import timeit
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf

dictionary=['Книги', 'Фильмы', 'Музыка', 'Фотографии', 'Документы', 'Проекты', 'Дистрибутивы', 'Загрузки', 'Игры']

path="C:\\Users\\Maksim Korotchenkov\\PycharmProjects\\MIAODI\\venv\\files\\sample\\"#путь к записям
files=os.listdir(path)#массив названий файлов, лежащих в папке по пути path
print(files)

frame_time=0.02#Длительность кадра
frame_shift=0.5#Коэффициент перекрытия кадров (сдвиг кадров друг относительно друга)
n_mfcc=9 #Количество МЧКК в кадре
dct_type=2 #Тип дискретного косинусного преобразования
norm = 'ortho'
epochs=1000
optimizer='adam' # алгоритм обучения
activation = "relu"

f_ind = 0
all_data = []
dataGraph = []
word_count=[]

for w_ind in range(len(dictionary)):
    filename=dictionary[w_ind]
    w_count=0
    f_ind_array=[]
    for f in range(len(files)):
        if files[f].find(filename)>=0:
            fname=path+files[f]   #сформировали путь к голосовым командам
            # print(fname)
            #откроем голосовые команды
            y_int, sr= sf.read(file=fname, #путь к нарезанной голосовой команде
                               dtype='int16') #формат отсчетов-целочисленные значения 16 бит)
            #файлы были открыты и загружены в массив y_int и частота дискретезации записана в переменную sr
#разобьем эти файлы на кадры и внутри кадров посчитаем мел-частотные кепстральные коэффициенты
            #функция mfcc требует не целочисленные а float значения и сам у должен быть в массиве numpy, поэтому сделаем y_int типо float
            y=np.array(y_int, dtype='float32')

            frameWidth=int(frame_time*sr) #посчитаем ширину кадра в количестве отсчетов
            frameShift=int(frame_shift*frameWidth) #посчитаем на сколько отсчетов происходит сдвиг
            df = frameWidth - frameShift #найдем число не перекрытых отсчетов
            frameCount = int(len(y) / df) - 1
            frameCount2=int(frameWidth / df) * (int(len(y) / frameWidth) - 1) + 1 # Посчитаем число кадров с учетом сдвига

            wnd=sc.hanning(frameWidth) #окно Ханна
            # wnd = sc.hamming(frameWidth) #окно Хемминга
            # wnd = sc.parzen(frameWidth)
            # wnd = sc.bartlett(frameWidth) #окно Бартлетта

# записать в массив Mfccs мел частотные кепстральные коэффициенты
            Mfccs=ftr.mfcc(y=y,sr=sr,n_mfcc=n_mfcc,
                           hop_length=df,
                           window=wnd, #укажем вид окна- Ханнинга
                           win_length=frameWidth, #указали ширину окна
                           dct_type=dct_type
                           )

# #МЧКК в другом виде
            Mfccs = Mfccs.ravel(order='F')

# #расчет МЧКК не разбивая на кадры (для всей реализации в в целом) (используя метод core)
#
            S = core.power_to_db(ftr.melspectrogram(y=y, sr=sr
                                                    , hop_length=len(y) + 1
                                                    , win_length=frameWidth
                                                    , window=wnd
                                                    )) #расчет мощности
            M = sc.fftpack.dct(S, axis=0, type=dct_type, norm=norm)[:n_mfcc] #МЧКК и дискретно косинусное преобразование

#запишем все реализации в единый массив (последовательно, так как они обрабатываются)
            all_data.append({'n': w_ind, #в поле n будем закладывать индекс слова
                             'filename': filename, #в поле filename будем закладывать название команды
                             'fname': fname, #в поле n будем закладывать путь к файлу(одна реализация одного слова)
                             'mfccs_fr': Mfccs, #в поле mfccs_fr будем закладывать Mfccs
                             'mfcc': M.ravel()[2:n_mfcc]}) #в поле mfcc будем закладывать М в определенном фаормате
            f_ind_array.append(f_ind) #порядковый номер реализации (файла) с которым мы работаем
            f_ind += 1
            w_count += 1 #количество реализаций (одного слова-команды)
            # break
    word_count.append({'fcount': w_count, 'findexes': f_ind_array}) # данный массив содержит в первом поле количество повторений, а во втором- уникальные индексы порядка слов

#создание обучающего и тестового массивов для алгоритма DTW
train_data = []
test_data = []
tr_count = 0.7 #проборции обучающей и тестовой выбоки (обучающая выборка содержит 70 процентов исходных данных)
tst_count = 0.3
#заполнение этих двух массивов
for w_ind in range(len(dictionary)):
    filename = dictionary[w_ind]

    fcount = word_count[w_ind]['fcount'] #в эту переменную закладываем количествуо реализаций данного слова (сколько раз произносили)
    findexes = word_count[w_ind]['findexes'] #в эту переменную закладываем массив с индексами файлов

    for j in range(int(fcount * tr_count)): #пробегаем от 0 до количества реализаций слова (обычно 10)*0,7
        f_ind = rnd.choice(findexes) #выбираю рандомный элемент из массива индексов
        train_data.append(all_data[f_ind]) #в массив train_data добавляю исходные данные о МЧКК с выбранным индексом
        findexes.remove(f_ind) #удаляем данный индекс, чтобы он не попал в тестовую часть. В итоге в обучающей выборке будет по 10*0,7 рандомных реализаций слова

    for j in range(int(fcount * tst_count)): #точно такой же цикл, колько заполняем тестовые данные
        f_ind = rnd.choice(findexes)
        test_data.append(all_data[f_ind])
        findexes.remove(f_ind) #В итоге в тестовой выборке будет по 10*0,3 рандомных реализаций слова

#алгоритм DTW (Поиск расстояний между последовательностями МЧКК, которые попали в тестовый массив и последовательностями МЧКК, которые попали в обучающий массив)
t1 = timeit.default_timer()
res = []
for i in range(len(test_data)):
    # Y = test_data[i]['mfcc']
    Y = test_data[i]['mfccs_fr'] # МЧКК по кадрам из тестовой части
    min_dist = np.inf # константа бесконечности
    min_j = -1
    for j in range(len(train_data)): #выбранная голосовая команда из тестовых данных (выше) сравнивается со всеми голосовыми командами из обучающей выборки
        # X = train_data[j]['mfcc']
        X = train_data[j]['mfccs_fr']

        D, wp = seq.dtw(X, Y) # динамическая трансформация шкалы времени: получаем матрицу расстояний D и матрицу координат тех элементов матрицы расстояний, которые формируют оптимальный путь трансформации
        gc = 0

        for k in range(len(wp)):
            gc = gc + D[wp[k][0]][wp[k][1]] # расчет оптимального пути трансформации GC (см лекцию)
        gc = gc / len(wp)
        # gc = fd.fastdtw(X,Y, 10)[0]
        # print(gc)

        if min_dist > gc: #поиск минимального значения GC
            min_dist = gc
            min_j = j #сохраняем индекс реализации для которой найдено минимальное расстояние
    res.append({'find': min_j, 'dist': min_dist}) #В res будет записываться индекс реализации с минимальным расстоянием и само минимальное расстояние
    print('Обрабатывается: ' + str(i) + ' из ' + str(len(test_data)))
t2 = timeit.default_timer()
print('Время обработки с помощью алгоритма DTW', t2-t1)
# print(res)

errors = 0
for k in range(len(res)):
    cl = train_data[res[k]['find']]['filename'] # метки полученные из обучающей части
    est = test_data[k]['filename'] # метки из тестовой части

    errStr = ''
    if est != cl:
        errors += 1
        errStr = ' error'

    print(str(res[k]['dist']) + ' ' + cl + ' ' + est + '  ' + errStr)
print("Точность метода DTW", 1-errors / len(res))

sys.exit(1)

# NN (подготовка данных для работы с ней)
mfcc_count = 0
for f_ind in range(len(all_data)):
    if len(all_data[f_ind]['mfccs_fr']) > mfcc_count:
        mfcc_count = len(all_data[f_ind]['mfccs_fr'])
data = np.ndarray(shape=[len(all_data), mfcc_count])
print(data.shape)
labels = np.ndarray(shape=[len(all_data), 1])
for i in range(len(data)):
    labels[i] = all_data[i]['n']
    for j in range(len(all_data[i]['mfccs_fr'])):
        data[i,j] = all_data[i]['mfccs_fr'][j]

#архитектура нейронной сети-на выходе столько нейронов, сколько классов (слов) неообходимо распозновать из словаря,
# такая архитектура решает задачу классификации лучше,чем с одним нейроном на выходе
lb = LabelBinarizer() # некоторое преобразование labels (см 2:17) к виду меток класса,где каждая метка представляет
# собой вектор состоящий из ноликов и единичек,стоящих на той позиции, номер класса которого кодирует вектор
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42) #разбиение двух массивов на обучающую и тестовую части

count_neurons_in_input_dence=len(trainX[0])#количество нейронов во входном слое

t5 = timeit.default_timer()
model = tf.keras.Sequential() #инициализируем модель персепрона

neuron_in_layer=[frame_shift,optimizer,activation,epochs,n_mfcc,count_neurons_in_input_dence, 800,0.8]

model.add(tf.keras.layers.Dense(neuron_in_layer[6], activation = activation, input_shape=(len(trainX[0]) #-количество нейронов во входном слое
                                                                          , ))) #добавляем в модель слои-входной

# слой имеет столько нейронов, сколько МЧКК во входном массиве
model.add(tf.keras.layers.Dropout(neuron_in_layer[7], noise_shape=None, seed=None))
model.add(tf.keras.layers.Dense(len(dictionary), activation = 'sigmoid')) #метод Dense создает 1 полносвязный слой персептрона
model.summary()

# compiling the model
model.compile( # компиляция модели сети
 optimizer = optimizer, #алгоритм обучения (способ оценки ошибки классификации)
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)

results = model.fit( # запуск процедуры обучения
 trainX, trainY,
 epochs=epochs,
 batch_size=128,
 shuffle=False,
 verbose=2,
 validation_data=(testX, testY)
)
print("Test-Accuracy:", np.mean(results.history["val_accuracy"]))
t6= timeit.default_timer()
# print(results)

metrics = results.history # массив содержащий информацию как изменялась ошибка обучения и ошибка валидации

plt.figure()
plt.plot(results.epoch, metrics['loss'], metrics['val_loss'])# построение графиков (ошибка обучения от количества эпох) и (ошибки валидации от количества эпох)
plt.legend(['train_loss', 'val_loss'])
plt.show()


t3 = timeit.default_timer()
y_pred = np.argmax(model.predict(testX), axis=1) # получение массива предсказанных меток на тестовом наборе данных
t4 = timeit.default_timer()
y_true = testY

labelsY = lb.inverse_transform(testY, 0.5) # обратное преобразование векторов нулей и единичек в массив номеров позиций
labelsY = np.array(labelsY, dtype='int16') # привод к целочисленному виду
print('Метки расставленные экспертом',labelsY)

test_acc = sum(y_pred == labelsY) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')
print('Время на классификацию тестового набора многослойным персептроном', t4-t3)
print('Время на инциализацию модели и обучение персептрона', t6-t5)
test_accur=int(test_acc)


pathNW='C:\\Users\\Maksim Korotchenkov\\PycharmProjects\\MIAODI\\NETWORKS\\'

for i in range(len(neuron_in_layer)):
    a=str(neuron_in_layer[i])
    pathNW=pathNW+'_'+a
pathNW=pathNW+'_'+str(round(test_acc*100))+'%'+'.hdf5'
# print(pathNW)
model.save(pathNW)
