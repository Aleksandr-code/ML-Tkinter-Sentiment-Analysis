import tkinter as tk
import tkinter.filedialog as fd 
from tkinter import ttk
import tkinter.messagebox as mb

import Tokenization
import RemoveStopWords
import Stemming
import Lemmatization
import SplitAndLabelsData
import Vectorization
import DictAndLibrary
import pandas as pd

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import joblib
from pathlib import Path
from functools import partial

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('Конфигуратор моделей для анализа тональности текста')
        self.iconbitmap(default="favicon.ico")
        self.width = 800
        self.heigh = 500
        self.geometry(f"{self.width}x{self.heigh}")
        self.resizable(width=False, height=True)
        self.load_data = None
        self.data_column_var = tk.IntVar()
        self.data_column_var.set(1)
        self.sentiment_column_var = tk.IntVar()
        self.sentiment_column_var.set(1)
        self.vectorizer_model = None
        self.preload_model = None
        

        #Cтили для элементов
        style_button_select_all = ttk.Style()
        style_button_select_all.configure('SA.TButton', font="helvetica 10", padding=5)
        style_button = ttk.Style()
        style_button.configure('TButton', font="helvetica 12 bold", padding=10)     

        #Создание набора вкладок
        notebook = ttk.Notebook()
        notebook.pack(expand=True, fill='both')

        #Создание общих фреймов - Вкладка Создание модели / Вкладка тестирование модели
        tab1_frame = ttk.Frame(notebook)
        tab2_frame = ttk.Frame(notebook)

        tab1_frame.pack(fill='both', expand=True)
        tab2_frame.pack(fill='both', expand=True)

        notebook.add(tab1_frame, text="Разработка модели")
        notebook.add(tab2_frame, text="Тестирование модели")

        #Cкрол для 1 вкладки
        self.canvas_tab1 = tk.Canvas(tab1_frame)
        self.canvas_tab1.pack(side='left', fill='both', expand=True)

        scrollbar_tab1 = ttk.Scrollbar(tab1_frame, orient=tk.VERTICAL, command=self.canvas_tab1.yview)
        scrollbar_tab1.pack(side='right', fill='y')

        self.canvas_tab1.configure(yscrollcommand=scrollbar_tab1.set)
        self.canvas_tab1.bind('<Configure>', lambda e: self.canvas_tab1.configure(scrollregion=self.canvas_tab1.bbox('all')))

        tab1_inside_frame = ttk.Frame(self.canvas_tab1)
        self.canvas_tab1.create_window((0,0), window=tab1_inside_frame, anchor='nw')

        tab1_inside_frame.bind('<Enter>', self._bound_to_mousewheel)
        tab1_inside_frame.bind('<Leave>', self._unbound_to_mousewheel)

        #Блок - Загрузка файла
        btn_file = ttk.Button(tab1_inside_frame, text="Выбрать файл",
                             command=self.choose_file, takefocus='0',   
                             style='TButton')
        #btn_file.grid(row='0', column='0', columnspan=2, sticky='n', padx=10, pady=10)
        btn_file.pack(padx=0, pady=5, fill='x')

        self.frame_load_data = tk.Frame(tab1_inside_frame, height=1)
        self.frame_load_data.pack(padx=0, pady=5, fill='x')

        #Блок токенизации
        frame_tokenize = tk.Frame(tab1_inside_frame, height=200)
        frame_tokenize.pack(padx=0, pady=5, fill='x')

        l_tokenize = ttk.Label(frame_tokenize, text='Выберите метод токенизации',
                              font='Helvetica 12 bold')
        l_tokenize.grid(row='0', column='0', columnspan='2', sticky='w', padx=10, pady=10)

        # self.tokenizers = {
        #     1 : 'Split',
        #     2 : 'Word_tokenize',
        #     3 : 'WordPunctTokenizer',
        #     4 : 'Razdel',
        #     5 : 'TweetTokenizer',
        #     6 : 'Spacy',
        #     7 : 'Re.findall(\w+|\d+|\p+)'
        # }

        self.tokenize_var = tk.IntVar()
        self.tokenize_var.set(1)

        radio_tokenize_split = ttk.Radiobutton(frame_tokenize, text='Split', variable=self.tokenize_var, takefocus='0', value=1)
        radio_tokenize_split.grid(row='1', column='0', padx=10, pady=10, sticky='w')

        radio_tokenize_word = ttk.Radiobutton(frame_tokenize, text='Word_tokenize', variable=self.tokenize_var, takefocus='0', value=2)
        radio_tokenize_word.grid(row='1', column='1', padx=10, pady=10, sticky='w')

        radio_tokenize_wordPunct = ttk.Radiobutton(frame_tokenize, text='WordPunctTokenizer', variable=self.tokenize_var, takefocus='0', value=3)
        radio_tokenize_wordPunct.grid(row='2', column='0', padx=10, pady=10, sticky='w')

        radio_tokenize_razdel = ttk.Radiobutton(frame_tokenize, text='Razdel', variable=self.tokenize_var, takefocus='0', value=4)
        radio_tokenize_razdel.grid(row='2', column='1', padx=10, pady=10, sticky='w')

        radio_tokenize_tweet = ttk.Radiobutton(frame_tokenize, text='TweetTokenizer', variable=self.tokenize_var, takefocus='0', value=5)
        radio_tokenize_tweet.grid(row='3', column='0', padx=10, pady=10, sticky='w')

        # radio_tokenize_spacy = ttk.Radiobutton(frame_tokenize, text='Spacy', variable=self.tokenize_var, takefocus='0', value=6)
        # radio_tokenize_spacy.grid(row='2', column='2', padx=10, pady=10, sticky='w')
        
        radio_tokenize_re = ttk.Radiobutton(frame_tokenize, text='Re.findall(\w+|\d+|\p+)', variable=self.tokenize_var, takefocus='0', value=6)
        radio_tokenize_re.grid(row='3', column='1', padx=10, pady=10, sticky='w')
        
        #Блок нормализации (очистка излишней информации)
        frame_normalize = tk.Frame(tab1_inside_frame, height=200)
        frame_normalize.pack(padx=0, pady=10, fill='x')

        l_normalize = ttk.Label(frame_normalize, text='Выберите метод / методы нормализации                                                                                                         ',
                              font='Helvetica 12 bold')
        l_normalize.grid(row='0', column='0', columnspan='2', sticky='w', padx=10, pady=10)

        # self.normalize_var = tk.IntVar()
        
        self.check_strip_html_var = tk.IntVar()
        self.check_strip_html = ttk.Checkbutton(frame_normalize, variable=self.check_strip_html_var, text='Удаление html разметки', takefocus='0')
        self.check_strip_html.grid(row='1', column='0', padx=10, pady=10, sticky='w')

        self.check_square_brackets_var = tk.IntVar()
        self.check_square_brackets = ttk.Checkbutton(frame_normalize, variable=self.check_square_brackets_var, text='Удаление квадратных скобок', takefocus='0')
        self.check_square_brackets.grid(row='1', column='1', padx=10, pady=10, sticky='w')

        self.check_special_characters_ru_var = tk.IntVar()
        self.check_special_characters_ru = ttk.Checkbutton(frame_normalize, variable=self.check_special_characters_ru_var, text='Удаление специальных символов "russian"', takefocus='0')
        self.check_special_characters_ru.grid(row='2', column='0', padx=10, pady=10, sticky='w')

        self.check_special_characters_en_var = tk.IntVar()
        self.check_special_characters_en = ttk.Checkbutton(frame_normalize, variable=self.check_special_characters_en_var, text='Удаление специальных символов "english"', takefocus='0')
        self.check_special_characters_en.grid(row='2', column='1', padx=10, pady=10, sticky='w')

        self.check_stop_words_var = tk.IntVar()
        self.check_stop_words = ttk.Checkbutton(frame_normalize, variable=self.check_stop_words_var, text='Удаление стоп-слов "english"', takefocus='0')
        self.check_stop_words.grid(row='3', column='0', padx=10, pady=10, sticky='w')

        btn_select_all = ttk.Button(frame_normalize, text="Выбрать все", command=self.normalize_select_all, takefocus='0', style='SA.TButton')
        btn_select_all.grid(row='4', column='0', padx=10, pady=10, sticky='w')

        #Блок стемминга
        frame_stemming = tk.Frame(tab1_inside_frame, height=200)
        frame_stemming.pack(padx=0, pady=5, fill='x')

        l_stemming = ttk.Label(frame_stemming, text='Выберите метод стемминга',
                              font='Helvetica 12 bold')
        l_stemming.grid(row='0', column='0', sticky='w', padx=10, pady=10)

        self.stemming_var = tk.IntVar()
        self.stemming_var.set(1)

        radio_stemming_null = ttk.Radiobutton(frame_stemming, text='Ничего не выбрано', variable=self.stemming_var, takefocus='0', value=1)
        radio_stemming_null.grid(row='1', column='0', padx=10, pady=10, sticky='w')

        radio_stemming_porter = ttk.Radiobutton(frame_stemming, text='Стеммер Портера', variable=self.stemming_var, takefocus='0', value=2)
        radio_stemming_porter.grid(row='2', column='0', padx=10, pady=10, sticky='w')

        radio_stemming_snowBall = ttk.Radiobutton(frame_stemming, text='Стеммер SnowBall', variable=self.stemming_var, takefocus='0', value=3)
        radio_stemming_snowBall.grid(row='3', column='0', padx=10, pady=10, sticky='w')

        radio_stemming_lancaster = ttk.Radiobutton(frame_stemming, text='Стеммер Ланкастера', variable=self.stemming_var, takefocus='0', value=4)
        radio_stemming_lancaster.grid(row='4', column='0', padx=10, pady=10, sticky='w')

        #Блок лемматизации
        frame_lemmatization = tk.Frame(tab1_inside_frame, height=200)
        frame_lemmatization.pack(padx=0, pady=5, fill='x')

        l_lemmatization = ttk.Label(frame_lemmatization, text='Выберите метод лемматизации',
                              font='Helvetica 12 bold')
        l_lemmatization.grid(row='0', column='0', sticky='w', padx=10, pady=10)

        self.lemmatization_var = tk.IntVar()
        self.lemmatization_var.set(1)

        radio_lemmatization_null = ttk.Radiobutton(frame_lemmatization, text='Ничего не выбрано', variable=self.lemmatization_var, takefocus='0', value=1)
        radio_lemmatization_null.grid(row='1', column='0', padx=10, pady=10, sticky='w')

        radio_lemmatization_wordNet = ttk.Radiobutton(frame_lemmatization, text='Лемматизатор WordNet', variable=self.lemmatization_var, takefocus='0', value=2)
        radio_lemmatization_wordNet.grid(row='2', column='0', padx=10, pady=10, sticky='w')


        radio_lemmatization_pyMorphy = ttk.Radiobutton(frame_lemmatization, text='Лемматизатор pyMorphy2', variable=self.lemmatization_var, takefocus='0', value=3)
        radio_lemmatization_pyMorphy.grid(row='3', column='0', padx=10, pady=10, sticky='w')

        #Блок векторизации
        frame_vectorazation = tk.Frame(tab1_inside_frame, height=200)
        frame_vectorazation.pack(padx=0, pady=5, fill='x')

        l_vectorazation = ttk.Label(frame_vectorazation, text='Выберите метод векторизации', 
                              font='Helvetica 12 bold')
        l_vectorazation.grid(row='0', column='0', columnspan='2', sticky='w', padx=10, pady=10)

        self.vectorazation_var = tk.IntVar()
        self.vectorazation_var.set(1)

        radio_vectorazation_bow = ttk.Radiobutton(frame_vectorazation, text='Bag of words', variable=self.vectorazation_var, takefocus='0', value=1)
        radio_vectorazation_bow.grid(row='1', column='0', padx=10, pady=10, sticky='w')

        radio_vectorazation_tfidf = ttk.Radiobutton(frame_vectorazation, text='TF-IDF', variable=self.vectorazation_var, takefocus='0', value=2)
        radio_vectorazation_tfidf.grid(row='1', column='1', padx=10, pady=10, sticky='w')

        #Блок выбора машинного метода для тренировки данных
        frame_machine_learning = tk.Frame(tab1_inside_frame, height=200)
        frame_machine_learning.pack(padx=0, pady=5, fill='x')

        l_machine_learning = ttk.Label(frame_machine_learning, text='Выберите метод машинного обучения',
                              font='Helvetica 12 bold')
        l_machine_learning.grid(row='0', column='0', columnspan='2', sticky='w', padx=10, pady=10)

        self.ml_var = tk.IntVar()
        self.ml_var.set(1)

        radio_logistic_regression = ttk.Radiobutton(frame_machine_learning, text='Логистическая регрессия', variable=self.ml_var, takefocus='0', value=1)
        radio_logistic_regression.grid(row='1', column='0', padx=10, pady=10, sticky='w')

        radio_naive_bayes = ttk.Radiobutton(frame_machine_learning, text='Наивный байесовский классификатор', variable=self.ml_var, takefocus='0', value=2)
        radio_naive_bayes.grid(row='1', column='1', padx=10, pady=10, sticky='w')

        radio_stochastic_gradient_descent = ttk.Radiobutton(frame_machine_learning, text='Cтохастический градиентный спуск', variable=self.ml_var, takefocus='0', value=3)
        radio_stochastic_gradient_descent.grid(row='2', column='0', padx=10, pady=10, sticky='w')
        
        radio_kNeighbors = ttk.Radiobutton(frame_machine_learning, text='Метод k-ближайших соседей', variable=self.ml_var, takefocus='0', value=4)
        radio_kNeighbors.grid(row='2', column='1', padx=10, pady=10, sticky='w')

        radio_svm = ttk.Radiobutton(frame_machine_learning, text='Метод опорных векторов', variable=self.ml_var, takefocus='0', value=5)
        radio_svm.grid(row='3', column='0', padx=10, pady=10, sticky='w')

        radio_decisionTree = ttk.Radiobutton(frame_machine_learning, text='Деревья решений', variable=self.ml_var, takefocus='0', value=6)
        radio_decisionTree.grid(row='3', column='1', padx=10, pady=10, sticky='w')

        radio_randomForest = ttk.Radiobutton(frame_machine_learning, text='Случайный лес', variable=self.ml_var, takefocus='0', value=7)
        radio_randomForest.grid(row='4', column='0', padx=10, pady=10, sticky='w')

        radio_gradientBoosting = ttk.Radiobutton(frame_machine_learning, text='Градиентный бустинг', variable=self.ml_var, takefocus='0', value=8)
        radio_gradientBoosting.grid(row='4', column='1', padx=10, pady=10, sticky='w')

        radio_mlp = ttk.Radiobutton(frame_machine_learning, text='Многослойный перцептрон', variable=self.ml_var, takefocus='0', value=9)
        radio_mlp.grid(row='5', column='0', padx=10, pady=10, sticky='w')

        #Блок - натренировать модель
        btn_train_model = ttk.Button(tab1_inside_frame, text="Тренировать модель", takefocus='0',
                                     command=self.train_model, style='TButton')
        btn_train_model.pack(padx=0, pady=10, fill='x')

        #Блок - Выровнять колонки в фреймах
        for columns in range(2): 
            self.frame_load_data.columnconfigure(index=columns, weight=1)
            frame_tokenize.columnconfigure(index=columns, weight=1)
            frame_normalize.columnconfigure(index=columns, weight=1)
            frame_vectorazation.columnconfigure(index=columns, weight=1)
            frame_machine_learning.columnconfigure(index=columns, weight=1)

        #Блок - загрузить модель
        btn_load_model = ttk.Button(tab2_frame, text="Загрузить модель", takefocus='0',
                                     command=self.load_model, style='TButton')
        btn_load_model.pack(padx=0, pady=10, fill='x')
        self.l_load_model = ttk.Label(tab2_frame, text='Модель не загружена',
                              font='Helvetica 12 italic', foreground="#B12C49")
        self.l_load_model.pack(padx=10, pady=10, fill='x')

        #Блок - тестирование модели
        frame_test_model = tk.Frame(tab2_frame, height=200)
        frame_test_model.pack(padx=0, pady=5, fill='x')
        # for columns in range(2): frame_test_model.columnconfigure(index=columns, weight=1)

        self.check_library_text_blob_var = tk.IntVar()
        self.check_library_text_blob = ttk.Checkbutton(frame_test_model, variable=self.check_library_text_blob_var, text='Использовать библиотеку Textblob "english"', takefocus='0', command=partial(self.check_btn_library,self.check_library_text_blob_var))
        self.check_library_text_blob.grid(row='0', column='0', padx=10, pady=10, sticky='w')

        self.check_dict_kartaslovsent_var = tk.IntVar()
        self.check_dict_kartaslovsent = ttk.Checkbutton(frame_test_model, variable=self.check_dict_kartaslovsent_var, text='Использовать словарь Kartaslovsent "russian"', takefocus='0', command=partial(self.check_btn_library,self.check_dict_kartaslovsent_var))
        self.check_dict_kartaslovsent.grid(row='0', column='1', padx=10, pady=10, sticky='w')

        l_own_text = ttk.Label(frame_test_model, text='Введите свой текст', font='Helvetica 12 bold')
        l_own_text.grid(row='1', column='0', padx=10, pady=10, sticky='w')
        l_result = ttk.Label(frame_test_model, text='Результат:', font='Helvetica 12 bold')
        l_result.grid(row='1', column='1', padx=10, pady=10, sticky='w')

        self.t_input_text = tk.Text(frame_test_model, font="Consolas", wrap='word',  width=45,  height=7, relief="flat")
        self.t_input_text.grid(row='2', column='0', padx=10, pady=10, sticky='w')
        self.t_input_text.bind("<Control-KeyPress>", self._keypress_text_in_widget)
        self.l_output_result = ttk.Label(frame_test_model, text='Unknown', font='Helvetica 20 bold')
        self.l_output_result.grid(row='2', column='1', padx=10, pady=10, sticky='N')
        
        #self.clipboard_get()

        btn_predict_text = ttk.Button(frame_test_model, text="Классифицировать текст", takefocus='0',
                                     command=self.classify_text, style='TButton')
        btn_predict_text.grid(row='3', column='0', padx=10, pady=10, sticky='w')


    #Text - взаимодействие виджета с буфером обмена
    def _keypress_text_in_widget(self, event):
        if event.keycode == 86:
            event.widget.event_generate('<<Paste>>')
        elif event.keycode == 67:
            event.widget.event_generate('<<Copy>>')
        elif event.keycode == 88:
            event.widget.event_generate('<<Cut>>')
    
    #Checkbuttons - переключение флажка
    def check_btn_library(self, var):
        if var == self.check_library_text_blob_var:
            self.check_dict_kartaslovsent_var.set(0)
        if var == self.check_dict_kartaslovsent_var:
            self.check_library_text_blob_var.set(0)

    #Scrollbar - прокрутка мышью
    def _bound_to_mousewheel(self, event):
        self.canvas_tab1.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbound_to_mousewheel(self, event):
        self.canvas_tab1.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.canvas_tab1.yview_scroll(int(-1*(event.delta/120)), "units")
    
    #Загрузка файла
    def choose_file(self):
        filetypes = (("Текстовый файл", "*.csv; *.tsv"),
                     ("Любой", "*"))
        self.filename = fd.askopenfilename(title="Открыть файл", initialdir="/",
                                      filetypes=filetypes)
        # self.focus()
        if self.filename:
            #print(self.filename)
            if Path(self.filename).stem == 'test':
                load_data=pd.read_csv(self.filename, sep='\t')
            else: 
                load_data=pd.read_csv(self.filename)
            self.load_data = load_data

            #Очистка данных в фрейме выбора столбцов данных и сентимента
            for widget in self.frame_load_data.winfo_children():
                widget.destroy()
            #Добавление виджетов в фрейм выбора столбцов данных и сентимента
            i = 1
            j = 1
            ttk.Label(self.frame_load_data, text='Выберите столбец с данными',
                        font='Helvetica 10 bold').grid(row='1', column='0', sticky='w', padx=10, pady=10)
            for dataColumn in self.load_data.columns.values.tolist():
                ttk.Radiobutton(self.frame_load_data, text=dataColumn, 
                               variable=self.data_column_var, takefocus='0', value=i).grid(row=i+1, column='0', padx=10, pady=10, sticky='w')
                i += 1
            ttk.Label(self.frame_load_data, text='Выберите столбец с метками тональности',
                        font='Helvetica 10 bold').grid(row='1', column='1', sticky='w', padx=10, pady=10)
            for sentimentColumn in self.load_data.columns.values.tolist():
                ttk.Radiobutton(self.frame_load_data, text=sentimentColumn, 
                               variable=self.sentiment_column_var, takefocus='0', value=j).grid(row=j+1, column='1', padx=10, pady=10, sticky='w')
                j += 1
            #Изменяем размер окна после появления выбора столбцов
            self.heigh +=1
            self.geometry(f"{self.width}x{self.heigh}")
            print('OK')
            
            # print(load_data.shape)
            # print(load_data.head(10))
    
    #Блок нормализации - Выбрать всё
    def normalize_select_all(self):
        # for check in [self.check_strip_html, self.check_square_brackets, 
        #               self.check_special_characters, self.check_stop_words]:
        #     check.select()
        self.check_strip_html_var.set(1)
        self.check_square_brackets_var.set(1)
        self.check_special_characters_ru_var.set(1)
        self.check_special_characters_en_var.set(1)
        self.check_stop_words_var.set(1)

    #Натренировать модель
    def train_model(self):
        if (self.load_data is None):
            return mb.showwarning("Предупреждение", 'Файл данных не загружен')
        load_data_columns = self.load_data.columns.values.tolist()
        target_data_index = load_data_columns[self.data_column_var.get()-1]
        sentiment_data_index = load_data_columns[self.sentiment_column_var.get()-1]
        print(target_data_index)
        print(sentiment_data_index)
        print(self.load_data.head(10))
        # print(self.data_column_var.get())
        # print(self.sentiment_column_var.get())
        print('check_strip_html ', self.check_strip_html_var.get())
        if self.check_strip_html_var.get() == 1:
            self.load_data[target_data_index] = self.load_data[target_data_index].apply(RemoveStopWords.strip_html)
            print(self.load_data[target_data_index][:10])
        print('check_square_brackets ', self.check_square_brackets_var.get())
        if self.check_square_brackets_var.get() == 1:
            self.load_data[target_data_index] = self.load_data[target_data_index].apply(RemoveStopWords.remove_between_square_brackets)
            print(self.load_data[target_data_index][:10])    
        print('check_special_characters_ru ', self.check_special_characters_ru_var.get())
        if self.check_special_characters_ru_var.get() == 1:
            self.load_data[target_data_index] = self.load_data[target_data_index].apply(RemoveStopWords.remove_special_characters_ru)
            print(self.load_data[target_data_index][:10])
        print('check_special_characters_en ', self.check_special_characters_en_var.get())
        if self.check_special_characters_en_var.get() == 1:
            self.load_data[target_data_index] = self.load_data[target_data_index].apply(RemoveStopWords.remove_special_characters_en)
            print(self.load_data[target_data_index][:10])
        print('Tokenize', self.tokenize_var.get())
        if self.tokenize_var.get() == 1:
            self.load_data[target_data_index] = self.load_data[target_data_index].apply(Tokenization.split_method)
            print(self.load_data[target_data_index][:10])
        if self.tokenize_var.get() == 2:
            self.load_data[target_data_index] = self.load_data[target_data_index].apply(Tokenization.nltk_word_tokenize)
            print(self.load_data[target_data_index][:10])
        if self.tokenize_var.get() == 3:
            self.load_data[target_data_index] = self.load_data[target_data_index].apply(Tokenization.nltk_WordPunctTokenizer)
            print(self.load_data[target_data_index][:10])
        if self.tokenize_var.get() == 4:
            self.load_data[target_data_index] = self.load_data[target_data_index].apply(Tokenization.razdel_tokenizer)
            print(self.load_data[target_data_index][:10])
        if self.tokenize_var.get() == 5:
            self.load_data[target_data_index] = self.load_data[target_data_index].apply(Tokenization.nltk_TweetTokenizer)
            print(self.load_data[target_data_index][:10])
        if self.tokenize_var.get() == 6:
            self.load_data[target_data_index] = self.load_data[target_data_index].apply(Tokenization.re_tokenize)
            print(self.load_data[target_data_index][:10])
        print('check_stop_words ', self.check_stop_words_var.get())
        if self.check_stop_words_var.get() == 1:
            self.load_data[target_data_index] = self.load_data[target_data_index].apply(RemoveStopWords.remove_stopwords)
        print('Stemming', self.stemming_var.get())
        if self.stemming_var.get() == 2:
            self.load_data[target_data_index] = self.load_data[target_data_index].apply(Stemming.porterStemmer)
        if self.stemming_var.get() == 3:
            self.load_data[target_data_index] = self.load_data[target_data_index].apply(Stemming.snowBallStemmer)
        if self.stemming_var.get() == 4:
            self.load_data[target_data_index] = self.load_data[target_data_index].apply(Stemming.lancasterStemmer)
        print('Lemmatization', self.lemmatization_var.get())
        if self.lemmatization_var.get() == 2:
            self.load_data[target_data_index] = self.load_data[target_data_index].apply(Lemmatization.wordNetLemmatizer)
        if self.lemmatization_var.get() == 3:
            self.load_data[target_data_index] = self.load_data[target_data_index].apply(Lemmatization.pyMorphy2)
        print('Делаем предложения из отфильтрованных токенов')
        self.load_data[target_data_index] = self.load_data[target_data_index].apply(RemoveStopWords.filtered_text)
        print(self.load_data[target_data_index].loc[1])
        print('Уникальные значения',len(self.load_data[sentiment_data_index].unique()))
        if len(self.load_data[sentiment_data_index].unique()) == 2:
            sentiment_data = SplitAndLabelsData.defineLabels(self.load_data[sentiment_data_index])
        elif len(self.load_data[sentiment_data_index].unique()) == 5:
            sentiment_data = self.load_data[sentiment_data_index].apply(SplitAndLabelsData.get_sentiment_five_label)
        else:
            sentiment_data = self.load_data[sentiment_data_index].apply(SplitAndLabelsData.get_sentiment_many_label)
        print(sentiment_data[:10])
        norm_train_data, norm_test_data, train_sentiments, test_sentiments = SplitAndLabelsData.splitDataset(self.load_data[target_data_index], sentiment_data)
        print('Тренировочные данные',norm_train_data[:10])
        print('Тестовые данные',norm_test_data[:10])
        print('Тренировочные метки',train_sentiments[:10])
        print('Тестовые метки',test_sentiments[:10])
        print('Vectorization', self.vectorazation_var.get())
        if self.vectorazation_var.get() == 1:
            self.vectorizer_model, vec_train_data, vec_test_data = Vectorization.vectorize_BoW(norm_train_data, norm_test_data)
        if self.vectorazation_var.get() == 2:
            self.vectorizer_model, vec_train_data, vec_test_data = Vectorization.vectorize_Tfidf(norm_train_data, norm_test_data)
        print('Тренировочные данные (после векторизации)', vec_train_data[:10])
        print('Тестовые данные (после векторизации)',vec_test_data[:10])
        print('ML method', self.ml_var.get())
        if self.ml_var.get() == 1:
            lr = LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)
            self.preload_model = self.model_train_pattern(lr, vec_train_data, train_sentiments, vec_test_data, test_sentiments)
        if self.ml_var.get() == 2:
            mnb = MultinomialNB()
            self.preload_model = self.model_train_pattern(mnb, vec_train_data, train_sentiments, vec_test_data, test_sentiments)
        if self.ml_var.get() == 3:
            sgdc = SGDClassifier(loss='hinge',max_iter=500,random_state=42)
            self.preload_model = self.model_train_pattern(sgdc, vec_train_data, train_sentiments, vec_test_data, test_sentiments)
        if self.ml_var.get() == 4:
            kneighbors = KNeighborsClassifier(n_neighbors = 1)
            self.preload_model = self.model_train_pattern(kneighbors, vec_train_data, train_sentiments, vec_test_data, test_sentiments)
        if self.ml_var.get() == 5:
            svm = SVC()
            self.preload_model = self.model_train_pattern(svm, vec_train_data, train_sentiments, vec_test_data, test_sentiments)
        if self.ml_var.get() == 6:
            dtc = tree.DecisionTreeClassifier(random_state=0, max_depth=2)
            self.preload_model = self.model_train_pattern(dtc, vec_train_data, train_sentiments, vec_test_data, test_sentiments)
        if self.ml_var.get() == 7:
            rfc = RandomForestClassifier(n_estimators=10)
            self.preload_model = self.model_train_pattern(rfc, vec_train_data, train_sentiments, vec_test_data, test_sentiments)
        if self.ml_var.get() == 8:
            gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                    max_depth=1, random_state=0)
            self.preload_model = self.model_train_pattern(gbc, vec_train_data, train_sentiments, vec_test_data, test_sentiments)
        if self.ml_var.get() == 9:
            mlp = MLPClassifier(solver = 'adam', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
            self.preload_model = self.model_train_pattern(mlp, vec_train_data, train_sentiments, vec_test_data, test_sentiments)
        if (self.preload_model is None):
            return 0
        self.l_load_model.config(text="Модель загружена успешно", foreground='#55A62A')
        print('model OK')


    #Паттерн обучения модели машинного обучения
    def model_train_pattern(self, ml_method, vec_train_data, train_sentiments, vec_test_data, test_sentiments):
        #Обучение модели
        model=ml_method.fit(vec_train_data,train_sentiments)
        print(model)
        #Предсказание модели 
        predict=ml_method.predict(vec_test_data)
        # print(predict)
        #Точность модели
        score=accuracy_score(test_sentiments,predict)
        print("Test score: {0:.2f} %".format(100 * score))
        #Сохранение модели
        filename_model = 'models/'+Path(self.filename).stem+'_model.pkl'
        objects_to_save = (model, self.vectorizer_model, score)
        joblib.dump(objects_to_save, filename_model)
        return model
    
    #Загрузка модели
    def load_model(self):
        filetypes = (("Файл модели", "*.pkl"),
                     ("Любой", "*"))
        model_filename = fd.askopenfilename(title="Выбрать модель", initialdir="models/",
                                      filetypes=filetypes)
        if model_filename:
            print(model_filename)
            model, vectorizer, score = joblib.load(model_filename)
            print('Загруженные данные', model, vectorizer, score)
            self.preload_model = model
            self.vectorizer_model = vectorizer
            self.l_load_model.config(text="Модель загружена успешно", foreground='#55A62A')
            return model
        
    #Классификация текста 
    def classify_text(self):
        text = self.t_input_text.get("1.0", "end")
        print(text)
        if self.check_library_text_blob_var.get() == 1:
            predict_library = DictAndLibrary.textblob_analyse(text)
            print(predict_library)
            if predict_library > 0:
                self.l_output_result.config(text='POSITIVE', foreground='#55A62A')
            elif predict_library < 0:
                self.l_output_result.config(text='NEGATIVE', foreground="#B12C49")
            else:
                self.l_output_result.config(text='Unknown', foreground='#000')
        elif self.check_dict_kartaslovsent_var.get() == 1:
            predict_dict = DictAndLibrary.kartaslovsent_analyse(text)
            print(predict_dict)
            if predict_dict >= 0.5:
                self.l_output_result.config(text='POSITIVE', foreground='#55A62A')
            elif predict_dict <= -0.35:
                self.l_output_result.config(text='NEGATIVE', foreground="#B12C49")
            else:
                self.l_output_result.config(text='Unknown', foreground='#000')
        elif self.preload_model is not None:
            print(self.vectorizer_model)
            text_vectorize = self.vectorizer_model.transform([text])
            print(text_vectorize.shape)
            predict_text = self.preload_model.predict(text_vectorize)
            print(predict_text)
            if predict_text[0] == 0:
                self.l_output_result.config(text='NEGATIVE', foreground="#B12C49")
            elif predict_text[0] == 1:
                self.l_output_result.config(text='POSITIVE', foreground='#55A62A')
            else:
                self.l_output_result.config(text='Unknown', foreground='#000')
        else:
            return mb.showwarning("Уведомление", 'Выберите метод анализа или загрузите модель')
        
if __name__ == "__main__":
    app = App()
    app.mainloop()

